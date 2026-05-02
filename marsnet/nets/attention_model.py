import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint

from nets.graph_encoder import GraphAttentionEncoder
from nets.robot_encoder import RobotAttentionEncoder
from problems.ahasp.paramet_ahasp import paramet_ahasp
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModel(nn.Module):

    def __init__(
            self,
            embedding_dim,
            hidden_dim,
            problem,
            n_encode_layers=2,
            tanh_clipping=10.,
            mask_inner=True,
            mask_logits=True,
            normalization='batch',
            n_heads=8,
            checkpoint_encoder=False,
            shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        self.norm_query_task = nn.LayerNorm(embedding_dim)
        self.norm_query_robot = nn.LayerNorm(embedding_dim)
        self.norm_glimpse_task = nn.LayerNorm(embedding_dim)
        self.norm_glimpse_robot = nn.LayerNorm(embedding_dim)

        node_dim = 6
        robot_dim = 3

        self.robot_init_linear = nn.Linear(robot_dim, embedding_dim, bias=False)
        self.graph_init_linear = nn.Linear(node_dim, embedding_dim, bias=False)

        self.graph_embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=1,
            normalization=normalization
        )
        self.robot_embedder = RobotAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=1,
            normalization='layer'
        )

        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_robot_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_task_query = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)

        assert embedding_dim % n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out_robot = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.project_robot_query = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)
        self.project_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

        self.empty_robots_embedding = nn.Parameter(torch.empty(embedding_dim))
        nn.init.xavier_uniform_(self.empty_robots_embedding.unsqueeze(0))

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        task_representations = self._precompute_task_representations(input)
        _log_p, pi, cost, distance, tardiness = self._inner(input, task_representations)

        ll = _log_p.sum(1)
        if return_pi:
            return cost, ll, pi

        return cost, ll, distance, tardiness

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def _calc_log_likelihood(self, _log_p, a):
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        return log_p.sum(1)

    def graph_init_embed(self, input):
        max_deadline = input['deadline'].max()
        paramet_ahasp.time_norm = math.ceil(max_deadline)
        node_info = torch.cat((
            input['source'] / paramet_ahasp.location_norm,
            input['destination'] / paramet_ahasp.location_norm,
            input['deadline'][:, :, None] / paramet_ahasp.time_norm,
            input['operation_time'][:, :, None] / paramet_ahasp.time_norm
        ), dim=-1)
        return self.graph_init_linear(node_info)

    def robot_init_embed(self, state):
        assert paramet_ahasp.time_norm is not None, 'paramet_ahasp.time_norm is None'
        assert paramet_ahasp.location_norm is not None, 'paramet_ahasp.location_norm is None'
        robot_init_info = torch.cat((
            state.cur_coord / paramet_ahasp.location_norm,
            state.cur_time.unsqueeze(-1) / paramet_ahasp.time_norm,
        ), -1)
        return self.robot_init_linear(robot_init_info)

    def _precompute_task_representations(self, input):
        task_init_embeddings = self.graph_init_embed(input)

        if self.checkpoint_encoder and self.training:
            task_embeddings, _ = checkpoint(self.graph_embedder, task_init_embeddings, use_reentrant=False)
        else:
            task_embeddings, _ = self.graph_embedder(task_init_embeddings)
        return task_embeddings

    def _get_task_decoder_embeddings(self, node_embeddings, state):
        return node_embeddings

    def _encode_dynamic_context(self, task_representations, state):
        robot_init_embeddings = self.robot_init_embed(state)

        node_embeddings = task_representations
        if self.checkpoint_encoder and self.training:
            robot_embeddings, robot_embd_mean = checkpoint(
                self.robot_embedder, robot_init_embeddings, use_reentrant=False
            )
        else:
            robot_embeddings, robot_embd_mean = self.robot_embedder(robot_init_embeddings)
        return node_embeddings, robot_embeddings, robot_embd_mean

    def _build_env_context(self, node_embeddings, robot_embd_mean):
        env_info = torch.cat((node_embeddings.mean(1), robot_embd_mean), dim=1)
        return self.project_context(env_info)

    def _build_selected_robot_one_hot(self, selected_carrier, selected_worker_local):
        num_carriers = int(paramet_ahasp.ROBOT_NUM_LIST[0].item())
        num_workers = int(paramet_ahasp.ROBOT_NUM_LIST[1].item())

        selected_carrier_one_hot = F.one_hot(selected_carrier, num_classes=num_carriers).float()
        selected_worker_one_hot = F.one_hot(selected_worker_local, num_classes=num_workers).float()
        return torch.cat((selected_carrier_one_hot, selected_worker_one_hot), dim=-1)

    def _decode_task_first_step(self, node_embeddings, robot_embeddings, robot_embd_mean, state):
        task_decoder_embeddings = self._get_task_decoder_embeddings(node_embeddings, state)
        env_context = self._build_env_context(task_decoder_embeddings, robot_embd_mean)

        task_log_p, mask = self._get_log_p_task(
            task_decoder_embeddings,
            task_decoder_embeddings.mean(1),
            robot_embd_mean,
            state
        )
        selected_task = self._select_node(task_log_p.exp()[:, 0, :], mask[:, 0, :])
        selected_task_log_p = torch.gather(
            task_log_p.squeeze(1),
            dim=1,
            index=selected_task.unsqueeze(-1)
        )

        selected_task_embedding = task_decoder_embeddings[self.batch_idx, selected_task]
        selected_robot_one_hot, robot_selected, robot_log_p, _ = self.robot_decision(
            robot_embeddings,
            env_context + selected_task_embedding
        )
        selected_robot_log_p = torch.gather(
            robot_log_p.squeeze(1),
            dim=1,
            index=robot_selected
        )

        log_p = selected_task_log_p + selected_robot_log_p.sum(dim=1).unsqueeze(1)
        action = torch.cat([selected_task.unsqueeze(-1), robot_selected], dim=-1)
        return log_p, action, selected_task, selected_robot_one_hot

    def _inner(self, input, task_representations):
        outputs = []
        sequences = []

        state = self.problem.make_state(input)
        self.batch_idx = torch.arange(task_representations.size(0), device=task_representations.device)

        while not state.all_finished():
            node_embeddings, robot_embeddings, robot_embd_mean = self._encode_dynamic_context(
                task_representations, state
            )
            log_p, action, selected_task, selected_robot_one_hot = self._decode_task_first_step(
                node_embeddings, robot_embeddings, robot_embd_mean, state
            )

            state = state.update(selected_task, selected_robot_one_hot)

            outputs.append(log_p)
            sequences.append(action)

        cost = state.length.unsqueeze(-1) * paramet_ahasp.WEIGHT + state.tardiness * (1 - paramet_ahasp.WEIGHT)
        return torch.stack(outputs, 1), torch.stack(sequences, 1), cost, state.length.unsqueeze(-1), state.tardiness

    def sample_many(self, _input, batch_rep=1, iter_rep=1):
        task_representations = self._precompute_task_representations(_input)
        return sample_many(
            lambda input: self._inner(*input),
            (_input, task_representations),
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _select_robot(self, probs):
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _get_log_p_task(self, node_embeddings, node_embeddings_mean, robot_embd_mean, state, normalize=True):
        query_info = torch.cat((node_embeddings_mean, robot_embd_mean), dim=-1)
        query = self.project_task_query(query_info)[:, None, :]
        query = self.norm_query_task(query + node_embeddings_mean.unsqueeze(1))

        glimpse_K, glimpse_V, logit_K = self.project_node_embeddings(node_embeddings[:, None, :, :]).chunk(3, dim=-1)
        glimpse_K = self._make_heads(glimpse_K, 1)
        glimpse_V = self._make_heads(glimpse_V, 1)
        mask = state.get_mask()

        log_p, _ = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, mask

    def robot_decision(self, robot_embeddings, env_context):
        selected_robots_embedding_list = []
        selected_robot_one_hot = None
        robot_log_p = None
        robot_selected = None

        for robot_type_index in range(paramet_ahasp.ROBOT_TYPE_NUM):
            robot_type_log_p, robot_type_embeddings = self._get_log_p_robot(
                robot_embeddings, env_context, robot_type_index, selected_robots_embedding_list
            )
            selected_type_robot = self._select_robot(robot_type_log_p.exp()[:, 0, :])

            selected_robot_type_embedding = robot_type_embeddings[self.batch_idx, selected_type_robot]
            selected_robots_embedding_list.append(selected_robot_type_embedding[:, None, :])

            selected_type_robot_one_hot = F.one_hot(
                selected_type_robot,
                num_classes=paramet_ahasp.ROBOT_NUM_LIST[robot_type_index]
            ).float()
            if selected_robot_one_hot is None:
                selected_robot_one_hot = selected_type_robot_one_hot
            else:
                selected_robot_one_hot = torch.cat([selected_robot_one_hot, selected_type_robot_one_hot], -1)

            if robot_log_p is None:
                robot_log_p = robot_type_log_p
            else:
                robot_log_p = torch.cat([robot_log_p, robot_type_log_p], -1)

            if robot_selected is None:
                robot_selected = selected_type_robot.unsqueeze(-1)
            else:
                offset = paramet_ahasp.ROBOT_NUM_LIST[:robot_type_index].sum()
                robot_selected = torch.cat([robot_selected, (selected_type_robot + offset).unsqueeze(-1)], -1)

        return selected_robot_one_hot, robot_selected, robot_log_p, selected_robots_embedding_list

    def _get_log_p_robot(self, robot_embeddings, env_context, robot_type_index, selected_robots_embedding_list,
                         normalize=True):
        start_index = paramet_ahasp.ROBOT_NUM_LIST[:robot_type_index].sum()
        end_index = start_index + paramet_ahasp.ROBOT_NUM_LIST[robot_type_index]
        robot_type_embedding = robot_embeddings[:, start_index:end_index, :]

        if len(selected_robots_embedding_list) == 0:
            selected_robots_embedding_mean = self.empty_robots_embedding.expand(robot_type_embedding.size(0), -1)
        else:
            selected_robots_embedding = torch.cat(selected_robots_embedding_list, dim=1)
            selected_robots_embedding_mean = selected_robots_embedding.mean(dim=1)

        context_info = torch.cat(
            (env_context, robot_type_embedding.mean(dim=1), selected_robots_embedding_mean),
            dim=-1
        )

        query = self.project_robot_query(context_info).unsqueeze(1)
        query = self.norm_query_robot(query + robot_type_embedding.mean(dim=1).unsqueeze(1))

        glimpse_K_, glimpse_V_, logit_K_ = self.project_robot_embeddings(robot_type_embedding[:, None, :, :]).chunk(
            3, dim=-1
        )
        glimpse_K = self._make_heads(glimpse_K_, 1)
        glimpse_V = self._make_heads(glimpse_V_, 1)
        logit_K = logit_K_.contiguous()

        log_p, _ = self._one_to_many_logits_robot(query, glimpse_K, glimpse_V, logit_K)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, robot_type_embedding

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )
        glimpse = self.norm_glimpse_task(glimpse + query.unsqueeze(1))
        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _one_to_many_logits_robot(self, query, glimpse_K, glimpse_V, logit_K):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = self.project_out_robot(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )
        glimpse = self.norm_glimpse_robot(glimpse + query.unsqueeze(1))
        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        return logits, glimpse.squeeze(-2)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )
