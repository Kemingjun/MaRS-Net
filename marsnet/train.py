import os
import time
import csv
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

try:
    import psutil
except ImportError:
    psutil = None


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def get_memory_stats(device):
    stats = {}

    if psutil is not None:
        process = psutil.Process(os.getpid())
        stats['cpu_rss_mb'] = process.memory_info().rss / (1024 ** 2)

    if isinstance(device, torch.device) and device.type == 'cuda' and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        stats['gpu_alloc_mb'] = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
        stats['gpu_reserved_mb'] = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
        stats['gpu_peak_mb'] = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)

    return stats


def append_memory_log(save_dir, row):
    memory_log_path = os.path.join(save_dir, 'memory_log.csv')
    file_exists = os.path.exists(memory_log_path)

    with open(memory_log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, _, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if opts.use_cuda:
        torch.cuda.reset_peak_memory_stats(opts.device)

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()  
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    epoch_memory_stats = get_memory_stats(opts.device)
    if 'gpu_peak_mb' in epoch_memory_stats:
        print("Epoch {} peak gpu memory: {:.2f} MB".format(epoch, epoch_memory_stats['gpu_peak_mb']))
    append_memory_log(opts.save_dir, {
        'record_type': 'epoch_end',
        'epoch': epoch,
        'step': step,
        **epoch_memory_stats
    })
    if not opts.no_tensorboard:
        if 'gpu_peak_mb' in epoch_memory_stats:
            tb_logger.log_value('epoch_gpu_peak_mb', epoch_memory_stats['gpu_peak_mb'], step)
        if 'cpu_rss_mb' in epoch_memory_stats:
            tb_logger.log_value('epoch_cpu_rss_mb', epoch_memory_stats['cpu_rss_mb'], step)

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)   
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, distance, tardiness = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()

    max_norm_threshold = 1e5  

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            if grad_norm > max_norm_threshold:
                print(f"[Large gradient warning] parameter: {name}, gradient norm: {grad_norm:.2e}")
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        memory_stats = get_memory_stats(opts.device)
        log_values(cost, distance, tardiness, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, memory_stats=memory_stats)
        append_memory_log(opts.save_dir, {
            'record_type': 'train_step',
            'epoch': epoch,
            'batch_id': batch_id,
            'step': step,
            **memory_stats
        })
