# Adapted TDRL Baseline

This directory contains the TDRL-based DRL baseline adapted to CMRSP.

The implementation preserves the baseline's token-style state coding and GRU-based dynamic token update mechanism, while adapting the problem-specific components to synchronized marsupial robot scheduling:

- task, robot, and environment tokens encode CMRSP states;
- robot and environment tokens are updated according to synchronized cooperative execution;
- the action space is adapted to carrier-worker-task assignment;
- the decoder includes a carrier-worker coupler for feasible cooperative execution.

Use the repository-level scripts for training and evaluation:

```bash
python scripts/train_drl.py --method tdrl --graph_size 20 --run_name tdrl_20
python scripts/eval_drl.py --method tdrl --dataset Synthetic_Dataset --model checkpoints/tdrl/size_20 --decode_strategy greedy --eval_batch_size 1
```
