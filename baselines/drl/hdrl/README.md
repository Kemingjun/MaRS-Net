# Adapted HDRL Baseline

This directory contains the HDRL-based DRL baseline adapted to the Collaborative Carrier-Worker Scheduling Problem (CCWSP).

The implementation preserves the baseline's vehicle-aware dispatch and route-context-aware decoding principle, while replacing the problem-specific state, action, and transition definitions with CCWSP-compatible components:

- task and robot states encode task attributes and the locations and availability times of robots after already scheduled operations;
- the action space is adapted to carrier-worker-task assignment;
- the decoder includes a carrier-worker coupler for feasible cooperative execution;
- the transition model updates synchronized carrier-worker execution states and tardiness.

Use the repository-level scripts for training and evaluation:

```bash
python scripts/train_drl.py --method hdrl --graph_size 20 --run_name hdrl_20
python scripts/eval_drl.py --method hdrl --dataset Synthetic_Dataset --model checkpoints/hdrl/size_20 --decode_strategy greedy --eval_batch_size 1
```
