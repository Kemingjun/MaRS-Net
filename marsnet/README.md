# MaRS-Net Model

This directory contains the final MaRS-Net implementation used in the revised manuscript.

MaRS-Net solves the Cooperative Marsupial Robot Scheduling Problem (CMRSP) as an autoregressive constructive policy. The released model uses:

- dual-stream task and robot encoding;
- task-first cooperative decoding;
- carrier-worker pair assignment conditioned on the selected task;
- hard masking of completed tasks to preserve feasibility;
- synchronized state transitions for docking, transport, undocking, task execution, robot current time, and tardiness accumulation.

Use the repository-level scripts for training and evaluation:

```bash
python scripts/train_drl.py --method marsnet --graph_size 20 --run_name marsnet_20
python scripts/eval_drl.py --method marsnet --dataset Synthetic_Dataset --model checkpoints/marsnet/size_20 --decode_strategy greedy --eval_batch_size 1
```

The checkpoint folders under `checkpoints/marsnet/` are compatible with this implementation.
