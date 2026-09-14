# MaRS-Net Model

This directory contains the MaRS-Net constructive scheduling policy accompanying the ICRA 2027 submission.

MaRS-Net addresses the Collaborative Carrier-Worker Scheduling Problem (CCWSP) through a structured MDP with composite task and carrier-worker actions. The released model uses:

- dual-stream task and robot encoding;
- task-first cooperative decoding;
- carrier-worker pair assignment conditioned on the selected task;
- hard masking of already scheduled tasks to prevent duplicate assignment;
- synchronized state transitions for docking, fetching, transport, undocking, task execution, robot availability, and tardiness accumulation.

Use the repository-level scripts for training and evaluation:

```bash
python scripts/train_drl.py --method marsnet --graph_size 20 --run_name marsnet_20
python scripts/eval_drl.py --method marsnet --dataset Synthetic_Dataset --model checkpoints/marsnet/size_20 --decode_strategy greedy --eval_batch_size 1
```

The checkpoint folders under `checkpoints/marsnet/` retain their original metadata. See [configuration provenance and experiment coverage](../docs/reproducibility_status.md) for unresolved differences from the manuscript's training settings. Internal names such as `ahasp` and `cur_time` are retained for compatibility; `cur_time` represents robot availability time.
