# Reproducibility Commands

Run these examples from the repository root after activating the `marsnet` environment described in the main README. They evaluate the released weights or exercise the training entrypoints.

## Static Check

```bash
python -m compileall marsnet baselines scripts
```

## MaRS-Net Evaluation

```bash
python scripts/eval_drl.py --method marsnet --dataset Synthetic_Dataset --model checkpoints/marsnet/size_20 --decode_strategy greedy --eval_batch_size 1
python scripts/eval_drl.py --method marsnet --dataset Synthetic_Dataset --model checkpoints/marsnet/size_20 --decode_strategy sample --width 1280 --eval_batch_size 1
```

## DRL Baseline Evaluation

```bash
python scripts/eval_drl.py --method hdrl --dataset Synthetic_Dataset --model checkpoints/hdrl/size_20 --decode_strategy greedy --eval_batch_size 1
python scripts/eval_drl.py --method tdrl --dataset Synthetic_Dataset --model checkpoints/tdrl/size_20 --decode_strategy greedy --eval_batch_size 1
```

## DRL Training Smoke Test

These short runs test the training entrypoints using the explicitly supplied arguments and otherwise default settings.

```bash
python scripts/train_drl.py --method marsnet --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_marsnet
python scripts/train_drl.py --method hdrl --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_hdrl
python scripts/train_drl.py --method tdrl --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_tdrl
```

## Conventional Baselines

The commands below are short usage examples. For the manuscript's stopping limits, use `--max_iterations 100 --time_limit 3600` for a metaheuristic and `--time_limit 3600` for a solver.

```bash
python scripts/run_conventional.py --solver or_tool --instance Synthetic_Dataset/size_10_uniform/T10_I1_uniform.xlsx --time_limit 60
python scripts/run_conventional.py --solver ALNS --instance Synthetic_Dataset/size_10_uniform/T10_I1_uniform.xlsx --max_iterations 100 --seed 1234
```

## Batch Benchmark

```bash
python scripts/benchmark_all.py --dataset Synthetic_Dataset --methods marsnet hdrl tdrl --decode_strategies greedy sample --sample_width 1280 --eval_batch_size 1 --sizes 10 20 40 60 100 --out_prefix final_synthetic
```

## Industrial Evaluation

```bash
python scripts/benchmark_all.py --dataset Industrial_Dataset --methods marsnet hdrl tdrl --decode_strategies sample --sample_width 1280 --eval_batch_size 1 --sizes 10 20 30 40 --out_prefix industrial_drl
```

This evaluates the three released DRL methods. Gurobi and ALNS are run separately with `scripts/run_conventional.py`; the DRL batch script does not merge their results.

## Cross-Scale Evaluation

Dataset family shortcuts infer the test size from the checkpoint directory. For cross-scale testing, specify the test directory explicitly. For example, this evaluates a model trained with 40 tasks on 60-task instances:

```bash
python scripts/eval_drl.py --method marsnet --dataset Instance/Synthetic_Dataset/size_60_uniform --model checkpoints/marsnet/size_40 --decode_strategy sample --width 1280 --eval_batch_size 1
```

Use the corresponding `hdrl` or `tdrl` method and checkpoint directory for the baselines. The 30-task synthetic dataset is not included; 30-task industrial data is available.
