# Reproducibility Commands

## Static Check

```bash
python -m compileall marsnet baselines scripts
```

## MaRS-Net Evaluation

```bash
python scripts/eval_drl.py --method marsnet --dataset Instance/uniform_100_per_scale_20260413/size_20_uniform --model checkpoints/marsnet/size_20 --decode_strategy greedy --eval_batch_size 1
python scripts/eval_drl.py --method marsnet --dataset Instance/uniform_100_per_scale_20260413/size_20_uniform --model checkpoints/marsnet/size_20 --decode_strategy sample --width 1280 --eval_batch_size 1
```

## DRL Baseline Evaluation

```bash
python scripts/eval_drl.py --method hdrl --dataset Instance/uniform_100_per_scale_20260413/size_20_uniform --model checkpoints/hdrl/size_20 --decode_strategy greedy --eval_batch_size 1
python scripts/eval_drl.py --method tdrl --dataset Instance/uniform_100_per_scale_20260413/size_20_uniform --model checkpoints/tdrl/size_20 --decode_strategy greedy --eval_batch_size 1
```

## DRL Training Smoke Test

```bash
python scripts/train_drl.py --method marsnet --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_marsnet
python scripts/train_drl.py --method hdrl --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_hdrl
python scripts/train_drl.py --method tdrl --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_tdrl
```

## Conventional Baselines

```bash
python scripts/run_conventional.py --solver or_tool --instance Instance/uniform_100_per_scale_20260413/size_10_uniform/T10_I1_uniform.xlsx --time_limit 60
python scripts/run_conventional.py --solver ALNS --instance Instance/uniform_100_per_scale_20260413/size_10_uniform/T10_I1_uniform.xlsx --max_iterations 100 --seed 1234
```

## Batch Benchmark

```bash
python scripts/benchmark_all.py --dataset Instance/uniform_100_per_scale_20260413/size_20_uniform --methods marsnet hdrl tdrl --decode_strategies greedy sample --sample_width 1280 --eval_batch_size 1 --out_prefix final_size20_uniform
```
