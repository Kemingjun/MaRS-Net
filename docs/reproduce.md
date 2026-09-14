# Reproducibility Commands

Run these examples from the repository root after activating the `marsnet` environment described in the main README. They evaluate the released weights or exercise the training entrypoints. They do not establish that every manuscript experiment is reproduced; see [experiment coverage and configuration provenance](reproducibility_status.md).

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

These short runs test execution only. They use the implementation defaults except for the explicitly supplied arguments and are not the training protocol for the manuscript tables. In particular, the training defaults use batch size 512 and seed 1234; published checkpoint metadata records additional configurations.

```bash
python scripts/train_drl.py --method marsnet --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_marsnet
python scripts/train_drl.py --method hdrl --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_hdrl
python scripts/train_drl.py --method tdrl --graph_size 20 --n_epochs 1 --epoch_size 512 --batch_size 128 --run_name smoke_tdrl
```

## Conventional Baselines

The commands below are short usage examples. For the manuscript's stopping limits, use `--max_iterations 100 --time_limit 3600` for a metaheuristic and `--time_limit 3600` for a solver. Repeated-run selection and the source of aggregate results must also be specified before reproducing a table.

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

## Statistical Interpretation

- The Excel-directory loader reads every `.xlsx` file in the directory; `--val_size` and `--offset` do not subset this input mode. For a one-instance loading test, pass an explicit file such as `Instance/Synthetic_Dataset/size_20_uniform/T20_I1_uniform.xlsx`. This behavior differs from the pickle input mode.
- In the released `eval.py` scripts, `Average cost: mean +- h` uses `h = 2 * np.std(costs) / sqrt(N)` with NumPy's default `ddof=0`. It is approximately two standard errors, not the standard deviation across instances and not an exact 95% confidence interval.
- `benchmark_all.py` retains this value in the historical `objective_ci95` column. The column name does not change the estimator. For the manuscript's across-instance Std, calculate the standard deviation from the individual objective values and state the `ddof` convention. Do not copy the reported `+-` value into the Std column.
- Saved evaluation results contain `(results, parallelism)`; each entry in `results` contains `(cost, sequence, duration)`. Distance and tardiness are printed as means, not saved as per-instance values by this interface. A full per-instance metrics table requires the experiment-specific records.
- With `--eval_batch_size 1`, reported serial durations correspond to single-instance evaluation calls. At larger batches the batch duration is repeated for its instances; it is not single-instance latency. Warm-up, device, batching, and sampling budget must be reported for time comparisons. These scripts are not a standardized latency benchmark.
- Main comparison tables use per-instance RPD against the best objective among all compared methods, then aggregate over instances. Ablation gaps use Main as the reference. These are different quantities; a zero RPD does not certify optimality. The DRL-only batch script does not compute the full comparison RPD.
- Across-instance standard deviation and variability across independently trained seeds must be reported separately. The release contains one checkpoint per method and scale, not five independently trained models per scale.
