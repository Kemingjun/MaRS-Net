import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_model, move_to

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark all trained models on a dataset and summarize greedy/sample results."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Directory that contains trained model folders."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Synthetic_Dataset/size_20_uniform",
        help="Dataset folder under Instance/, or an absolute path inside Instance/."
    )
    parser.add_argument(
        "--sample_width",
        type=int,
        default=1280,
        help="Number of samples used for sampling-based evaluation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size. Default 1 gives instance-level solve time."
    )
    parser.add_argument(
        "--max_calc_batch_size",
        type=int,
        default=10000,
        help="Maximum effective sample batch used inside model.sample_many."
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for sampling."
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA."
    )
    parser.add_argument(
        "--precision",
        choices=["32", "bf16"],
        default="32",
        help="Evaluation precision."
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Optional output file prefix. Defaults to benchmark_<dataset_name>_sample<width>."
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Disable progress bar."
    )
    return parser.parse_args()


def normalize_dataset_arg(dataset_arg: str, repo_root: Path) -> str:
    dataset_path = Path(dataset_arg)
    instance_root = repo_root / "Instance"
    if not instance_root.exists():
        instance_root = repo_root.parent / "Instance"
    if dataset_path.exists():
        try:
            relative = dataset_path.resolve().relative_to(instance_root.resolve())
            return relative.as_posix()
        except ValueError:
            raise ValueError(f"Dataset path must be inside {instance_root}, got: {dataset_arg}")
    candidate = instance_root / dataset_path
    if candidate.exists():
        return candidate.relative_to(instance_root).as_posix()
    return dataset_arg.replace("\\", "/")


def discover_model_dirs(outputs_dir: Path):
    model_dirs = []
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "args.json").exists():
            continue
        if not any(p.suffix == ".pt" for p in child.iterdir()):
            continue
        model_dirs.append(child)
    return model_dirs


def sequence_to_list(problem_name, seq):
    if problem_name in ("hrsp", "ahasp"):
        return seq.tolist()
    raise AssertionError(f"Unknown problem: {problem_name}")


def evaluate_best_of_k(model, batch, width, max_calc_batch_size, precision):
    batch_size = next(iter(batch.values())).size(0)
    max_batch_rep = max(1, max_calc_batch_size // batch_size)

    best_seq = None
    best_cost = None
    best_distance = None
    best_tardiness = None
    remaining = width

    while remaining > 0:
        chunk_width = min(remaining, max_batch_rep)
        if precision == "32":
            seq, cost, distance, tardiness = model.sample_many(batch, batch_rep=chunk_width, iter_rep=1)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                seq, cost, distance, tardiness = model.sample_many(batch, batch_rep=chunk_width, iter_rep=1)

        if best_cost is None:
            best_seq = seq
            best_cost = cost
            best_distance = distance
            best_tardiness = tardiness
        else:
            improved = cost < best_cost
            best_cost = torch.where(improved, cost, best_cost)
            best_distance = torch.where(improved, distance, best_distance)
            best_tardiness = torch.where(improved, tardiness, best_tardiness)
            best_seq = torch.where(improved.unsqueeze(-1), seq, best_seq)

        remaining -= chunk_width

    return best_seq, best_cost, best_distance, best_tardiness


def evaluate_model(model, dataset, device, decode_strategy, sample_width, eval_batch_size, max_calc_batch_size,
                   precision, no_progress_bar):
    model.to(device)
    model.eval()
    model.set_decode_type("greedy" if decode_strategy == "greedy" else "sampling", temp=1.0)

    dataloader = DataLoader(dataset, batch_size=eval_batch_size)
    process = psutil.Process(os.getpid()) if psutil is not None else None
    cpu_peak_rss_mb = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    results = []
    costs = []
    distances = []
    tardinesses = []
    durations = []

    for batch in dataloader:
        batch = move_to(batch, device)

        if process is not None:
            cpu_peak_rss_mb = max(cpu_peak_rss_mb, process.memory_info().rss / (1024 ** 2))

        start = time.perf_counter()
        with torch.no_grad():
            if decode_strategy == "greedy":
                if precision == "32":
                    seq, cost, distance, tardiness = model.sample_many(batch, batch_rep=1, iter_rep=1)
                else:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        seq, cost, distance, tardiness = model.sample_many(batch, batch_rep=1, iter_rep=1)
            else:
                seq, cost, distance, tardiness = evaluate_best_of_k(
                    model, batch, sample_width, max_calc_batch_size, precision
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - start

        seq = seq.detach().cpu().numpy()
        cost = cost.detach().cpu().reshape(-1).numpy()
        distance = distance.detach().cpu().reshape(-1).numpy()
        tardiness = tardiness.detach().cpu().reshape(-1).numpy()

        for i in range(len(cost)):
            route = sequence_to_list(model.problem.NAME, seq[i])
            results.append((float(cost[i]), route, float(duration)))
            costs.append(float(cost[i]))
            distances.append(float(distance[i]))
            tardinesses.append(float(tardiness[i]))
            durations.append(float(duration))

        if process is not None:
            cpu_peak_rss_mb = max(cpu_peak_rss_mb, process.memory_info().rss / (1024 ** 2))

    gpu_peak_alloc_mb = 0.0
    gpu_peak_reserved_mb = 0.0
    if device.type == "cuda":
        gpu_peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        gpu_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    return {
        "results": results,
        "costs": costs,
        "distances": distances,
        "tardinesses": tardinesses,
        "durations": durations,
        "gpu_peak_alloc_mb": gpu_peak_alloc_mb,
        "gpu_peak_reserved_mb": gpu_peak_reserved_mb,
        "cpu_peak_rss_mb": cpu_peak_rss_mb,
    }


def build_summary_row(model_dir: Path, decode_strategy: str, sample_width: int, metrics: dict):
    costs = np.array(metrics["costs"], dtype=np.float64)
    distances = np.array(metrics["distances"], dtype=np.float64)
    tardinesses = np.array(metrics["tardinesses"], dtype=np.float64)
    durations = np.array(metrics["durations"], dtype=np.float64)
    n = len(costs)

    return {
        "model": model_dir.name,
        "decode": decode_strategy if decode_strategy == "greedy" else f"sample{sample_width}",
        "num_instances": n,
        "avg_cost_mean": float(costs.mean()),
        "avg_cost_std": float(costs.std(ddof=0)),
        "avg_distance_mean": float(distances.mean()),
        "avg_distance_std": float(distances.std(ddof=0)),
        "avg_tardiness_mean": float(tardinesses.mean()),
        "avg_tardiness_std": float(tardinesses.std(ddof=0)),
        "solve_time_mean_s": float(durations.mean()),
        "solve_time_std_s": float(durations.std(ddof=0)),
        "total_solve_time_s": float(durations.sum()),
        "gpu_peak_alloc_mb": float(metrics["gpu_peak_alloc_mb"]),
        "gpu_peak_reserved_mb": float(metrics["gpu_peak_reserved_mb"]),
        "cpu_peak_rss_mb": float(metrics["cpu_peak_rss_mb"]),
    }


def write_csv(rows, path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path: Path):
    headers = [
        "model", "decode", "avg_cost_mean", "avg_cost_std", "avg_distance_mean", "avg_distance_std",
        "avg_tardiness_mean", "avg_tardiness_std", "solve_time_mean_s", "solve_time_std_s",
        "gpu_peak_alloc_mb", "gpu_peak_reserved_mb", "cpu_peak_rss_mb"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    opts = parse_args()
    repo_root = Path(__file__).resolve().parent
    outputs_dir = (repo_root / opts.outputs_dir).resolve()
    dataset_arg = normalize_dataset_arg(opts.dataset, repo_root)

    model_dirs = discover_model_dirs(outputs_dir)
    if not model_dirs:
        raise RuntimeError(f"No trained model folders found in {outputs_dir}")

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataset_name = Path(dataset_arg).name
    out_prefix = opts.out_prefix or f"benchmark_{dataset_name}_sample{opts.sample_width}"
    summary_rows = []

    for model_dir in model_dirs:
        print(f"[INFO] Evaluating {model_dir.name}")
        model, _ = load_model(str(model_dir))
        dataset = model.problem.make_dataset(filename=dataset_arg)

        for decode_strategy in ("greedy", "sample"):
            print(f"  - {decode_strategy}")
            metrics = evaluate_model(
                model=model,
                dataset=dataset,
                device=device,
                decode_strategy=decode_strategy,
                sample_width=opts.sample_width,
                eval_batch_size=opts.eval_batch_size,
                max_calc_batch_size=opts.max_calc_batch_size,
                precision=opts.precision,
                no_progress_bar=opts.no_progress_bar,
            )
            summary_rows.append(
                build_summary_row(model_dir, decode_strategy, opts.sample_width, metrics)
            )

        del model
        del dataset
        if use_cuda:
            torch.cuda.empty_cache()

    summary_rows.sort(key=lambda row: (row["decode"], row["avg_cost_mean"]))

    csv_path = repo_root / "results" / f"{out_prefix}.csv"
    md_path = repo_root / "results" / f"{out_prefix}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, csv_path)
    write_markdown(summary_rows, md_path)

    print(f"[DONE] CSV saved to: {csv_path}")
    print(f"[DONE] Markdown saved to: {md_path}")
    for row in summary_rows:
        print(
            f"{row['model']:>20} | {row['decode']:<10} | "
            f"cost {row['avg_cost_mean']:.4f} +/- {row['avg_cost_std']:.4f} | "
            f"dist {row['avg_distance_mean']:.4f} +/- {row['avg_distance_std']:.4f} | "
            f"tard {row['avg_tardiness_mean']:.4f} +/- {row['avg_tardiness_std']:.4f} | "
            f"time {row['solve_time_mean_s']:.4f}s +/- {row['solve_time_std_s']:.4f}s | "
            f"gpu_peak {row['gpu_peak_alloc_mb']:.1f} MB"
        )
    return

    print(f"[DONE] CSV saved to: {csv_path}")
    print(f"[DONE] Markdown saved to: {md_path}")
    for row in summary_rows:
        print(
            f"{row['model']:>20} | {row['decode']:<10} | "
            f"cost {row['avg_cost_mean']:.4f} ± {row['avg_cost_std']:.4f} | "
            f"dist {row['avg_distance_mean']:.4f} ± {row['avg_distance_std']:.4f} | "
            f"tard {row['avg_tardiness_mean']:.4f} ± {row['avg_tardiness_std']:.4f} | "
            f"time {row['solve_time_mean_s']:.4f}s ± {row['solve_time_std_s']:.4f}s | "
            f"gpu_peak {row['gpu_peak_alloc_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()
