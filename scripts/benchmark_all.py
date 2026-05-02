import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd):
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return completed.stdout + completed.stderr


def _parse_eval_output(text):
    patterns = {
        "objective": r"Average cost:\s*([0-9.eE+-]+)\s*\+-\s*([0-9.eE+-]+)",
        "distance_tardiness": r"Average distance:\s*([0-9.eE+-]+)\s*tardiness:\s*([0-9.eE+-]+)",
        "runtime": r"Average serial duration:\s*([0-9.eE+-]+)\s*\+-\s*([0-9.eE+-]+)",
    }
    objective = re.search(patterns["objective"], text)
    distance_tardiness = re.search(patterns["distance_tardiness"], text)
    runtime = re.search(patterns["runtime"], text)
    return {
        "objective_mean": float(objective.group(1)) if objective else None,
        "objective_ci95": float(objective.group(2)) if objective else None,
        "distance_mean": float(distance_tardiness.group(1)) if distance_tardiness else None,
        "tardiness_mean": float(distance_tardiness.group(2)) if distance_tardiness else None,
        "runtime_mean_s": float(runtime.group(1)) if runtime else None,
        "runtime_ci95_s": float(runtime.group(2)) if runtime else None,
    }


def _write_markdown(csv_path, md_path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    headers = rows[0].keys()
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DRL checkpoints on one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", default=["marsnet", "hdrl", "tdrl"])
    parser.add_argument("--decode_strategies", nargs="+", default=["greedy", "sample"])
    parser.add_argument("--sample_width", type=int, default=1280)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--out_prefix", default="benchmark_results")
    args = parser.parse_args()

    rows = []
    for method in args.methods:
        checkpoint_root = REPO_ROOT / "checkpoints" / method
        for model_dir in sorted(checkpoint_root.glob("size_*")):
            for strategy in args.decode_strategies:
                cmd = [
                    sys.executable,
                    "scripts/eval_drl.py",
                    "--method",
                    method,
                    "--dataset",
                    args.dataset,
                    "--model",
                    str(model_dir),
                    "--decode_strategy",
                    strategy,
                    "--eval_batch_size",
                    str(args.eval_batch_size),
                ]
                if strategy == "sample":
                    cmd.extend(["--width", str(args.sample_width)])
                output = _run(cmd)
                row = {
                    "method": method,
                    "model": model_dir.name,
                    "decode_strategy": strategy,
                    **_parse_eval_output(output),
                }
                rows.append(row)
                print(row)

    csv_path = REPO_ROOT / f"{args.out_prefix}.csv"
    md_path = REPO_ROOT / f"{args.out_prefix}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(csv_path, md_path)


if __name__ == "__main__":
    main()
