import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from solver_adapters import EXACT_SOLVERS, META_SOLVERS, PROJECT_ROOT, INSTANCE_ROOT, run_solver_task


RAW_COLUMNS = [
    "instance_id",
    "instance_path",
    "size",
    "solver_family",
    "solver_name",
    "rep",
    "status",
    "fitness",
    "distance",
    "tardiness",
    "runtime_s",
    "is_optimal_proven",
    "best_bound",
    "gap",
    "error_message",
    "started_at",
    "finished_at",
    "rpd",
    "reference_fitness",
    "reference_is_proven_optimum",
]

EXACT_SOLVER_SIZE_LIMITS = {
    "gurobi": {10, 20},
    "or_tool": {10, 20, 40},
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _safe_std(series: pd.Series) -> float:
    value = series.std()
    return float(value) if pd.notna(value) else math.nan


def _to_bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"} if pd.notna(value) else False)


def _make_seed(base_seed: int, instance_id: str, solver_name: str, rep: int) -> int:
    key = f"{base_seed}:{instance_id}:{solver_name}:{rep}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16) % 2147483647


def _parse_size(folder_name: str) -> int:
    return int(folder_name.split("_")[1])


def discover_instances(instances_root: Path, sizes: Iterable[int] = None, max_instances_per_size: int = None) -> List[Dict[str, Any]]:
    selected_sizes = set(sizes) if sizes else None
    instances: List[Dict[str, Any]] = []
    for size_dir in sorted(instances_root.glob("size_*_uniform")):
        size = _parse_size(size_dir.name)
        if selected_sizes and size not in selected_sizes:
            continue
        files = sorted(size_dir.glob("*.xlsx"))
        if max_instances_per_size is not None:
            files = files[:max_instances_per_size]
        for path in files:
            rel_path = path.relative_to(INSTANCE_ROOT).as_posix()
            instances.append(
                {
                    "instance_id": path.stem,
                    "instance_abs_path": str(path.resolve()),
                    "instance_rel_path": rel_path,
                    "size": size,
                }
            )
    return instances


def _registry_template(instance_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instance_id": instance_meta["instance_id"],
        "instance_path": instance_meta["instance_rel_path"],
        "size": instance_meta["size"],
        "current_best_fitness": None,
        "current_best_solver": None,
        "current_best_solver_family": None,
        "current_best_rep": None,
        "is_optimal_proven": False,
        "optimal_fitness": None,
        "optimal_proved_by": None,
        "optimal_proved_at": None,
        "exact_runs_completed": 0,
        "meta_runs_completed": 0,
    }


def load_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_registry(registry_path: Path, registry: Dict[str, Dict[str, Any]]) -> None:
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def load_raw_runs(raw_runs_path: Path) -> pd.DataFrame:
    if raw_runs_path.exists():
        return pd.read_csv(raw_runs_path)
    return pd.DataFrame(columns=RAW_COLUMNS)


def append_raw_run(raw_runs_path: Path, row: Dict[str, Any]) -> None:
    raw_runs_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not raw_runs_path.exists()
    with raw_runs_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in RAW_COLUMNS})


def completed_keys(raw_df: pd.DataFrame) -> set:
    if raw_df.empty:
        return set()
    return {
        (str(row.instance_id), str(row.solver_name), int(row.rep))
        for row in raw_df.itertuples(index=False)
    }


def rebuild_registry_counts(registry: Dict[str, Dict[str, Any]], raw_df: pd.DataFrame) -> None:
    for item in registry.values():
        item["exact_runs_completed"] = 0
        item["meta_runs_completed"] = 0
    if raw_df.empty:
        return
    for row in raw_df.itertuples(index=False):
        record = registry.get(str(row.instance_id))
        if record is None:
            continue
        if row.solver_family == "exact":
            record["exact_runs_completed"] += 1
        else:
            record["meta_runs_completed"] += 1


def update_registry_with_result(registry: Dict[str, Dict[str, Any]], result: Dict[str, Any]) -> None:
    record = registry[result["instance_id"]]
    if result["solver_family"] == "exact":
        record["exact_runs_completed"] += 1
    else:
        record["meta_runs_completed"] += 1

    fitness = result.get("fitness")
    if fitness is not None and pd.notna(fitness):
        if record["current_best_fitness"] is None or fitness < record["current_best_fitness"] - 1e-9:
            record["current_best_fitness"] = float(fitness)
            record["current_best_solver"] = result["solver_name"]
            record["current_best_solver_family"] = result["solver_family"]
            record["current_best_rep"] = int(result["rep"])

    if result["solver_family"] == "exact" and result.get("is_optimal_proven") and fitness is not None and pd.notna(fitness):
        if (not record["is_optimal_proven"]) or record["optimal_fitness"] is None or fitness < record["optimal_fitness"] - 1e-9:
            record["is_optimal_proven"] = True
            record["optimal_fitness"] = float(fitness)
            record["optimal_proved_by"] = result["solver_name"]
            record["optimal_proved_at"] = result["finished_at"]


def build_tasks(
    instances: List[Dict[str, Any]],
    skip_keys: set,
    time_limit_s: int,
    max_iterations: int,
    meta_reps: int,
    base_seed: int,
    num_search_workers: int,
    run_group: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    exact_tasks = []
    meta_tasks = []
    allow_exact = run_group in {"all", "gurobi", "or_tool"}
    allow_meta = run_group in {"all", "metaheuristic"}
    for instance in instances:
        if allow_exact:
            for solver_name in EXACT_SOLVERS:
                if run_group == "gurobi" and solver_name != "gurobi":
                    continue
                if run_group == "or_tool" and solver_name != "or_tool":
                    continue
                allowed_sizes = EXACT_SOLVER_SIZE_LIMITS.get(solver_name)
                if allowed_sizes is not None and instance["size"] not in allowed_sizes:
                    continue
                key = (instance["instance_id"], solver_name, 1)
                if key in skip_keys:
                    continue
                exact_tasks.append(
                    {
                        **instance,
                        "solver_family": "exact",
                        "solver_name": solver_name,
                        "rep": 1,
                        "time_limit_s": time_limit_s,
                        "max_iterations": max_iterations,
                        "seed": _make_seed(base_seed, instance["instance_id"], solver_name, 1),
                        "num_search_workers": num_search_workers,
                    }
                )
        if allow_meta:
            for solver_name in META_SOLVERS:
                for rep in range(1, meta_reps + 1):
                    key = (instance["instance_id"], solver_name, rep)
                    if key in skip_keys:
                        continue
                    meta_tasks.append(
                        {
                            **instance,
                            "solver_family": "meta",
                            "solver_name": solver_name,
                            "rep": rep,
                            "time_limit_s": time_limit_s,
                            "max_iterations": max_iterations,
                            "seed": _make_seed(base_seed, instance["instance_id"], solver_name, rep),
                            "num_search_workers": num_search_workers,
                        }
                    )
    return exact_tasks, meta_tasks


def _submit_until_full(
    executor: ProcessPoolExecutor,
    pending: Dict[Any, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    cursor: int,
    max_workers: int,
) -> int:
    while cursor < len(tasks) and len(pending) < max_workers:
        task = tasks[cursor]
        future = executor.submit(run_solver_task, task)
        pending[future] = task
        cursor += 1
    return cursor


def finalize_results(results_dir: Path, registry: Dict[str, Dict[str, Any]]) -> None:
    raw_runs_path = results_dir / "raw_runs.csv"
    raw_df = load_raw_runs(raw_runs_path)
    if raw_df.empty:
        pd.DataFrame(columns=RAW_COLUMNS).to_csv(results_dir / "instance_summary.csv", index=False)
        pd.DataFrame().to_csv(results_dir / "method_summary.csv", index=False)
        return

    numeric_cols = ["size", "rep", "fitness", "distance", "tardiness", "runtime_s", "best_bound", "gap"]
    for col in numeric_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    raw_df["is_optimal_proven"] = _to_bool_series(raw_df["is_optimal_proven"])

    raw_df["reference_fitness"] = raw_df["instance_id"].map(
        lambda iid: registry[str(iid)]["optimal_fitness"] if registry[str(iid)]["is_optimal_proven"] else registry[str(iid)]["current_best_fitness"]
    )
    raw_df["reference_is_proven_optimum"] = raw_df["instance_id"].map(lambda iid: registry[str(iid)]["is_optimal_proven"])

    valid_mask = raw_df["fitness"].notna() & raw_df["reference_fitness"].notna() & (raw_df["reference_fitness"] != 0)
    raw_df["rpd"] = pd.NA
    raw_df.loc[valid_mask, "rpd"] = (
        (raw_df.loc[valid_mask, "fitness"] - raw_df.loc[valid_mask, "reference_fitness"])
        / raw_df.loc[valid_mask, "reference_fitness"]
        * 100
    )
    raw_df.to_csv(raw_runs_path, index=False)

    instance_rows = []
    solver_names = list(EXACT_SOLVERS) + list(META_SOLVERS)
    for instance_id, group in raw_df.groupby("instance_id", sort=True):
        reg = registry[str(instance_id)]
        row = {
            "instance_id": instance_id,
            "instance_path": reg["instance_path"],
            "size": reg["size"],
            "current_best_fitness": reg["current_best_fitness"],
            "is_optimal_proven": reg["is_optimal_proven"],
            "optimal_fitness": reg["optimal_fitness"],
            "optimal_proved_by": reg["optimal_proved_by"],
            "reference_fitness": reg["optimal_fitness"] if reg["is_optimal_proven"] else reg["current_best_fitness"],
            "reference_is_proven_optimum": reg["is_optimal_proven"],
        }
        exact_success = group[(group["solver_family"] == "exact") & group["fitness"].notna()]
        meta_success = group[(group["solver_family"] == "meta") & group["fitness"].notna()]
        row["best_exact_fitness"] = exact_success["fitness"].min() if not exact_success.empty else math.nan
        row["best_meta_fitness"] = meta_success["fitness"].min() if not meta_success.empty else math.nan

        for solver_name in solver_names:
            sg = group[(group["solver_name"] == solver_name) & group["fitness"].notna()]
            row[f"{solver_name}_best_fitness"] = sg["fitness"].min() if not sg.empty else math.nan
            row[f"{solver_name}_avg_fitness"] = sg["fitness"].mean() if not sg.empty else math.nan
            row[f"{solver_name}_std_fitness"] = _safe_std(sg["fitness"]) if not sg.empty else math.nan
            row[f"{solver_name}_avg_runtime_s"] = sg["runtime_s"].mean() if not sg.empty else math.nan
            row[f"{solver_name}_std_runtime_s"] = _safe_std(sg["runtime_s"]) if not sg.empty else math.nan
        instance_rows.append(row)

    pd.DataFrame(instance_rows).to_csv(results_dir / "instance_summary.csv", index=False)

    method_rows = []
    for (size, solver_name), group in raw_df.groupby(["size", "solver_name"], sort=True):
        success = group[group["fitness"].notna()]
        per_instance_best = success.groupby("instance_id")["fitness"].min() if not success.empty else pd.Series(dtype=float)
        optimal_match_mask = success["reference_fitness"].notna() & ((success["fitness"] - success["reference_fitness"]).abs() <= 1e-6)
        method_rows.append(
            {
                "size": int(size),
                "solver_name": solver_name,
                "mean_fitness": success["fitness"].mean() if not success.empty else math.nan,
                "std_fitness": _safe_std(success["fitness"]) if not success.empty else math.nan,
                "mean_runtime_s": success["runtime_s"].mean() if not success.empty else math.nan,
                "std_runtime_s": _safe_std(success["runtime_s"]) if not success.empty else math.nan,
                "mean_rpd": success["rpd"].mean() if not success.empty else math.nan,
                "std_rpd": _safe_std(pd.to_numeric(success["rpd"], errors="coerce")) if not success.empty else math.nan,
                "best_fitness_mean": per_instance_best.mean() if not per_instance_best.empty else math.nan,
                "success_count": int(success.shape[0]),
                "optimal_match_count": int(optimal_match_mask.sum()) if not success.empty else 0,
                "optimal_proven_count": int(success["is_optimal_proven"].sum()) if solver_name in EXACT_SOLVERS and not success.empty else 0,
            }
        )

    pd.DataFrame(method_rows).to_csv(results_dir / "method_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch benchmark runner for AHASP exact and metaheuristic solvers.")
    parser.add_argument(
        "--instances_root",
        type=str,
        default=str(INSTANCE_ROOT / "uniform_100_per_scale_20260413"),
        help="Root directory containing size_*_uniform instance folders.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(PROJECT_ROOT / "experiment" / "results"),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument("--resume_dir", type=str, default=None, help="Existing results directory to resume.")
    parser.add_argument("--time_limit_s", type=int, default=3600, help="Per-run time limit in seconds.")
    parser.add_argument("--max_iterations", type=int, default=100, help="Per-run max iterations for metaheuristics.")
    parser.add_argument("--meta_reps", type=int, default=5, help="Number of repetitions for each metaheuristic.")
    parser.add_argument("--exact_workers", type=int, default=2, help="Number of concurrent exact-solver workers.")
    parser.add_argument("--meta_workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Number of concurrent metaheuristic workers.")
    parser.add_argument("--sizes", type=int, nargs="*", default=None, help="Optional subset of sizes to run.")
    parser.add_argument("--max_instances_per_size", type=int, default=None, help="Optional limit of instances per size for smoke tests.")
    parser.add_argument("--base_seed", type=int, default=1234, help="Base seed used to derive per-run seeds.")
    parser.add_argument("--or_num_search_workers", type=int, default=8, help="Number of CP-SAT workers for OR-Tools.")
    parser.add_argument(
        "--run_group",
        type=str,
        choices=["all", "gurobi", "or_tool", "metaheuristic"],
        default="all",
        help="Select which solver group to run.",
    )
    parser.add_argument(
        "--raw_only",
        action="store_true",
        help="Only record raw runs and registry, skip final summary/RPD generation.",
    )
    return parser.parse_args()


def _format_runtime(runtime_s: Any) -> str:
    if runtime_s is None or pd.isna(runtime_s):
        return "NA"
    return f"{float(runtime_s):.2f}s"


def _format_fitness(fitness: Any) -> str:
    if fitness is None or pd.isna(fitness):
        return "NA"
    return f"{float(fitness):.4f}"


def main() -> None:
    args = parse_args()
    instances_root = Path(args.instances_root).resolve()
    results_root = Path(args.results_root).resolve()

    if args.resume_dir:
        results_dir = Path(args.resume_dir).resolve()
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = results_root / _timestamp()
        results_dir.mkdir(parents=True, exist_ok=True)

    raw_runs_path = results_dir / "raw_runs.csv"
    registry_path = results_dir / "instance_registry.json"
    config_path = results_dir / "run_config.json"

    instances = discover_instances(instances_root, args.sizes, args.max_instances_per_size)
    registry = load_registry(registry_path)
    for instance in instances:
        registry.setdefault(instance["instance_id"], _registry_template(instance))

    raw_df = load_raw_runs(raw_runs_path)
    rebuild_registry_counts(registry, raw_df)
    skip_keys = completed_keys(raw_df)
    exact_tasks, meta_tasks = build_tasks(
        instances,
        skip_keys,
        args.time_limit_s,
        args.max_iterations,
        args.meta_reps,
        args.base_seed,
        args.or_num_search_workers,
        args.run_group,
    )
    total_task_count = len(skip_keys) + len(exact_tasks) + len(meta_tasks)
    completed_task_count = len(skip_keys)
    run_start_time = time.time()

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    write_registry(registry_path, registry)
    print(
        f"[BatchBenchmark] results_dir={results_dir} "
        f"instances={len(instances)} completed={completed_task_count}/{total_task_count} "
        f"pending_exact={len(exact_tasks)} pending_meta={len(meta_tasks)} "
        f"mode=phased_parallel exact_workers={args.exact_workers} meta_workers={args.meta_workers} "
        f"run_group={args.run_group} raw_only={args.raw_only}",
        flush=True,
    )

    def error_result(task: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        return {
            "instance_id": task["instance_id"],
            "instance_path": task["instance_rel_path"],
            "size": task["size"],
            "solver_family": task["solver_family"],
            "solver_name": task["solver_name"],
            "rep": task["rep"],
            "status": "ERROR",
            "fitness": None,
            "distance": None,
            "tardiness": None,
            "runtime_s": None,
            "is_optimal_proven": False,
            "best_bound": None,
            "gap": None,
            "error_message": str(exc),
            "started_at": None,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "rpd": None,
            "reference_fitness": None,
            "reference_is_proven_optimum": None,
        }

    def record_result(result: Dict[str, Any]) -> None:
        nonlocal completed_task_count
        append_raw_run(raw_runs_path, result)
        update_registry_with_result(registry, result)
        write_registry(registry_path, registry)
        completed_task_count += 1
        elapsed_s = time.time() - run_start_time
        print(
            f"[{completed_task_count}/{total_task_count}] "
            f"{result['solver_name']} {result['instance_id']} rep={result['rep']} "
            f"status={result['status']} fitness={_format_fitness(result.get('fitness'))} "
            f"runtime={_format_runtime(result.get('runtime_s'))} elapsed={elapsed_s:.1f}s",
            flush=True,
        )

    def run_parallel_phase(tasks: List[Dict[str, Any]], max_workers: int, phase_name: str) -> None:
        if not tasks:
            return
        print(f"[BatchBenchmark] starting phase={phase_name} tasks={len(tasks)} workers={max_workers}", flush=True)
        cursor = 0
        pending: Dict[Any, Dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            cursor = _submit_until_full(executor, pending, tasks, cursor, max_workers)
            while pending:
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    task = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = error_result(task, exc)
                    record_result(result)
                cursor = _submit_until_full(executor, pending, tasks, cursor, max_workers)
        print(f"[BatchBenchmark] finished phase={phase_name}", flush=True)

    run_parallel_phase(exact_tasks, args.exact_workers, "exact")
    run_parallel_phase(meta_tasks, args.meta_workers, "meta")

    write_registry(registry_path, registry)
    if args.raw_only:
        print(f"[BatchBenchmark] raw_only=True, skipped final summary generation.", flush=True)
    else:
        finalize_results(results_dir, registry)
    print(f"Results written to: {results_dir}")


if __name__ == "__main__":
    main()
