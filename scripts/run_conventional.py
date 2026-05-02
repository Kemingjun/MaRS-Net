import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVENTIONAL_ROOT = REPO_ROOT / "baselines" / "conventional"
EXPERIMENT_ROOT = CONVENTIONAL_ROOT / "experiment"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(CONVENTIONAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONVENTIONAL_ROOT))

from solver_adapters import EXACT_SOLVERS, META_SOLVERS, INSTANCE_ROOT, run_solver_task  # noqa: E402


def _resolve_instance(raw_path):
    path = Path(raw_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _parse_size(path):
    for part in path.parts:
        if part.startswith("size_"):
            try:
                return int(part.split("_")[1])
            except (IndexError, ValueError):
                pass
    stem = path.stem
    if stem.startswith("T"):
        digits = "".join(ch for ch in stem[1:] if ch.isdigit())
        if digits:
            return int(digits)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run one conventional CMRSP baseline on one instance.")
    parser.add_argument("--solver", required=True, choices=list(EXACT_SOLVERS) + list(META_SOLVERS))
    parser.add_argument("--instance", required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--num_search_workers", type=int, default=8)
    args = parser.parse_args()

    instance_path = _resolve_instance(args.instance)
    if not instance_path.exists():
        raise FileNotFoundError(instance_path)

    try:
        instance_rel_path = instance_path.relative_to(INSTANCE_ROOT).as_posix()
    except ValueError:
        instance_rel_path = str(instance_path)

    task = {
        "instance_id": instance_path.stem,
        "instance_abs_path": str(instance_path),
        "instance_rel_path": instance_rel_path,
        "size": _parse_size(instance_path),
        "solver_family": "exact" if args.solver in EXACT_SOLVERS else "metaheuristic",
        "solver_name": args.solver,
        "rep": 1,
        "time_limit_s": args.time_limit,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "num_search_workers": args.num_search_workers,
    }
    result = run_solver_task(task)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
