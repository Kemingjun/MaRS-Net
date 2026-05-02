import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
REPO_ROOT = next((parent for parent in CURRENT_DIR.parents if (parent / "Instance").exists()), PROJECT_ROOT)
INSTANCE_ROOT = REPO_ROOT / "Instance"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metaheuristic.ALNS import ALNS  # noqa: E402
from metaheuristic.DABC import DABC  # noqa: E402
from metaheuristic.DIWO import DIWO  # noqa: E402
from metaheuristic.IGA import IGA  # noqa: E402
from GUROBI.gurobi import solve_vrp_model_structured as solve_gurobi_structured  # noqa: E402
from OR_TOOL.or_tool import solve_vrp_model_structured as solve_or_tool_structured  # noqa: E402
from Util.load_data import read_excel  # noqa: E402


EXACT_SOLVERS = ("gurobi", "or_tool")
META_SOLVERS = ("ALNS", "IGA", "DABC", "DIWO")


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _relative_instance_path(instance_path: str) -> str:
    path = Path(instance_path).resolve()
    return path.relative_to(INSTANCE_ROOT).as_posix()


def _instance_name_no_ext(instance_path: str) -> str:
    return str(Path(_relative_instance_path(instance_path)).with_suffix("")).replace("\\", "/")


def _base_result(task: Dict[str, Any]) -> Dict[str, Any]:
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
        "error_message": None,
        "started_at": None,
        "finished_at": None,
        "rpd": None,
        "reference_fitness": None,
        "reference_is_proven_optimum": None,
    }


@contextlib.contextmanager
def _suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _run_gurobi(task: Dict[str, Any]) -> Dict[str, Any]:
    result = _base_result(task)
    result["started_at"] = _utc_now()
    try:
        with _suppress_output():
            structured = solve_gurobi_structured(
                fixed_routes=None,
                instance_name=task["instance_abs_path"],
                time_limit=task["time_limit_s"],
            )
        result.update(structured)
    except Exception as exc:
        result["error_message"] = str(exc)
    result["finished_at"] = _utc_now()
    return result


def _run_or_tool(task: Dict[str, Any]) -> Dict[str, Any]:
    result = _base_result(task)
    result["started_at"] = _utc_now()
    try:
        with _suppress_output():
            structured = solve_or_tool_structured(
                fixed_routes=None,
                instance_name=task["instance_abs_path"],
                time_limit=task["time_limit_s"],
                log_search_progress=False,
                num_search_workers=task.get("num_search_workers", 8),
                random_seed=task.get("seed"),
            )
        result.update(structured)
    except Exception as exc:
        result["error_message"] = str(exc)
    result["finished_at"] = _utc_now()
    return result


def _run_meta(task: Dict[str, Any]) -> Dict[str, Any]:
    result = _base_result(task)
    result["started_at"] = _utc_now()
    try:
        instance = read_excel(task["instance_rel_path"])
        instance_name = _instance_name_no_ext(task["instance_abs_path"])
        solver_name = task["solver_name"]

        with _suppress_output():
            if solver_name == "ALNS":
                best_solution, best_fitness, runtime_s = ALNS(
                    instance_name=instance_name,
                    max_iterations=task["max_iterations"],
                    time_limit_s=task["time_limit_s"],
                    seed=task["seed"],
                )
            elif solver_name == "IGA":
                best_solution, best_fitness, runtime_s = IGA(
                    instance_name=instance_name,
                    max_iterations=task["max_iterations"],
                    time_limit_s=task["time_limit_s"],
                    seed=task["seed"],
                )
            elif solver_name == "DABC":
                best_solution, best_fitness, runtime_s = DABC(
                    instance_name=instance_name,
                    max_iterations=task["max_iterations"],
                    time_limit_s=task["time_limit_s"],
                    seed=task["seed"],
                )
            elif solver_name == "DIWO":
                best_solution, best_fitness, runtime_s = DIWO(
                    instance_name=instance_name,
                    max_iterations=task["max_iterations"],
                    time_limit_s=task["time_limit_s"],
                    seed=task["seed"],
                )
            else:
                raise ValueError(f"Unsupported meta solver: {solver_name}")

        best_solution.get_fitness()
        result["status"] = "SUCCESS"
        result["fitness"] = float(best_fitness)
        result["distance"] = float(best_solution.distance)
        result["tardiness"] = float(best_solution.tardiness)
        result["runtime_s"] = float(runtime_s)
    except Exception as exc:
        result["error_message"] = str(exc)
    result["finished_at"] = _utc_now()
    return result


def run_solver_task(task: Dict[str, Any]) -> Dict[str, Any]:
    if task["solver_name"] == "gurobi":
        return _run_gurobi(task)
    if task["solver_name"] == "or_tool":
        return _run_or_tool(task)
    return _run_meta(task)
