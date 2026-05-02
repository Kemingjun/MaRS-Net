import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from typing import Dict, Any, List, Optional, Tuple
import argparse  # Added
import sys  # Added
import time  # Added
import numpy as np  # Added
import os

# --- Import Utility Functions ---
# Assuming these are in the locations from your other files
# try:
    # REMOVED: data_utils, file_util, find_project_root
    # from utils.data_utils import collect_result_to_excel, collect_result_to_excel_with_lock, fileName_increase
    # from utils.file_util import find_project_root
    # from heuristic.ALNS import _calculate_and_package_metrics
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import Util.config as Config
from Util.generate_init_solution import generate_solution_nearest2
from Util.load_data import read_excel
    # from heuristic.Util.tensorUtil import parse_vehicle_routes
# except ImportError as e:
#     print(f"Error: Could not import utility functions from 'heuristic'. {e}")
#     print("Please ensure the 'heuristic' directory is accessible.")
#     sys.exit(1)


def _calculate_and_package_metrics(costs, dist_costs, delay_costs, avg_runtime_per_rep):
    """
    Calculates statistics and packages data for a given set of solution metrics.

    Args:
        costs (list): List of total costs (fitness) for each run.
        dist_costs (list): List of distance costs for each run.
        delay_costs (list): List of delay costs (tardiness) for each run.
        avg_runtime_per_rep (float): Average runtime per repetition.

    Returns:
        list: A data package containing calculated statistics.
    """
    costs_np = np.array(costs)
    delay_costs_np = np.array(delay_costs)
    dist_costs_np = np.array(dist_costs)

    min_cost_delay = delay_costs_np.min()
    min_cost_dist = dist_costs_np.min()
    mean_cost_delay = delay_costs_np.mean()
    mean_cost_dist = dist_costs_np.mean()
    mean_cost = costs_np.mean()
    overall_best_cost = costs_np.min()  # The overall best cost found across all runs

    data_package = [
        None,  # Placeholder (e.g., for best solution code, if needed in package)
        dist_costs,
        delay_costs,
        avg_runtime_per_rep,
        min_cost_delay,
        min_cost_dist,
        mean_cost_delay,
        mean_cost_dist,
        mean_cost,
        overall_best_cost,
    ]
    return data_package

nodes = None

def _load_and_preprocess_data(instance_name: str) -> Dict[str, Any]:
    """(Unchanged) Loads instance data and preprocesses sets, parameters, and distance matrices."""

    global nodes

    print(f"--- 1. Loading and Preprocessing Data from: {instance_name} ---")
    instance = read_excel(instance_name)

    # --- 1a. Define Sets ---
    n = len(instance)
    T_h = {i: instance[i - 1][6] for i in range(1, n + 1)}
    request = {i for i in range(1, n + 1)}  # P, pickup nodes
    depot_start = 0
    depot_end = n + 1  # End depot
    nodes = {depot_start, depot_end} | request
    N = nodes - {depot_end}  # N = {0, 1, ..., n}
    N_0 = {depot_start}  # Start depot
    R = request

    carriers = {'C1', 'C2', 'C3', 'C4'}
    shuttles = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'}
    vehicles = carriers | shuttles
    vehicle_list = sorted(list(shuttles)) + sorted(list(carriers))

    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    # --- 1b. Calculate Distances and Times ---
    dist = {}
    dist[(depot_start, depot_end)] = 0
    dist[(depot_start, depot_start)] = 0
    for i in R:
        dist[(0, i)] = instance[i - 1][1] + instance[i - 1][2]
        dist[(i, 0)] = instance[i - 1][3] + instance[i - 1][4]
        dist[(n, i + n)] = instance[i - 1][3] + instance[i - 1][4]
        dist[(i + n, n)] = instance[i - 1][3] + instance[i - 1][4]

        dist[(i, depot_end)] = 0
        dist[(i)] = abs(instance[i - 1][3] - instance[i - 1][1]) + abs(instance[i - 1][4] - instance[i - 1][2])
        dist[(i, i)] = 0
        dist[(i + n, i + n)] = 0
        for j in R:
            if i == j:
                continue
            dist[(i, j)] = abs(instance[i - 1][3] - instance[j - 1][1]) + abs(instance[i - 1][4] - instance[j - 1][2])
            dist[(i + n, j + n)] = abs(instance[i - 1][3] - instance[j - 1][3]) + abs(instance[i - 1][4] - instance[j - 1][4])

    TC = {i: instance[i - 1][5] for i in R}

    M_dist = 99
    for i, j in arcs:
        if (i, j) not in dist:
            dist[i, j] = M_dist

    # parameter_instance = pd.read_excel(instance_name, sheet_name="Vehicles")
    v = Config.V
    travel_time = {k: d / v for k, d in dist.items()}

    M_large = 100000
    omega = Config.weight

    return {
        "instance": instance, "n": n, "T_h": T_h, "request": request,
        "depot_start": depot_start, "depot_end": depot_end, "nodes": nodes,
        "N": N, "R": R, "vehicles": vehicles, "vehicle_list": vehicle_list,
        "shuttles": shuttles, "carriers": carriers, "arcs": arcs, "dist": dist,
        "TC": TC, "travel_time": travel_time, "M_large": M_large,
        "M_dist": M_dist, "omega": omega
    }


def _build_vrp_model(data: Dict[str, Any]) -> Tuple[gp.Model, Dict[str, Any]]:
    """(Unchanged) Builds the Gurobi model, variables, objective, and constraints."""
    print("--- 2. Building VRP Model ---")

    n = data['n']
    nodes, N, R = data['nodes'], data['N'], data['R']
    depot_start, depot_end = data['depot_start'], data['depot_end']
    vehicles, shuttles, carriers = data['vehicles'], data['shuttles'], data['carriers']
    arcs = data['arcs']
    dist, TC, T_h = data['dist'], data['TC'], data['T_h']
    travel_time = data['travel_time']
    M, M_dist = data['M_large'], data['M_dist']
    omega = data['omega']

    model = gp.Model("TwoEchelon_VRPPD")

    # --- 3. Decision Variables ---
    x = model.addVars(arcs, vehicles, vtype=GRB.BINARY, name="x")
    T_F = model.addVars(nodes, vehicles, vtype=GRB.CONTINUOUS, lb=0.0, name="T_F")
    xi = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="xi")
    align_X = model.addVars(R, N, N, vtype=GRB.BINARY, name="align_X")
    delta = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="delta")
    dist_var = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="dist_var")

    # --- 4. Objective Function ---
    total_distance = gp.quicksum(
        align_X[j, i, r] * (dist.get((i + n, r + n), M_dist) + dist.get((r, j), M_dist) + dist.get((j), M_dist))
        for j in R for i in N for r in N
        if (i, j) in arcs and (r, j) in arcs
    )
    total_tardiness = gp.quicksum(delta[i] for i in R)
    model.setObjective(
        omega * total_distance + (1 - omega) * total_tardiness,
        GRB.MINIMIZE
    )

    # --- 5. Constraints ---
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for j in N | {depot_end} if i != j for k in shuttles) == 1
         for i in R), name="task_out_shuttle"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for j in N | {depot_end} if i != j for k in carriers) == 1
         for i in R), name="task_out_carrier"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for i in N if i != j for k in shuttles) == 1
         for j in R), name="task_in_shuttle"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for i in N if i != j for k in carriers) == 1
         for j in R), name="task_in_carrier"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for i in N if i != j) == gp.quicksum(x[j, r, k] for r in R | {depot_end} if r != j)
         for j in R for k in vehicles), name="flow_conservation"
    )
    model.addConstrs(
        (gp.quicksum(x[depot_start, j, k] for j in N | {depot_end} if j != depot_start) == 1
         for k in vehicles), name="depot_start"
    )
    model.addConstrs(
        (gp.quicksum(x[i, depot_end, k] for i in N if i != depot_end) == 1
         for k in vehicles), name="depot_end"
    )
    for j in R:
        for i in N:
            for r in N:
                if i == j or r == j: continue
                carrier_arrives_from_i = gp.quicksum(x[i, j, k] for k in carriers)
                shuttle_arrives_from_r = gp.quicksum(x[r, j, k_prime] for k_prime in shuttles)

                model.addConstr(align_X[j, i, r] <= carrier_arrives_from_i, name=f"align_le_carrier_{j}_{i}_{r}")
                model.addConstr(align_X[j, i, r] <= shuttle_arrives_from_r, name=f"align_le_shuttle_{j}_{i}_{r}")
                model.addConstr(align_X[j, i, r] >= carrier_arrives_from_i + shuttle_arrives_from_r - 1, name=f"align_ge_sum_{j}_{i}_{r}")
    model.addConstrs(
        (T_F[j, k_prime] >= T_F[j, k] + T_h[j]
         for j in R
         for k_prime in shuttles
         for k in carriers),
        name="end_time_shuttle"
    )
    model.addConstrs(
        (T_F[j, k] >= travel_time.get((j), M) - M * (1 - align_X[j, i, r]) + Config.T_couple + Config.T_decouple + travel_time.get((r, j), M) + Config.T_load + xi[j]
         for j in R
         for i in N
         for r in N
         for k in carriers
         if (i, j) in arcs),
        name="end_time_carrier"
    )
    model.addConstrs(
        (xi[j] >= T_F[i, k] - M * (1 - x[i, j, k])
         for i in N for j in R for k in vehicles if (i, j) in arcs),
        name="sync_time_carrier"
    )
    model.addConstrs(
        (xi[j] >= T_F[i, k] + travel_time.get((i + n, r + n), M) - M * (1 - align_X[j, i, r])
         for j in R
         for i in N for r in N
         for k in carriers
         if (i, j) in arcs and (r, j) in arcs),
        name="sync_time_shuttle"
    )
    model.addConstrs(
        (delta[i] >= T_F[i, k] - TC.get(i)
         for i in R for k in shuttles),
        name="tardiness"
    )

    variables = {
        "x": x, "T_F": T_F, "xi": xi, "align_X": align_X,
        "delta": delta, "dist_var": dist_var,
        "total_distance": total_distance, "total_tardiness": total_tardiness
    }
    return model, variables


def _apply_initial_solution(model: gp.Model, x: gp.Var, instance: Any, n: int, vehicles: set):
    """(Unchanged) Generates a greedy solution and applies it as a Warm Start."""
    print("--- Applying initial solution (Warm Start) ---")
    try:
        solution = generate_solution_nearest2(instance)
        predefined_routes = convert_routes_to_dict(solution.get_code(), n)

        if predefined_routes:
            for k, path in predefined_routes.items():
                if k in vehicles:
                    for idx in range(len(path) - 1):
                        start_node = path[idx]
                        end_node = path[idx + 1]
                        if (start_node, end_node, k) in x:
                            x[start_node, end_node, k].Start = 1
    except Exception as e:
        print(f"Warning: Could not generate or apply initial solution. {e}")


def _extract_and_package_solution(
        model: gp.Model,
        data: Dict[str, Any],
        vars: Dict[str, Any],
        time_limit: Optional[int]
) -> List:
    """Extracts the solution, prints routes, and packages metrics."""
    print("\n" + "=" * 30)
    print(f"Solution found! (Status: {model.Status})")
    print(f"Total Objective Cost: {model.ObjVal:.2f}")
    # ADDED: Print Gap and Runtime
    print(f"MIP Gap: {model.MIPGap * 100:.4f}%")
    print(f"Runtime: {model.Runtime:.2f}s")
    print("=" * 30 + "\n")

    # --- Unpack data and variables ---
    R, N, n = data['R'], data['N'], data['n']
    depot_start, depot_end = data['depot_start'], data['depot_end']
    vehicles, vehicle_list = data['vehicles'], data['vehicle_list']
    arcs = data['arcs']
    omega = data['omega']

    x, T_F, delta, dist_var, align_X = vars['x'], vars['T_F'], vars['delta'], vars['dist_var'], vars['align_X']
    total_distance, total_tardiness = vars['total_distance'], vars['total_tardiness']

    # --- Print Details ---
    tardiness_values = {i: delta[i].X for i in R if delta[i].X > 0.01}
    if tardiness_values:
        print("\nTardiness observed:")
        for i, val in tardiness_values.items():
            print(f"  - Task {i}: {val:.2f} time units late")

    travel_time_values = {i: dist_var[i].X for i in R if dist_var[i].X > 0.01}
    if travel_time_values:
        print("\nTravel time (dist_var):")
        for i, val in travel_time_values.items():
            print(f"  - Task {i}: {val:.2f}")

    print("\nAlignment (align_X[j, i, r] = 1):")
    for j, i, r in align_X:
        if align_X[j, i, r].X > 0.5:
            print(f"  - Order {j} aligns carrier from {i} and shuttle from {r}")
    print("\n" + "-" * 30 + "\n")

    # --- Print Routes ---
    for k in vehicles:
        print(f"Route for vehicle {k}:")
        try:
            current_node = depot_start
            path = [depot_start]
            for _ in range(len(nodes) + 1):  # +1 for safety
                found_next = False
                for j in N | {depot_end}:
                    if j != current_node and (current_node, j) in arcs and x[current_node, j, k].X > 0.5:
                        path.append(j)
                        current_node = j
                        found_next = True
                        break
                if not found_next or current_node == depot_end:
                    break
            print(" -> ".join(map(str, path)))

            # *** START OF CORRECTED BLOCK ***
            # Iterate through the path and print finish times
            for node in path:
                if node in R: # R is the set of request nodes {1, ..., n}
                    print(f"  - Node {node}: Finish @ {T_F[node, k].X:.2f}")
            # *** END OF CORRECTED BLOCK ***

        except (gp.GurobiError, AttributeError) as e:
            print(f"  Could not retrieve route details for {k}. Error: {e}")

    # --- Package Results ---
    # NOTE: Gurobi provides a single result, but we put it in lists
    # to match the format expected by _calculate_and_package_metrics
    costs = [model.ObjVal]
    dist_costs = [omega * total_distance.getValue()]
    delay_costs = [(1 - omega) * total_tardiness.getValue()]

    raw_routes = []
    all_vehicle_routes = []
    for k in vehicle_list:  # Use sorted list for consistent order
        path = []
        try:
            # Check if vehicle is used
            if gp.quicksum(x[depot_start, j, k] for j in R if (depot_start, j) in arcs).getValue() > 0.5:
                current_node = depot_start
                while current_node != depot_end and len(path) <= n:
                    for j in N | {depot_end}:
                        if j != current_node and (current_node, j) in arcs and x[current_node, j, k].X > 0.5:
                            if j in R:
                                path.append(j - 1)  # Convert to 0-indexed customer ID
                            current_node = j
                            break
                    else:
                        break  # No next node found
        except (gp.GurobiError, AttributeError):
            pass  # Vehicle not used

        padded_route = path + [-1] * (n - len(path))
        all_vehicle_routes.append(padded_route)

    raw_routes.append([all_vehicle_routes])  # Match required nesting

    # Use the actual model runtime, not the time limit
    actual_runtime = model.Runtime
    data_package = _calculate_and_package_metrics(costs, dist_costs, delay_costs, actual_runtime)
    data_package[0] = raw_routes
    return data_package

def _package_empty_solution(n: int, vehicle_list: List[str], time_limit: Optional[int]) -> List:
    """(Unchanged) Returns an empty/default metrics package when the model is infeasible or unsolved."""
    costs = [float('inf')]
    dist_costs = [float('inf')]
    delay_costs = [float('inf')]

    empty_route = [-1] * n
    all_vehicle_routes = [empty_route for _ in vehicle_list]
    raw_routes = [[[all_vehicle_routes]]]

    data_package = _calculate_and_package_metrics(costs, dist_costs, delay_costs, time_limit)
    data_package[0] = raw_routes
    return data_package


def solve_vrp_model(fixed_routes: Optional[Dict[str, List[int]]] = None,
                    instance_name: Optional[str] = None,
                    time_limit: Optional[int] = None) -> List:
    """(Unchanged) Main function: Builds and solves the VRP model."""

    data = _load_and_preprocess_data(instance_name)

    model, variables = _build_vrp_model(data)

    if fixed_routes:
        print("--- 3. Applying fixed routes ---")
        x = variables['x']
        for k, path in fixed_routes.items():
            if k in data['vehicles']:
                for i in range(len(path) - 1):
                    start_node = path[i]
                    end_node = path[i + 1]
                    if (start_node, end_node, k) in x:
                        model.addConstr(x[start_node, end_node, k] == 1, name=f"fix_route_{k}_{start_node}_{end_node}")
                    else:
                        print(f"Warning: Arc ({start_node}, {end_node}) for vehicle {k} not in 'x' variables.")
    else:
        print("--- 3. Applying Warm Start ---")
        _apply_initial_solution(
            model,
            variables['x'],
            data['instance'],
            data['n'],
            data['vehicles']
        )

    # --- 4. Solve Model ---
    if time_limit:
        model.Params.TimeLimit = time_limit

    print(f"--- 4. Starting model optimization (TimeLimit: {time_limit}s) ---")
    model.optimize()

    # --- 5. Extract and Package Solution ---
    if model.Status == GRB.INFEASIBLE:
        print("\n" + "!" * 30)
        print("Model is INFEASIBLE.")
        # ADDED: Compute and print IIS
        print("Computing IIS (Irreducible Inconsistent Subsystem)...")
        model.computeIIS()
        model.write("model_iis.ilp")
        print("IIS written to model_iis.ilp")
        print("!" * 30 + "\n")
        return _package_empty_solution(data['n'], data['vehicle_list'], time_limit)

    elif model.SolCount > 0:
        print("--- 5. Extracting and Packaging Solution ---")
        return _extract_and_package_solution(model, data, variables, time_limit)

    else:
        print("\n" + "!" * 30)
        print(f"Optimization stopped with no solution found (Status: {model.Status}).")
        print(f"(Likely hit time limit before finding a feasible integer solution)")
        print("!" * 30 + "\n")
        return _package_empty_solution(data['n'], data['vehicle_list'], time_limit)


def _gurobi_status_name(status: int) -> str:
    status_map = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_map.get(status, str(status))


def solve_vrp_model_structured(
        fixed_routes: Optional[Dict[str, List[int]]] = None,
        instance_name: Optional[str] = None,
        time_limit: Optional[int] = None) -> Dict[str, Any]:
    if instance_name is None:
        raise ValueError("instance_name must be provided")

    result = {
        "status": "ERROR",
        "fitness": None,
        "distance": None,
        "tardiness": None,
        "runtime_s": None,
        "is_optimal_proven": False,
        "best_bound": None,
        "gap": None,
        "error_message": None,
    }

    try:
        data = _load_and_preprocess_data(instance_name)
        model, variables = _build_vrp_model(data)

        if fixed_routes:
            x = variables['x']
            for k, path in fixed_routes.items():
                if k in data['vehicles']:
                    for i in range(len(path) - 1):
                        start_node = path[i]
                        end_node = path[i + 1]
                        if (start_node, end_node, k) in x:
                            model.addConstr(x[start_node, end_node, k] == 1, name=f"fix_route_{k}_{start_node}_{end_node}")
        else:
            _apply_initial_solution(model, variables['x'], data['instance'], data['n'], data['vehicles'])

        if time_limit:
            model.Params.TimeLimit = time_limit

        model.optimize()

        result["status"] = _gurobi_status_name(model.Status)
        result["runtime_s"] = float(model.Runtime)
        result["is_optimal_proven"] = model.Status == GRB.OPTIMAL

        try:
            result["best_bound"] = float(model.ObjBound)
        except Exception:
            result["best_bound"] = None

        if model.SolCount > 0:
            result["fitness"] = float(model.ObjVal)
            result["distance"] = float(variables["total_distance"].getValue())
            result["tardiness"] = float(variables["total_tardiness"].getValue())
            try:
                result["gap"] = float(model.MIPGap) * 100
            except Exception:
                result["gap"] = None
    except Exception as exc:
        result["error_message"] = str(exc)

    return result


# =============================================================================
# Helper Functions (Kept from original, needed by _apply_initial_solution)
# =============================================================================

def convert_routes_to_dict(raw_data: List[List[int]], num_tasks: int) -> Dict[str, List[int]]:
    """(Unchanged) Converts the 'get_code()' format from a greedy solution
    to the 'fixed_routes' dictionary format.
    """
    if len(raw_data) != 2:
        print(f"Warning: Expected 2 lists in raw_data, got {len(raw_data)}. Adapting...")
        carrier_sequence = raw_data[0] if len(raw_data) > 0 else []
        shuttle_sequence = raw_data[1] if len(raw_data) > 1 else []
    else:
        carrier_sequence = raw_data[0]
        shuttle_sequence = raw_data[1]

    carrier_routes = parse_sequence(carrier_sequence, 'C', num_tasks)
    shuttle_routes = parse_sequence(shuttle_sequence, 'S', num_tasks)
    all_routes = {**carrier_routes, **shuttle_routes}
    return all_routes


def parse_sequence(sequence: List[int], vehicle_prefix: str, n: int) -> Dict[str, List[int]]:
    """(Unchanged) Parses a 0-delimited sequence of nodes into routes for each vehicle."""
    routes = {}
    if not sequence:
        return routes

    vehicle_count = 1
    current_path = []
    depot_start = 0
    depot_end = n + 1

    for node in sequence:
        if node == 0:
            if current_path:
                full_route = [depot_start] + current_path + [depot_end]
                routes[f'{vehicle_prefix}{vehicle_count}'] = full_route
                vehicle_count += 1
                current_path = []
        elif node != -1:
            current_path.append(node)

    if current_path:
        full_route = [depot_start] + current_path + [depot_end]
        routes[f'{vehicle_prefix}{vehicle_count}'] = full_route

    return routes


# REMOVED: exp_start function
# REMOVED: convert_to_predefined_routes (was only used in old __main__)


# =============================================================================
# NEW: Functions for Command-Line Execution
# =============================================================================

def print_metrics_package(data_package: list, title: str):
    """
    Prints the metrics package in a human-readable format.
    The structure is based on _calculate_and_package_metrics.
    """
    try:
        # data_package structure from _calculate_and_package_metrics:
        # [0]: best_solution_code (list of routes)
        # [1]: dist_costs (list of 1)
        # [2]: delay_costs (list of 1)
        # [3]: avg_runtime_per_rep (this is the single actual_runtime)
        # [4]: min_cost_delay
        # [5]: min_cost_dist
        # [6]: mean_cost_delay
        # [7]: mean_cost_dist
        # [8]: mean_cost (avg total fitness)
        # [9]: overall_best_cost (min total fitness)

        actual_runtime = data_package[3]
        min_delay = data_package[4]
        min_dist = data_package[5]
        # mean_delay = data_package[6] # Same as min for Gurobi
        # mean_dist = data_package[7] # Same as min for Gurobi
        # mean_cost = data_package[8] # Same as best for Gurobi
        best_cost = data_package[9]

        print("\n" + "-" * 30)
        print(f"--- {title} ---")
        print(f"  Best Cost (Fitness):   {best_cost:.4f}")
        print(f"  Distance Cost:         {min_dist:.4f}")
        print(f"  Tardiness Cost:        {min_delay:.4f}")
        print(f"  Gurobi Actual Runtime: {actual_runtime:.4f}s")
        print("-" * 30)

    except IndexError:
        print(f"Error: Could not unpack metrics for {title}. Package was: {data_package}")
    except Exception as e:
        print(f"An error occurred while printing metrics: {e}")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Gurobi Exact Solver Runner")

    parser.add_argument(
        "--load_path",
        type=str,
        default="uniform_100_per_scale_20260413/size_10_uniform/T10_I1_uniform.xlsx",
        help="Full path to the instance .xlsx file.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=200,
        help="Maximum runtime in seconds for Gurobi. (Default: No limit)"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    print(f"--- Starting Experiment ---")
    print(f"Algorithm:      Gurobi")
    print(f"Instance Path: {args.load_path}")
    print(f"Time Limit:    {args.time_limit}s")
    print("-" * 30)

    start_time_total = time.time()

    # try:
        # Call the solver. fixed_routes=None means it will use Warm Start
    data_package = solve_vrp_model(
        fixed_routes=None,
        instance_name=args.load_path,
        time_limit=args.time_limit
    )

    # Print the results
    print("\n" + "=" * 30)
    print(f"--- FINAL RESULTS ---")

    # data_package[0] is the raw_routes
    best_solution_code = data_package[0]
    print(f"Best Solution Code (Routes):")

    # Pretty print the routes if found
    if best_solution_code and best_solution_code[0] and best_solution_code[0][0]:
        # The structure is [[all_vehicle_routes]]
        for i, route in enumerate(best_solution_code[0][0]):
            # Filter out -1 placeholders for cleaner printing
            clean_route = [node for node in route if node != -1]
            if clean_route:  # Only print if the route is not empty
                print(f"  Vehicle {i + 1}: {clean_route}")
    else:
        print("  No solution routes found or model was infeasible/untimed.")

    # Call the new print function
    print_metrics_package(data_package, f"Metrics")

    print("=" * 30 + "\n")

    # except FileNotFoundError:
    #     print(f"Error: Instance file not found at {args.load_path}")
    #     sys.exit(1)
    # except gp.GurobiError as e:
    #     print(f"A Gurobi error occurred: {e}")
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)

    end_time_total = time.time()
    print("-" * 30)
    print(f"Gurobi run finished in {end_time_total - start_time_total:.2f} seconds.")


# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    main()
