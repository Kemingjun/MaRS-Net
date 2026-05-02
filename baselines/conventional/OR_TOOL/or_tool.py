import argparse
import math
import os
import sys
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import Util.config as Config  # noqa: E402


def _calculate_and_package_metrics(costs, dist_costs, delay_costs, avg_runtime_per_rep):
    costs_np = np.array(costs)
    delay_costs_np = np.array(delay_costs)
    dist_costs_np = np.array(dist_costs)

    min_cost_delay = delay_costs_np.min()
    min_cost_dist = dist_costs_np.min()
    mean_cost_delay = delay_costs_np.mean()
    mean_cost_dist = dist_costs_np.mean()
    mean_cost = costs_np.mean()
    overall_best_cost = costs_np.min()

    return [
        None,
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


def _resolve_instance_path(instance_name: str) -> str:
    if os.path.isabs(instance_name) and os.path.exists(instance_name):
        return instance_name

    direct_path = os.path.join(PROJECT_ROOT, "Instance", instance_name)
    if os.path.exists(direct_path):
        return direct_path

    if os.path.exists(instance_name):
        return os.path.abspath(instance_name)

    raise FileNotFoundError(f"Instance file not found: {instance_name}")


def _read_instance(instance_name: str) -> List[List[float]]:
    file_path = _resolve_instance_path(instance_name)
    df = pd.read_excel(file_path)
    return [list(row) for _, row in df.iterrows()]


def _to_coord_int(value: float, coord_scale: int) -> int:
    return int(round(float(value) * coord_scale))


def _derive_objective_scaling(weight: float, time_scale: int, distance_scale: int) -> Tuple[int, int, int]:
    omega = Fraction(str(weight)).limit_denominator()
    tardiness_weight = Fraction(1, 1) - omega
    common_scale = math.lcm(omega.denominator * distance_scale, tardiness_weight.denominator * time_scale)
    dist_coeff = omega.numerator * (common_scale // (omega.denominator * distance_scale))
    tard_coeff = tardiness_weight.numerator * (common_scale // (tardiness_weight.denominator * time_scale))
    return dist_coeff, tard_coeff, common_scale


def _estimate_horizon(instance: List[List[float]], dist: Dict[Any, float], travel_time: Dict[Any, float], T_h: Dict[int, float], TC: Dict[int, float]) -> int:
    max_real_distance = max(float(v) for v in dist.values())
    max_task_travel = max(float(dist[i]) for i in T_h)
    max_processing = max(float(v) for v in T_h.values())
    max_deadline = max(float(v) for v in TC.values())

    # Safe but much tighter than a fixed 100000: each task can incur at most
    # one pickup-delivery leg, two synchronization/transfer legs, and the fixed
    # coupling/loading/processing times, summed sequentially over all tasks.
    per_task_upper = (
        Config.T_couple
        + Config.T_decouple
        + Config.T_load
        + max_processing
        + max_task_travel / Config.V
        + 2 * max_real_distance / Config.V
    )
    horizon_real = len(instance) * per_task_upper + max_deadline + max_real_distance / Config.V + 1
    return int(math.ceil(horizon_real))


def parse_sequence(sequence: List[int], vehicle_prefix: str, n: int) -> Dict[str, List[int]]:
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
                routes[f"{vehicle_prefix}{vehicle_count}"] = [depot_start] + current_path + [depot_end]
                vehicle_count += 1
                current_path = []
        elif node != -1:
            current_path.append(node)

    if current_path:
        routes[f"{vehicle_prefix}{vehicle_count}"] = [depot_start] + current_path + [depot_end]

    return routes


def convert_routes_to_dict(raw_data: List[List[int]], num_tasks: int) -> Dict[str, List[int]]:
    if len(raw_data) != 2:
        carrier_sequence = raw_data[0] if len(raw_data) > 0 else []
        shuttle_sequence = raw_data[1] if len(raw_data) > 1 else []
    else:
        carrier_sequence, shuttle_sequence = raw_data

    carrier_routes = parse_sequence(carrier_sequence, "C", num_tasks)
    shuttle_routes = parse_sequence(shuttle_sequence, "S", num_tasks)
    return {**carrier_routes, **shuttle_routes}


def _load_and_preprocess_data(instance_name: str) -> Dict[str, Any]:
    print(f"--- 1. Loading and Preprocessing Data from: {instance_name} ---")
    instance = _read_instance(instance_name)

    n = len(instance)
    T_h = {i: instance[i - 1][6] for i in range(1, n + 1)}
    request = {i for i in range(1, n + 1)}
    depot_start = 0
    depot_end = n + 1
    nodes = {depot_start, depot_end} | request
    N = nodes - {depot_end}
    R = request

    carriers = {"C1", "C2", "C3", "C4"}
    shuttles = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"}
    vehicles = carriers | shuttles
    vehicle_list = sorted(list(shuttles)) + sorted(list(carriers))

    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    coord_scale = 10
    distance_scale = coord_scale
    time_scale = 6 * coord_scale
    dist_time_coeff = 5

    dist = {}
    dist[(depot_start, depot_end)] = 0
    dist[(depot_start, depot_start)] = 0
    for i in R:
        source_x = _to_coord_int(instance[i - 1][1], coord_scale)
        source_y = _to_coord_int(instance[i - 1][2], coord_scale)
        destination_x = _to_coord_int(instance[i - 1][3], coord_scale)
        destination_y = _to_coord_int(instance[i - 1][4], coord_scale)

        dist[(0, i)] = source_x + source_y
        dist[(i, 0)] = destination_x + destination_y
        dist[(n, i + n)] = destination_x + destination_y
        dist[(i + n, n)] = destination_x + destination_y

        dist[(i, depot_end)] = 0
        dist[i] = abs(destination_x - source_x) + abs(destination_y - source_y)
        dist[(i, i)] = 0
        dist[(i + n, i + n)] = 0
        for j in R:
            if i == j:
                continue
            next_source_x = _to_coord_int(instance[j - 1][1], coord_scale)
            next_source_y = _to_coord_int(instance[j - 1][2], coord_scale)
            next_destination_x = _to_coord_int(instance[j - 1][3], coord_scale)
            next_destination_y = _to_coord_int(instance[j - 1][4], coord_scale)
            dist[(i, j)] = abs(destination_x - next_source_x) + abs(destination_y - next_source_y)
            dist[(i + n, j + n)] = abs(destination_x - next_destination_x) + abs(destination_y - next_destination_y)

    TC = {i: instance[i - 1][5] for i in R}

    M_dist = int(math.ceil(max(float(v) for v in dist.values())))
    for i, j in arcs:
        if (i, j) not in dist:
            dist[(i, j)] = M_dist

    travel_time = {k: d / (distance_scale * Config.V) for k, d in dist.items()}
    travel_time_scaled = {k: d * dist_time_coeff for k, d in dist.items()}
    dist_scaled = dict(dist)
    T_h_scaled = {i: int(T_h[i] * time_scale) for i in R}
    TC_scaled = {i: int(TC[i] * time_scale) for i in R}

    dist_real = {k: d / distance_scale for k, d in dist.items()}
    M_large = _estimate_horizon(instance, dist_real, travel_time, T_h, TC)
    M_large_scaled = M_large * time_scale
    M_dist_scaled = M_dist
    omega = Config.weight
    obj_dist_coeff, obj_tard_coeff, obj_scale = _derive_objective_scaling(omega, time_scale, distance_scale)

    return {
        "instance": instance,
        "instance_path": _resolve_instance_path(instance_name),
        "n": n,
        "T_h": T_h,
        "T_h_scaled": T_h_scaled,
        "request": request,
        "depot_start": depot_start,
        "depot_end": depot_end,
        "nodes": nodes,
        "N": N,
        "R": R,
        "vehicles": vehicles,
        "vehicle_list": vehicle_list,
        "shuttles": shuttles,
        "carriers": carriers,
        "arcs": arcs,
        "dist": dist,
        "dist_scaled": dist_scaled,
        "TC": TC,
        "TC_scaled": TC_scaled,
        "travel_time": travel_time,
        "travel_time_scaled": travel_time_scaled,
        "M_large": M_large,
        "M_large_scaled": M_large_scaled,
        "M_dist": M_dist,
        "M_dist_scaled": M_dist_scaled,
        "omega": omega,
        "time_scale": time_scale,
        "distance_scale": distance_scale,
        "coord_scale": coord_scale,
        "dist_time_coeff": dist_time_coeff,
        "obj_dist_coeff": obj_dist_coeff,
        "obj_tard_coeff": obj_tard_coeff,
        "obj_scale": obj_scale,
    }


def _build_cp_sat_model(data: Dict[str, Any]) -> Tuple[cp_model.CpModel, Dict[str, Any]]:
    print("--- 2. Building OR-Tools CP-SAT Model ---")

    n = data["n"]
    nodes, N, R = data["nodes"], data["N"], data["R"]
    depot_start, depot_end = data["depot_start"], data["depot_end"]
    vehicles, shuttles, carriers = data["vehicles"], data["shuttles"], data["carriers"]
    arcs = data["arcs"]
    dist_scaled, TC_scaled, T_h_scaled = data["dist_scaled"], data["TC_scaled"], data["T_h_scaled"]
    travel_time_scaled = data["travel_time_scaled"]
    M_scaled = data["M_large_scaled"]
    M_dist_scaled = data["M_dist_scaled"]
    obj_dist_coeff, obj_tard_coeff = data["obj_dist_coeff"], data["obj_tard_coeff"]
    time_scale = data["time_scale"]
    model = cp_model.CpModel()

    x = {(i, j, k): model.NewBoolVar(f"x_{i}_{j}_{k}") for i, j in arcs for k in vehicles}
    flow_c = {(i, j): model.NewBoolVar(f"flow_c_{i}_{j}") for j in R for i in N if i != j and (i, j) in arcs}
    flow_s = {(i, j): model.NewBoolVar(f"flow_s_{i}_{j}") for j in R for i in N if i != j and (i, j) in arcs}
    carrier_finish = {i: model.NewIntVar(0, M_scaled, f"carrier_finish_{i}") for i in R}
    shuttle_finish = {i: model.NewIntVar(0, M_scaled, f"shuttle_finish_{i}") for i in R}
    delta = {i: model.NewIntVar(0, M_scaled, f"delta_{i}") for i in R}
    per_task_dist_upper = max(1, 3 * M_dist_scaled)
    dist_var = {i: model.NewIntVar(0, per_task_dist_upper, f"dist_var_{i}") for i in R}
    z_case1 = {
        (j, i, r): model.NewBoolVar(f"z_case1_{j}_{i}_{r}")
        for j in R for i in R for r in R if i != j and r != j
    }
    z_case2 = {j: model.NewBoolVar(f"z_case2_{j}") for j in R}
    z_case3 = {(j, r): model.NewBoolVar(f"z_case3_{j}_{r}") for j in R for r in R if r != j}
    z_case4 = {(j, i): model.NewBoolVar(f"z_case4_{j}_{i}") for j in R for i in R if i != j}

    total_distance = model.NewIntVar(0, len(R) * per_task_dist_upper, "total_distance")
    total_tardiness = model.NewIntVar(0, len(R) * M_scaled, "total_tardiness")

    model.Add(total_distance == sum(dist_var.values()))
    model.Add(total_tardiness == sum(delta.values()))
    model.Minimize(obj_dist_coeff * total_distance + obj_tard_coeff * total_tardiness)

    for i in R:
        model.Add(sum(x[i, j, k] for j in (N | {depot_end}) if i != j for k in shuttles) == 1)
        model.Add(sum(x[i, j, k] for j in (N | {depot_end}) if i != j for k in carriers) == 1)

    for j in R:
        model.Add(sum(x[i, j, k] for i in N if i != j for k in shuttles) == 1)
        model.Add(sum(x[i, j, k] for i in N if i != j for k in carriers) == 1)

    for j in R:
        for k in vehicles:
            model.Add(
                sum(x[i, j, k] for i in N if i != j) ==
                sum(x[j, r, k] for r in (R | {depot_end}) if r != j)
            )

    for k in vehicles:
        model.Add(sum(x[depot_start, j, k] for j in (N | {depot_end}) if j != depot_start) == 1)
        model.Add(sum(x[i, depot_end, k] for i in N if i != depot_end) == 1)

    for j in R:
        carrier_flow_terms = []
        shuttle_flow_terms = []
        for i in N:
            if i == j or (i, j) not in arcs:
                continue
            carrier_incoming = sum(x[i, j, k] for k in carriers)
            shuttle_incoming = sum(x[i, j, k] for k in shuttles)
            model.Add(carrier_incoming == flow_c[i, j])
            model.Add(shuttle_incoming == flow_s[i, j])
            carrier_flow_terms.append(flow_c[i, j])
            shuttle_flow_terms.append(flow_s[i, j])

        model.Add(sum(carrier_flow_terms) == 1)
        model.Add(sum(shuttle_flow_terms) == 1)
        model.Add(shuttle_finish[j] == carrier_finish[j] + T_h_scaled[j])
        model.Add(delta[j] >= shuttle_finish[j] - TC_scaled[j])

    couple_scaled = int(Config.T_couple * time_scale)
    decouple_scaled = int(Config.T_decouple * time_scale)
    load_scaled = int(Config.T_load * time_scale)
    for j in R:
        base_from_depot = (
            couple_scaled
            + load_scaled
            + travel_time_scaled.get((0, j), M_scaled)
            + travel_time_scaled.get(j, M_scaled)
            + decouple_scaled
        )

        model.AddBoolAnd([flow_c[0, j], flow_s[0, j]]).OnlyEnforceIf(z_case2[j])
        model.AddBoolOr([flow_c[0, j].Not(), flow_s[0, j].Not()]).OnlyEnforceIf(z_case2[j].Not())
        model.Add(carrier_finish[j] == base_from_depot).OnlyEnforceIf(z_case2[j])
        model.Add(dist_var[j] == dist_scaled.get((0, j), M_dist_scaled) + dist_scaled.get(j, M_dist_scaled)).OnlyEnforceIf(z_case2[j])

        case_terms = [z_case2[j]]

        for i in R:
            if i == j:
                continue
            for r in R:
                if r == j:
                    continue
                z = z_case1[j, i, r]
                base_from_r = (
                    couple_scaled
                    + load_scaled
                    + travel_time_scaled.get((r, j), M_scaled)
                    + travel_time_scaled.get(j, M_scaled)
                    + decouple_scaled
                )
                model.AddBoolAnd([flow_c[i, j], flow_s[r, j]]).OnlyEnforceIf(z)
                model.AddBoolOr([flow_c[i, j].Not(), flow_s[r, j].Not()]).OnlyEnforceIf(z.Not())
                model.Add(carrier_finish[j] >= shuttle_finish[r] + base_from_r).OnlyEnforceIf(z)
                model.Add(
                    carrier_finish[j] >= carrier_finish[i] + travel_time_scaled.get((i + n, r + n), M_scaled) + base_from_r
                ).OnlyEnforceIf(z)
                model.Add(
                    dist_var[j] == (
                        dist_scaled.get((i + n, r + n), M_dist_scaled)
                        + dist_scaled.get((r, j), M_dist_scaled)
                        + dist_scaled.get(j, M_dist_scaled)
                    )
                ).OnlyEnforceIf(z)
                case_terms.append(z)

        for r in R:
            if r == j:
                continue
            z = z_case3[j, r]
            base_from_r = (
                couple_scaled
                + load_scaled
                + travel_time_scaled.get((r, j), M_scaled)
                + travel_time_scaled.get(j, M_scaled)
                + decouple_scaled
            )
            model.AddBoolAnd([flow_c[0, j], flow_s[r, j]]).OnlyEnforceIf(z)
            model.AddBoolOr([flow_c[0, j].Not(), flow_s[r, j].Not()]).OnlyEnforceIf(z.Not())
            model.Add(carrier_finish[j] >= shuttle_finish[r] + base_from_r).OnlyEnforceIf(z)
            model.Add(carrier_finish[j] >= travel_time_scaled.get((n, r + n), M_scaled) + base_from_r).OnlyEnforceIf(z)
            model.Add(
                dist_var[j] == (
                    dist_scaled.get((n, r + n), M_dist_scaled)
                    + dist_scaled.get((r, j), M_dist_scaled)
                    + dist_scaled.get(j, M_dist_scaled)
                )
            ).OnlyEnforceIf(z)
            case_terms.append(z)

        for i in R:
            if i == j:
                continue
            z = z_case4[j, i]
            model.AddBoolAnd([flow_c[i, j], flow_s[0, j]]).OnlyEnforceIf(z)
            model.AddBoolOr([flow_c[i, j].Not(), flow_s[0, j].Not()]).OnlyEnforceIf(z.Not())
            model.Add(
                carrier_finish[j] >= carrier_finish[i] + travel_time_scaled.get((i + n, n), M_scaled) + base_from_depot
            ).OnlyEnforceIf(z)
            model.Add(
                dist_var[j] == (
                    dist_scaled.get((i + n, n), M_dist_scaled)
                    + dist_scaled.get((0, j), M_dist_scaled)
                    + dist_scaled.get(j, M_dist_scaled)
                )
            ).OnlyEnforceIf(z)
            case_terms.append(z)

        model.Add(sum(case_terms) == 1)

    variables = {
        "x": x,
        "flow_c": flow_c,
        "flow_s": flow_s,
        "carrier_finish": carrier_finish,
        "shuttle_finish": shuttle_finish,
        "delta": delta,
        "dist_var": dist_var,
        "z_case1": z_case1,
        "z_case2": z_case2,
        "z_case3": z_case3,
        "z_case4": z_case4,
        "total_distance": total_distance,
        "total_tardiness": total_tardiness,
    }
    return model, variables


def _apply_fixed_routes(model: cp_model.CpModel, x: Dict[Tuple[int, int, str], cp_model.IntVar], fixed_routes: Dict[str, List[int]], vehicles: set):
    print("--- 3. Applying fixed routes ---")
    for k, path in fixed_routes.items():
        if k not in vehicles:
            continue
        for start_node, end_node in zip(path[:-1], path[1:]):
            key = (start_node, end_node, k)
            if key in x:
                model.Add(x[key] == 1)
            else:
                print(f"Warning: Arc ({start_node}, {end_node}) for vehicle {k} not in x variables.")


def _extract_and_package_solution(
    solver: cp_model.CpSolver,
    status: int,
    data: Dict[str, Any],
    vars: Dict[str, Any],
    time_limit: Optional[int],
) -> List:
    print("\n" + "=" * 30)
    print(f"Solution found! (Status: {solver.StatusName(status)})")
    print(f"Objective (scaled int): {solver.ObjectiveValue():.2f}")
    print(f"Best bound (scaled int): {solver.BestObjectiveBound():.2f}")
    print(f"Runtime: {solver.WallTime():.2f}s")
    print("=" * 30 + "\n")

    R, N, n = data["R"], data["N"], data["n"]
    depot_start, depot_end = data["depot_start"], data["depot_end"]
    vehicle_list = data["vehicle_list"]
    shuttles = data["shuttles"]
    arcs = data["arcs"]
    omega = data["omega"]
    time_scale = data["time_scale"]
    distance_scale = data["distance_scale"]
    obj_scale = data["obj_scale"]

    x = vars["x"]
    carrier_finish, shuttle_finish = vars["carrier_finish"], vars["shuttle_finish"]
    delta, dist_var = vars["delta"], vars["dist_var"]
    z_case1, z_case2, z_case3, z_case4 = vars["z_case1"], vars["z_case2"], vars["z_case3"], vars["z_case4"]
    total_distance, total_tardiness = vars["total_distance"], vars["total_tardiness"]

    tardiness_values = {i: solver.Value(delta[i]) / time_scale for i in R if solver.Value(delta[i]) > 0}
    if tardiness_values:
        print("\nTardiness observed:")
        for i, val in tardiness_values.items():
            print(f"  - Task {i}: {val:.2f} time units late")

    dist_values = {i: solver.Value(dist_var[i]) / distance_scale for i in R if solver.Value(dist_var[i]) > 0}
    if dist_values:
        print("\nDistance contribution by task:")
        for i, val in dist_values.items():
            print(f"  - Task {i}: {val:.2f}")

    print("\nSelected synchronization case:")
    for j in R:
        if solver.Value(z_case2[j]) > 0:
            print(f"  - Task {j}: case2 (carrier from depot, shuttle from depot)")
            continue
        found_case = False
        for key, var in z_case3.items():
            jj, r = key
            if jj == j and solver.Value(var) > 0:
                print(f"  - Task {j}: case3 (carrier from depot, shuttle from task {r})")
                found_case = True
                break
        if found_case:
            continue
        for key, var in z_case4.items():
            jj, i = key
            if jj == j and solver.Value(var) > 0:
                print(f"  - Task {j}: case4 (carrier from task {i}, shuttle from depot)")
                found_case = True
                break
        if found_case:
            continue
        for key, var in z_case1.items():
            jj, i, r = key
            if jj == j and solver.Value(var) > 0:
                print(f"  - Task {j}: case1 (carrier from task {i}, shuttle from task {r})")
                break
    print("\n" + "-" * 30 + "\n")

    for k in vehicle_list:
        print(f"Route for vehicle {k}:")
        current_node = depot_start
        path = [depot_start]
        for _ in range(len(N) + 2):
            found_next = False
            for j in N | {depot_end}:
                if j != current_node and (current_node, j) in arcs and solver.Value(x[current_node, j, k]) > 0:
                    path.append(j)
                    current_node = j
                    found_next = True
                    break
            if not found_next or current_node == depot_end:
                break
        print(" -> ".join(map(str, path)))

        for node in path:
            if node in R:
                finish_time = shuttle_finish[node] if k in shuttles else carrier_finish[node]
                print(f"  - Node {node}: Finish @ {solver.Value(finish_time) / time_scale:.2f}")

    total_distance_real = solver.Value(total_distance) / distance_scale
    total_tardiness_real = solver.Value(total_tardiness) / time_scale
    objective_real = solver.ObjectiveValue() / obj_scale

    costs = [objective_real]
    dist_costs = [omega * total_distance_real]
    delay_costs = [(1 - omega) * total_tardiness_real]

    raw_routes = []
    all_vehicle_routes = []
    for k in vehicle_list:
        path = []
        if any((depot_start, j, k) in x and solver.Value(x[depot_start, j, k]) > 0 for j in R):
            current_node = depot_start
            while current_node != depot_end and len(path) <= n:
                next_node = None
                for j in N | {depot_end}:
                    if j != current_node and (current_node, j) in arcs and solver.Value(x[current_node, j, k]) > 0:
                        next_node = j
                        break
                if next_node is None:
                    break
                if next_node in R:
                    path.append(next_node - 1)
                current_node = next_node

        padded_route = path + [-1] * (n - len(path))
        all_vehicle_routes.append(padded_route)

    raw_routes.append([all_vehicle_routes])

    actual_runtime = solver.WallTime()
    data_package = _calculate_and_package_metrics(costs, dist_costs, delay_costs, actual_runtime)
    data_package[0] = raw_routes
    return data_package


def _package_empty_solution(n: int, vehicle_list: List[str], runtime: float) -> List:
    costs = [float("inf")]
    dist_costs = [float("inf")]
    delay_costs = [float("inf")]

    empty_route = [-1] * n
    all_vehicle_routes = [empty_route for _ in vehicle_list]
    raw_routes = [[[all_vehicle_routes]]]

    data_package = _calculate_and_package_metrics(costs, dist_costs, delay_costs, runtime)
    data_package[0] = raw_routes
    return data_package


def solve_vrp_model(
    fixed_routes: Optional[Dict[str, List[int]]] = None,
    instance_name: Optional[str] = None,
    time_limit: Optional[int] = None,
    log_search_progress: bool = False,
    num_search_workers: int = 8,
    random_seed: Optional[int] = None,
) -> List:
    if instance_name is None:
        raise ValueError("instance_name must be provided")

    data = _load_and_preprocess_data(instance_name)
    model, variables = _build_cp_sat_model(data)

    if fixed_routes:
        _apply_fixed_routes(model, variables["x"], fixed_routes, data["vehicles"])

    solver = cp_model.CpSolver()
    if time_limit:
        solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = log_search_progress
    solver.parameters.num_search_workers = num_search_workers
    if random_seed is not None and hasattr(solver.parameters, "random_seed"):
        solver.parameters.random_seed = random_seed

    print(f"--- 4. Starting model optimization (TimeLimit: {time_limit}s) ---")
    if log_search_progress:
        print("--- OR-Tools search logging is ENABLED ---")
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("--- 5. Extracting and Packaging Solution ---")
        return _extract_and_package_solution(solver, status, data, variables, time_limit)

    print("\n" + "!" * 30)
    print(f"Optimization stopped with no solution found (Status: {solver.StatusName(status)}).")
    print("!" * 30 + "\n")
    return _package_empty_solution(data["n"], data["vehicle_list"], solver.WallTime())


def solve_vrp_model_structured(
    fixed_routes: Optional[Dict[str, List[int]]] = None,
    instance_name: Optional[str] = None,
    time_limit: Optional[int] = None,
    log_search_progress: bool = False,
    num_search_workers: int = 8,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
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
        model, variables = _build_cp_sat_model(data)

        if fixed_routes:
            _apply_fixed_routes(model, variables["x"], fixed_routes, data["vehicles"])

        solver = cp_model.CpSolver()
        if time_limit:
            solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.log_search_progress = log_search_progress
        solver.parameters.num_search_workers = num_search_workers
        if random_seed is not None and hasattr(solver.parameters, "random_seed"):
            solver.parameters.random_seed = random_seed

        status = solver.Solve(model)
        result["status"] = solver.StatusName(status)
        result["runtime_s"] = float(solver.WallTime())
        result["is_optimal_proven"] = status == cp_model.OPTIMAL

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_scale = data["obj_scale"]
            distance_scale = data["distance_scale"]
            time_scale = data["time_scale"]
            objective_value = float(solver.ObjectiveValue()) / obj_scale
            best_bound = float(solver.BestObjectiveBound()) / obj_scale
            result["fitness"] = objective_value
            result["distance"] = float(solver.Value(variables["total_distance"])) / distance_scale
            result["tardiness"] = float(solver.Value(variables["total_tardiness"])) / time_scale
            result["best_bound"] = best_bound
            if abs(objective_value) > 1e-9:
                result["gap"] = abs(objective_value - best_bound) / abs(objective_value) * 100
            else:
                result["gap"] = 0.0
    except Exception as exc:
        result["error_message"] = str(exc)

    return result


def print_metrics_package(data_package: list, title: str):
    try:
        actual_runtime = data_package[3]
        min_delay = data_package[4]
        min_dist = data_package[5]
        best_cost = data_package[9]

        print("\n" + "-" * 30)
        print(f"--- {title} ---")
        print(f"  Best Cost (Fitness):   {best_cost:.4f}")
        print(f"  Distance Cost:         {min_dist:.4f}")
        print(f"  Tardiness Cost:        {min_delay:.4f}")
        print(f"  OR-Tools Actual Runtime: {actual_runtime:.4f}s")
        print("-" * 30)
    except IndexError:
        print(f"Error: Could not unpack metrics for {title}. Package was: {data_package}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="OR-Tools Exact Solver Runner")
    parser.add_argument(
        "--load_path",
        type=str,
        default="Synthetic_Dataset/size_10_uniform/T10_I10_uniform.xlsx",
        help="Instance xlsx path or basename under Instance/.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Maximum runtime in seconds for OR-Tools.",
    )
    parser.add_argument(
        "--log_search_progress",
        action="store_true",
        help="Print OR-Tools CP-SAT search progress logs.",
    )
    parser.add_argument(
        "--num_search_workers",
        type=int,
        default=8,
        help="Number of CP-SAT search workers.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional random seed for CP-SAT search.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("--- Starting Experiment ---")
    print("Algorithm:      OR-Tools CP-SAT")
    print(f"Instance Path: {args.load_path}")
    print(f"Time Limit:    {args.time_limit}s")
    print(f"Search Log:    {'ON' if args.log_search_progress else 'OFF'}")
    print(f"Workers:       {args.num_search_workers}")
    print(f"Random Seed:   {args.random_seed if args.random_seed is not None else 'None'}")
    print("-" * 30)

    start_time_total = time.time()

    data_package = solve_vrp_model(
        fixed_routes=None,
        instance_name=args.load_path,
        time_limit=args.time_limit,
        log_search_progress=args.log_search_progress,
        num_search_workers=args.num_search_workers,
        random_seed=args.random_seed,
    )

    print("\n" + "=" * 30)
    print("--- FINAL RESULTS ---")
    best_solution_code = data_package[0]
    print("Best Solution Code (Routes):")
    if best_solution_code and best_solution_code[0] and best_solution_code[0][0]:
        for i, route in enumerate(best_solution_code[0][0]):
            clean_route = [node for node in route if node != -1]
            if clean_route:
                print(f"  Vehicle {i + 1}: {clean_route}")
    else:
        print("  No solution routes found or model was infeasible/untimed.")

    print_metrics_package(data_package, "Metrics")
    print("=" * 30 + "\n")

    end_time_total = time.time()
    print("-" * 30)
    print(f"OR-Tools run finished in {end_time_total - start_time_total:.2f} seconds.")


if __name__ == "__main__":
    main()
