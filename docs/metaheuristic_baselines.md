# Metaheuristic Baseline Parameters and Operators

This document records the parameter settings and operators used by the four metaheuristic baselines released in this repository:

- `ALNS`: Adaptive Large Neighborhood Search
- `IGA`: Iterated Greedy Algorithm
- `DABC`: Discrete Artificial Bee Colony
- `DIWO`: Discrete Invasive Weed Optimization

The implementations are located in `baselines/conventional/metaheuristic/`, with shared solution, fitness, and neighborhood utilities in `baselines/conventional/Util/`. Compact result tables may report representative core operators, while this document lists the full default parameter settings and operator sets used in the released code.

## Summary

| Algorithm | Default parameters | Default operators |
| --- | --- | --- |
| ALNS | `mu=T_coefficient=10`, `d=ceil(0.3n)`, `l_s=15`, `rho=0.1`, `(sigma_1,sigma_2,sigma_3)=(33,13,9)` | random removal, worst-cost removal, worst-distance removal, worst-tardiness removal, Shaw removal; greedy repair, urgency-priority greedy repair, cost-priority greedy repair |
| IGA | `mu=T_coefficient=10`, `d=ceil(0.3n)` | referenced local search, random destruction, greedy repair |
| DABC | `PS=100`, `PS_e=50`, `PS_o=50`, `LIMIT=1000`, `Rep=20` | insertion neighborhood, task-swap neighborhood |
| DIWO | `PS_init=50`, `PS_max=100`, `S_min=1`, `S_max=100` | insertion neighborhood, task-swap neighborhood |

Here, `n` denotes the number of tasks. All four algorithms also support a configurable stopping rule through `--max_iterations` and `--time_limit`.

## ALNS

Source files:

- `baselines/conventional/metaheuristic/ALNS.py`
- `baselines/conventional/Util/ALNS_config.py`
- `baselines/conventional/Util/operators.py`

### Parameters

- `d=ceil(0.3n)`: number of tasks removed in each destroy-repair iteration.
- `mu=T_coefficient=10`: coefficient used by `get_T(instance)` to compute the simulated-annealing temperature.
- `l_s=15`: operator weight update interval.
- `rho=0.1`: reaction factor for adaptive operator weight updating.
- `(sigma_1,sigma_2,sigma_3)=(33,13,9)`: scores assigned to operators when they generate a global-best solution, an improving solution, or an accepted non-improving solution.
- `shaw_weight=0.3`: coefficient used in the Shaw relatedness removal operator.

### Default operators

The released ALNS samples one destroy operator and one repair operator according to adaptive operator weights. Compact summaries may group these operators as removal and greedy-insertion operators; the full default enabled set is listed below.

Destruction operators:

- `destroy_couple_random`: randomly removes `d` tasks from the current schedule.
- `destroy_couple_worst_cost`: removes tasks with high marginal objective contribution.
- `destroy_couple_worst_distance`: removes tasks with high travel-distance contribution.
- `destroy_couple_worst_tardiness`: removes tasks with high tardiness contribution.
- `destroy_couple_shaw`: removes related tasks according to spatial, temporal, and structural similarity.

Repair operators:

- `repair_couple_greedy`: reinserts each removed task by exhaustively evaluating feasible CR-WR insertion positions and selecting the best objective value.
- `repair_couple_greedy_urgency_priority`: sorts removed tasks by urgency before greedy reinsertion.
- `repair_couple_greedy_cost_priority`: sorts removed tasks by current cost contribution before greedy reinsertion.

## IGA

Source files:

- `baselines/conventional/metaheuristic/IGA.py`
- `baselines/conventional/Util/ALNS_config.py`
- `baselines/conventional/Util/operators.py`

### Parameters

- `d=ceil(0.3n)`: number of destroyed tasks in each destruction-repair step.
- `mu=T_coefficient=10`: coefficient used by `get_T(instance)` to compute the simulated-annealing temperature for accepting worse solutions.

### Operators

- Referenced local search: iteratively removes a task and tests feasible CR-WR reinsertion positions; the first improving move is accepted.
- Random destruction: removes `d` tasks using `destroy_couple_random`.
- Greedy repair: reinserts removed tasks using `repair_couple_greedy`.

The IGA implementation alternates local search and destruction-repair, and accepts non-improving solutions using the same temperature function as ALNS.

## DABC

Source file:

- `baselines/conventional/metaheuristic/DABC.py`

### Parameters

- `PS=100`: total bee population size.
- `PS_e=50`: number of employed bees.
- `PS_o=50`: number of onlooker bees.
- `LIMIT=1000`: abandonment threshold for converting a stagnated nectar source into scout search.
- `Rep=20`: number of repeated onlooker update rounds in each iteration.

### Operators

- Insertion neighborhood: removes a task from its current CR-WR route positions and reinserts it into another feasible position.
- Task-swap neighborhood: exchanges two task positions while preserving feasibility after decoding.

The employed-bee phase applies a random neighborhood move to each nectar source. The onlooker phase selects nectar sources using roulette-wheel selection and repeats the neighborhood update `Rep` times. Stagnated nectar sources whose search counter exceeds `LIMIT` are replaced by scout-generated solutions.

## DIWO

Source file:

- `baselines/conventional/metaheuristic/DIWO.py`

### Parameters

- `PS_init=50`: initial weed population size.
- `PS_max=100`: maximum retained population size after selection.
- `S_min=1`: minimum number of offspring seeds generated by a weed.
- `S_max=100`: maximum number of offspring seeds generated by a weed.

### Operators

- Insertion neighborhood: generates a seed solution by moving a task to another feasible insertion position.
- Task-swap neighborhood: generates a seed solution by swapping two tasks.

DIWO starts from one nearest-insertion solution and additional random solutions. In each iteration, each weed generates a fitness-dependent number of seeds; fitter weeds produce more seeds. The parent and seed populations are merged, sorted by objective value, and truncated to `PS_max`.
