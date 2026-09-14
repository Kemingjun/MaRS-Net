# Methods

## MaRS-Net

`marsnet/` implements MaRS-Net for the Collaborative Carrier-Worker Scheduling Problem (CCWSP). The structured MDP represents a construction step as a task and a complete carrier-worker pair. Synchronized transitions maintain robot availability, and the action sequence induces the routes.

The policy uses dual-stream task/robot encoding and task-first hierarchical decoding. Static task embeddings are computed once per instance, while dynamic robot embeddings are recomputed from the locations and availability times after previously scheduled operations. A hard task-selection mask prevents duplicate scheduling; no mask bit is appended to the embeddings.

Internal identifiers such as `ahasp`, `visited_`, and `cur_time` remain unchanged for checkpoint and import compatibility. They refer to CCWSP, already scheduled tasks, and robot availability times, respectively. Construction steps are not shared physical execution timestamps.

## DRL Baselines

`baselines/drl/hdrl/` preserves the core vehicle-aware dispatch and route-context-aware decoding principle of HDRL, while adapting the state, action, and transition definitions to cooperative carrier-worker-task assignment.

`baselines/drl/tdrl/` preserves token-style state coding and GRU-based dynamic token updates, while adapting token transitions to synchronized cooperative execution states.

For both DRL baselines, the decoder is adapted with a carrier-worker coupler so that feasible carrier-worker pairs can be assigned to tasks under strict synchronization constraints.

## Conventional Baselines

`baselines/conventional/` includes exact/constraint-programming baselines and four metaheuristics:

- Gurobi MIP model.
- OR-Tools CP-SAT model.
- ALNS.
- IGA.
- DABC.
- DIWO.

The exact solvers are most suitable for small and medium instances. Metaheuristics are intended for scalable heuristic comparison.

## Release Coverage

The Webots video illustrates the execution workflow; the simulation world and controllers are not included.
