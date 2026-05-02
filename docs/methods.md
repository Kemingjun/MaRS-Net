# Methods

## MaRS-Net

`marsnet/` contains the final MaRS-Net model used in the revised manuscript. The model uses dual-stream task/robot encoding and task-first cooperative assignment without completion-mask augmentation.

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
