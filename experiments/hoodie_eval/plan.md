# HOODIE Evaluation Integration Plan

## Existing Modules To Reuse
- `environment.Environment`: full task/queue simulator with servers, cloud, matchmaker, and stochastic task generators.
- `environment.server.Server`, `environment.queues.*`, and `environment.task.Task`: queue semantics and FIFO processing that match HOODIE’s single-hop, two-stage model.
- `decision_makers.Agent`: DRL agent with Dueling DDQN + (optional) LSTM head, epsilon-greedy policy, replay buffer, and target network management.
- `decision_makers` baselines (`AllLocal`, `AllVertical`, `AllHorizontal`, `Random`, `RoundRobin`, `RuleBased`): starting point for FLC/VO/RO/HO/BCO variants.
- `hyperparameters/__main__.py`: CLI helper for generating topology/config bundles we can repurpose for scenario definitions.
- `topology_generators` helpers for constructing connection matrices when varying EA counts or link capacities.

## New Adapters & Utilities
- `experiments.hoodie_eval.config`: typed dataclasses for evaluation setup (seed, episode length, topology, queue/service capacities, DRL toggles).
- `experiments.hoodie_eval.adapters`:
  - Environment adapter that instantiates `Environment` from the config and instruments it to emit per-step events (arrivals, completions, drops, queue lengths, actions) without mutating core classes.
  - Agent adapter that enables Double/Dueling/LSTM options via hyperparameter presets and exposes a deterministic evaluation policy.
  - Wrapper around existing dummy decision makers to normalize action spaces for FLC/RO/VO/HO/BCO.
- `experiments.hoodie_eval.scenarios`: factories that leverage the config + topology generators to sweep load (P), EA count (N), and link/CPU capacities (R_H, R_V, f_*), plus dynamic traffic patterns.
- `experiments.hoodie_eval.baselines`: concrete implementations of the required baseline policies built on top of the adapters.
- `experiments.hoodie_eval.runner`: unified training/evaluation loop that drives the environment, records raw events, and returns structured results.
- `experiments.hoodie_eval.metrics`: transforms raw events into aggregate latency, p95, drop rate, throughput, offloading ratio, and queue statistics.
- `experiments.hoodie_eval.io`: deterministic artifact writer/loader (JSON metrics, Parquet events, cached config snapshot).
- `experiments.hoodie_eval.plots`: matplotlib exporters for latency/drop vs load, convergence, scalability, and ablation charts.
- `experiments.hoodie_eval.cli`: Typer-based CLI entrypoints (`run`, `compare`, `plots`) wrapping runner/metrics/plots.
- `experiments.hoodie_eval.paper_import`: loader to normalize any existing paper CSV/JSON logs into the standard artifact layout.

## Metrics & Artifact Strategy
- During each run the environment adapter logs per-task lifecycle timestamps, action decisions (DM(1)/DM(2)), queue lengths, and drop outcomes to an in-memory table seeded for determinism.
- `metrics.py` aggregates the logged events to compute:
  - Mean and p95 latency (completion time minus arrival).
  - Drop rate (dropped tasks / arrivals) and throughput (completed / episode time).
  - Offloading ratios (cloud vs edge) and queue length summaries.
- Artifacts are stored under `experiments/hoodie_eval/artifacts/<stamp>/<scenario>/` with:
  - `config.json` (resolved dataclass), `metrics.json`, and `events.parquet`.
  - `plots/*.png` for each regenerated figure (latency, drop rate, convergence, scalability, ablations).
- All RNGs (NumPy, Torch, Python) receive the configured seed to keep runs reproducible across reruns and comparisons.
