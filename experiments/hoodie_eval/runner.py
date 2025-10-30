from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from environment import Environment

from . import metrics
from .adapters import load_hyperparameters, make_environment, seed_everything
from .baselines import BaselinePolicy
from .config import BaselineConfig, HoodieEvalConfig
from .instrumentation import attach_instrumentation
from .io import ArtifactLayout, resolve_layout, write_config_snapshot, write_events, write_metrics


@dataclass
class RunResult:
    """Holds raw events and aggregate metrics for a single (system, seed) pair."""

    system: str
    seed: int
    metrics: metrics.AggregateMetrics
    events: List[Dict[str, Any]] = field(default_factory=list)
    metrics_path: Optional[Path] = None
    events_path: Optional[Path] = None


class EventRecorder:
    """Collects task-level events during simulation for post-hoc analysis."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._next_task_id = 0
        self._task_metadata: Dict[int, Dict[str, Any]] = {}

    def _ensure_task_id(self, task: Any) -> int:
        if not hasattr(task, "_hoodie_id"):
            setattr(task, "_hoodie_id", self._next_task_id)
            self._next_task_id += 1
        return int(getattr(task, "_hoodie_id"))

    def record_arrivals(
        self, time_step: int, tasks: Sequence[Any], server_ids: Iterable[int]
    ) -> None:
        for server_id, task in zip(server_ids, tasks):
            if task is None:
                continue
            task_id = self._ensure_task_id(task)
            self._task_metadata[task_id] = {
                "arrival_time": getattr(task, "arrival_time", time_step),
                "origin": getattr(task, "get_origin_server_id", lambda: server_id)(),
            }
            event = {
                "event": "task_arrived",
                "time": time_step,
                "server": server_id,
                "task_id": task_id,
                "arrival_time": getattr(task, "arrival_time", time_step),
                "timeout": getattr(task, "timeout_delay", None),
                "size": getattr(task, "size", None),
                "priority": getattr(task, "priotiry", None),
            }
            self.events.append(event)

    def record_actions(self, time_step: int, actions: Sequence[int]) -> None:
        self.events.append(
            {
                "event": "actions",
                "time": time_step,
                "load": sum(action is not None for action in actions) / max(len(actions), 1),
                "actions": list(actions),
            }
        )

    def record_info(self, time_step: int, info: Mapping[str, Any]) -> None:
        self.events.append(
            {
                "event": "info",
                "time": time_step,
                "payload": dict(info),
            }
        )

    def record_queue_lengths(self, env: Environment, time_step: int) -> None:
        for server in env.servers:
            self.events.append(
                {
                    "event": "queue_length",
                    "time": time_step,
                    "queue": f"server_{server.id}_private",
                    "length": getattr(server.processing_queue, "queue_length", 0.0),
                }
            )
            self.events.append(
                {
                    "event": "queue_length",
                    "time": time_step,
                    "queue": f"server_{server.id}_offloading",
                    "length": getattr(server.offloading_queue, "queue_length", 0.0),
                }
            )
            for origin_id, public_queue in server.public_queue_manager.public_queues.items():
                self.events.append(
                    {
                        "event": "queue_length",
                        "time": time_step,
                        "queue": f"server_{server.id}_public_from_{origin_id}",
                        "length": getattr(public_queue, "queue_length", 0.0),
                    }
                )
        cloud_manager = env.cloud.public_queue_manager
        for origin_id, public_queue in cloud_manager.public_queues.items():
            self.events.append(
                {
                    "event": "queue_length",
                    "time": time_step,
                    "queue": f"cloud_public_from_{origin_id}",
                    "length": getattr(public_queue, "queue_length", 0.0),
                }
            )

    def record_completion(
        self,
        task: Any,
        finish_time: float,
        context: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> None:
        task_id = self._ensure_task_id(task)
        metadata = self._task_metadata.get(task_id, {})
        arrival_time = metadata.get("arrival_time", snapshot.get("arrival_time", finish_time))
        latency = float(finish_time - arrival_time)
        event = {
            "event": "task_completed",
            "task_id": task_id,
            "time": finish_time,
            "latency": latency,
            "origin": snapshot.get("origin", metadata.get("origin")),
            "target": snapshot.get("target"),
            "destination": "cloud" if context.get("executor_type") == "cloud" else "edge",
            "executor": context.get("server_id"),
            "queue_type": context.get("queue_type"),
        }
        self.events.append(event)
        self._task_metadata.pop(task_id, None)

    def record_drop(
        self,
        task: Any,
        penalty: float,
        context: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> None:
        task_id = self._ensure_task_id(task)
        queue = context.get("queue")
        drop_time = getattr(queue, "current_time", None)
        event = {
            "event": "task_dropped",
            "task_id": task_id,
            "time": drop_time,
            "penalty": penalty,
            "origin": snapshot.get("origin"),
            "target": snapshot.get("target"),
            "destination": "cloud" if context.get("queue_type") == "cloud_public" else "edge",
            "queue_type": context.get("queue_type"),
        }
        self.events.append(event)
        self._task_metadata.pop(task_id, None)


def _run_episode(env: Environment, policy: BaselinePolicy, recorder: EventRecorder) -> EventRecorder:
    time_step = 0
    policy.reset()
    observations, done, _ = env.reset()
    local_obs, queue_obs = observations
    recorder.record_arrivals(time_step, env.tasks, range(env.number_of_servers))
    recorder.record_queue_lengths(env, time_step)
    while not done:
        actions = policy.choose_actions(local_obs, queue_obs)
        recorder.record_actions(time_step, actions)
        observations, rewards, done, info = env.step(actions)
        recorder.record_info(time_step, info)
        local_obs, queue_obs = observations
        time_step += 1
        recorder.record_arrivals(time_step, env.tasks, range(env.number_of_servers))
        recorder.record_queue_lengths(env, time_step)
    return recorder


def _aggregate(recorder: EventRecorder, episode_time: int) -> metrics.AggregateMetrics:
    return metrics.compute_metrics(recorder.events, episode_time=episode_time)


def _run_baseline(
    config: HoodieEvalConfig,
    baseline: BaselineConfig,
    hyperparameters: Mapping[str, Any],
    layout: ArtifactLayout,
) -> List[RunResult]:
    results: List[RunResult] = []
    for seed in config.scenario.seeds:
        seed_everything(seed)
        env = make_environment(hyperparameters)
        policy = BaselinePolicy(env, baseline)
        recorder = EventRecorder()
        attach_instrumentation(env, recorder)
        recorder = _run_episode(env, policy, recorder)
        aggregate_metrics = _aggregate(recorder, episode_time=hyperparameters["episode_time"])
        metrics_path = layout.metrics_path(baseline.name, seed)
        events_path = layout.events_path(baseline.name, seed)
        write_metrics(aggregate_metrics.to_dict(), metrics_path)
        write_events(recorder.events, events_path)
        results.append(
            RunResult(
                system=baseline.name,
                seed=seed,
                metrics=aggregate_metrics,
                events=recorder.events,
                metrics_path=metrics_path,
                events_path=events_path,
            )
        )
    return results


def run_experiment(config: HoodieEvalConfig) -> List[RunResult]:
    """
    Execute the configured experiment (agent + baselines) and persist artifacts.
    """

    layout = resolve_layout(config.output, config.scenario)
    write_config_snapshot(config, layout)
    hyperparameters = load_hyperparameters(config.scenario)

    results: List[RunResult] = []

    if config.baselines:
        for baseline in config.baselines:
            results.extend(_run_baseline(config, baseline, hyperparameters, layout))

    # Agent execution is more involved (training + evaluation) and is implemented
    # in a follow-up iteration.

    return results
