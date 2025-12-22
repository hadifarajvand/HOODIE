from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from environment import Environment

from . import metrics
from .adapters import load_hyperparameters, make_environment, seed_everything
from .baselines import BaselinePolicy
from .config import BaselineConfig, HoodieEvalConfig
from .instrumentation import attach_instrumentation
from .events import EventRecorder
from .io import (
    ArtifactLayout,
    resolve_layout,
    write_config_snapshot,
    write_events,
    write_metrics,
    write_training_log,
)
from .trainer import train_and_evaluate_agent


@dataclass
class RunResult:
    """Holds raw events and aggregate metrics for a single (system, seed) pair."""

    system: str
    seed: int
    metrics: metrics.AggregateMetrics
    events: List[Dict[str, Any]] = field(default_factory=list)
    metrics_path: Optional[Path] = None
    events_path: Optional[Path] = None


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


def _agent_variants(agent_config) -> List[Tuple[str, Dict[str, Any]]]:
    variants: List[Tuple[str, Dict[str, Any]]] = [(agent_config.name, {})]
    for entry in agent_config.ablations:
        name = entry.get("name")
        if not name:
            continue
        overrides: Dict[str, Any] = {}
        for key in ("dueling", "double", "lstm", "episodes", "evaluation_episodes", "lookback_window"):
            if key in entry:
                overrides[key] = entry[key]
        if "overrides" in entry:
            overrides.setdefault("overrides", {}).update(entry["overrides"])
        variants.append((name, overrides))
    return variants


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
    if config.agent:
        variants = _agent_variants(config.agent)
        for seed in config.scenario.seeds:
            for variant_name, variant_overrides in variants:
                result = train_and_evaluate_agent(
                    config.agent,
                    hyperparameters,
                    layout,
                    seed,
                    variant_name,
                    variant_overrides,
                )
                write_metrics(result.metrics.to_dict(), result.metrics_path)
                write_events(result.events, result.events_path)
                write_training_log(
                    [
                        {
                            "episode": entry.episode,
                            "reward": entry.reward,
                            "epsilon": entry.epsilon,
                            "loss": entry.loss,
                        }
                        for entry in result.log
                    ],
                    layout.training_log_path(result.system_name, result.seed),
                )
                results.append(
                    RunResult(
                        system=result.system_name,
                        seed=result.seed,
                        metrics=result.metrics,
                        events=result.events,
                        metrics_path=result.metrics_path,
                        events_path=result.events_path,
                    )
                )

    return results
