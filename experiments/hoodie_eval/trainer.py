from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .agents import AgentBundle, TelemetryAgent, build_telemetry_agents
from .adapters import make_environment, seed_everything, _deep_update
from .events import EventRecorder
from .instrumentation import attach_instrumentation
from .metrics import compute_metrics, AggregateMetrics


@dataclass
class TrainingLogEntry:
    episode: int
    reward: float
    epsilon: float
    loss: float


@dataclass
class AgentTrainingResult:
    agents: List[TelemetryAgent]
    log: List[TrainingLogEntry]
    metrics: AggregateMetrics
    events: List[Dict[str, Any]]
    metrics_path: Path
    events_path: Path
    system_name: str
    seed: int


def _build_variant_config(agent_config, variant_overrides: Mapping[str, Any]):
    payload = agent_config.to_dict()
    _deep_update(payload, variant_overrides)
    from .config import AgentConfig  # local import to avoid cycle

    return AgentConfig(
        name=payload.get("name", agent_config.name),
        episodes=payload.get("episodes", agent_config.episodes),
        evaluation_episodes=payload.get("evaluation_episodes", agent_config.evaluation_episodes),
        validate_every=payload.get("validate_every", agent_config.validate_every),
        dueling=payload.get("dueling", agent_config.dueling),
        double=payload.get("double", agent_config.double),
        lstm=payload.get("lstm", agent_config.lstm),
        lookback_window=payload.get("lookback_window", agent_config.lookback_window),
        epsilon_eval=payload.get("epsilon_eval", agent_config.epsilon_eval),
        checkpoint_path=payload.get("checkpoint_path"),
        hyperparameters_path=payload.get("hyperparameters_path"),
        overrides=payload.get("overrides", {}),
        ablations=payload.get("ablations", []),
        log_dir=payload.get("log_dir"),
    )


def train_and_evaluate_agent(
    agent_config,
    hyperparameters: Mapping[str, Any],
    layout,
    seed: int,
    system_name: str,
    variant_overrides: Mapping[str, Any],
) -> AgentTrainingResult:
    variant_config = _build_variant_config(agent_config, variant_overrides)
    seed_everything(seed)
    env = make_environment(hyperparameters)

    log_dir = layout.system_root(system_name, seed) / "models"
    bundle: AgentBundle = build_telemetry_agents(env, variant_config, hyperparameters, log_dir)

    training_log: List[TrainingLogEntry] = []

    for episode in range(variant_config.episodes):
        observations, done, _ = env.reset()
        local_obs, queue_obs = observations
        for agent in bundle.agents:
            agent.reset_lstm_history()

        episode_reward = 0.0
        time_step = 0
        while not done:
            actions = [
                bundle.agents[i].choose_action(local_obs[i], queue_obs[i])
                for i in range(env.number_of_servers)
            ]
            observations, rewards, done, info = env.step(actions)
            local_next, queue_next = observations
            for i in range(env.number_of_servers):
                bundle.agents[i].store_transitions(
                    state=local_obs[i],
                    lstm_state=queue_obs[i],
                    action=actions[i],
                    reward=rewards[i],
                    new_state=local_next[i],
                    new_lstm_state=queue_next[i],
                    done=done,
                )
            local_obs, queue_obs = local_next, queue_next
            episode_reward += float(np.sum(rewards))
            time_step += 1

        losses = []
        for agent in bundle.agents:
            agent.learn()
            if getattr(agent, "loss_history", None):
                losses.append(agent.loss_history[-1])
        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_epsilon = float(np.mean([agent.get_epsilon() for agent in bundle.agents]))
        training_log.append(
            TrainingLogEntry(
                episode=episode,
                reward=episode_reward,
                epsilon=mean_epsilon,
                loss=mean_loss,
            )
        )

    for agent in bundle.agents:
        agent.epsilon = variant_config.epsilon_eval
        agent.store_model(path=str(log_dir / f"agent_{agent.id}.pth"))
        agent.reset_lstm_history()

    combined_events: List[Dict[str, Any]] = []
    metric_samples: List[AggregateMetrics] = []

    for eval_episode in range(variant_config.evaluation_episodes):
        seed_everything(seed + eval_episode + 1000)
        evaluation_env = make_environment(hyperparameters)
        recorder = EventRecorder()
        attach_instrumentation(evaluation_env, recorder)

        observations, done, _ = evaluation_env.reset()
        local_obs, queue_obs = observations
        for agent in bundle.agents:
            agent.reset_lstm_history()
        time_step = 0
        recorder.record_arrivals(time_step, evaluation_env.tasks, range(evaluation_env.number_of_servers))
        recorder.record_queue_lengths(evaluation_env, time_step)

        while not done:
            actions = [
                bundle.agents[i].choose_action(local_obs[i], queue_obs[i])
                for i in range(evaluation_env.number_of_servers)
            ]
            recorder.record_actions(time_step, actions)
            observations, rewards, done, info = evaluation_env.step(actions)
            recorder.record_info(time_step, info)
            local_obs, queue_obs = observations
            time_step += 1
            recorder.record_arrivals(time_step, evaluation_env.tasks, range(evaluation_env.number_of_servers))
            recorder.record_queue_lengths(evaluation_env, time_step)

        metric_samples.append(
            compute_metrics(recorder.events, episode_time=hyperparameters["episode_time"])
        )
        combined_events.extend(recorder.events)

    def average_dict(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
        keys = set().union(*(d.keys() for d in dicts))
        return {key: float(np.mean([d.get(key, 0.0) for d in dicts])) for key in keys}

    aggregate_metrics = AggregateMetrics(
        mean_latency=float(np.mean([m.mean_latency for m in metric_samples])) if metric_samples else 0.0,
        load=float(np.mean([m.load for m in metric_samples])) if metric_samples else 0.0,
        p95_latency=float(np.mean([m.p95_latency for m in metric_samples])) if metric_samples else 0.0,
        drop_rate=float(np.mean([m.drop_rate for m in metric_samples])) if metric_samples else 0.0,
        throughput=float(np.mean([m.throughput for m in metric_samples])) if metric_samples else 0.0,
        offload_ratio_cloud=float(np.mean([m.offload_ratio_cloud for m in metric_samples])) if metric_samples else 0.0,
        offload_ratio_edge=float(np.mean([m.offload_ratio_edge for m in metric_samples])) if metric_samples else 0.0,
        avg_queue_lengths=average_dict([m.avg_queue_lengths for m in metric_samples]) if metric_samples else {},
        peak_queue_lengths=average_dict([m.peak_queue_lengths for m in metric_samples]) if metric_samples else {},
    )

    system_root = layout.system_root(system_name, seed)
    metrics_path = layout.metrics_path(system_name, seed)
    events_path = layout.events_path(system_name, seed)
    return AgentTrainingResult(
        agents=bundle.agents,
        log=training_log,
        metrics=aggregate_metrics,
        events=combined_events,
        metrics_path=metrics_path,
        events_path=events_path,
        system_name=system_name,
        seed=seed,
    )
