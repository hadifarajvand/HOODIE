from __future__ import annotations

from typing import List, Sequence

import numpy as np

from environment import Environment

from .adapters import build_baseline, seed_everything
from .config import BaselineConfig

import math


class MinimumLatencyEstimator:
    """
    Estimate completion latency for all available actions (local, horizontal peers,
    cloud) and select the minimum.
    """

    def __init__(self, env: Environment, number_of_actions: int, server_id: int) -> None:
        self.env = env
        self.number_of_actions = number_of_actions
        self.server_id = server_id

    def _local_latency(self, task, server) -> float:
        if task is None:
            return 0.0
        private_wait, _ = server.get_waiting_times()
        process_capacity = server.private_queue_computational_capacity
        process_per_slot = process_capacity / task.get_density()
        processing_time = math.ceil(task.get_size() / max(process_per_slot, 1e-6))
        return float(private_wait + processing_time)

    def _horizontal_latency(self, task, server, target_server_id: int) -> float:
        if task is None:
            return float("inf")
        if target_server_id == self.server_id:
            return self._local_latency(task, server)
        offloading_capacity = server.offloading_capacities.get(target_server_id)
        if not offloading_capacity:
            return float("inf")
        _, offload_wait = server.get_waiting_times()
        transmit_time = math.ceil(task.get_size() / max(offloading_capacity, 1e-6))
        target_server = self.env.servers[target_server_id]
        public_manager = target_server.public_queue_manager
        queue_length = public_manager.get_public_queue_server_length(self.server_id)
        public_capacity = target_server.public_queues_computational_capacity
        process_per_slot = public_capacity / task.get_density()
        processing_time = math.ceil(task.get_size() / max(process_per_slot, 1e-6))
        queue_wait = math.ceil(queue_length / max(process_per_slot, 1e-6))
        return float(offload_wait + transmit_time + queue_wait + processing_time)

    def _cloud_latency(self, task, server) -> float:
        if task is None:
            return float("inf")
        _, offload_wait = server.get_waiting_times()
        cloud_capacity = self.env.cloud.public_queue_manager.computational_capacity
        transmit_capacity = server.offloading_capacities.get(self.env.number_of_servers)
        if not transmit_capacity:
            transmit_capacity = self.env.connection_matrix[server.id][-1]
        transmit_time = math.ceil(task.get_size() / max(transmit_capacity, 1e-6))
        public_manager = self.env.cloud.public_queue_manager
        queue_length = public_manager.get_public_queue_server_length(self.server_id)
        process_per_slot = cloud_capacity / task.get_density()
        queue_wait = math.ceil(queue_length / max(process_per_slot, 1e-6))
        processing_time = math.ceil(task.get_size() / max(process_per_slot, 1e-6))
        return float(offload_wait + transmit_time + queue_wait + processing_time)

    def choose_action(self, local_obs, queue_obs, *, env=None, server_id=None) -> int:
        env = env or self.env
        server_id = self.server_id if server_id is None else server_id
        server = env.servers[server_id]
        task = env.tasks[server_id]
        if task is None:
            return 0
        matchmaker = env.matchmakers[server_id]
        latencies = []
        for target in matchmaker.possible_actions:
            if target == server_id:
                latencies.append(self._local_latency(task, server))
            elif target == env.number_of_servers:
                latencies.append(self._cloud_latency(task, server))
            else:
                latencies.append(self._horizontal_latency(task, server, int(target)))
        best_index = int(np.argmin(latencies))
        return best_index


class BaselinePolicy:
    """Thin wrapper that drives a baseline decision maker across all edge agents."""

    def __init__(self, env: Environment, config: BaselineConfig):
        self.env = env
        self.config = config
        self.seed = config.seed
        if self.seed is not None:
            seed_everything(self.seed)
        name = (config.name or "").lower()
        if name in {"mleo", "minimum_latency", "minimum_latency_estimator"}:
            self.decision_makers = [
                MinimumLatencyEstimator(env, env.servers[i].get_number_of_actions(), i)
                for i in range(env.number_of_servers)
            ]
        else:
            self.decision_makers = [
                build_baseline(config, env, server_id=i) for i in range(env.number_of_servers)
            ]

    def reset(self) -> None:
        """Reset any per-episode state on the underlying decision makers."""

        for decision_maker in self.decision_makers:
            reset = getattr(decision_maker, "reset_lstm_history", None)
            if callable(reset):
                reset()

    def choose_actions(
        self,
        local_observations: Sequence[Sequence[float]],
        queue_observations: Sequence[Sequence[float]],
    ) -> List[int]:
        """Return one action per edge agent."""

        actions: List[int] = []
        for idx, decision_maker in enumerate(self.decision_makers):
            local_obs = local_observations[idx]
            queue_obs = queue_observations[idx]
            if hasattr(decision_maker, "choose_action"):
                actions.append(
                    decision_maker.choose_action(
                        local_obs,
                        queue_obs,
                        env=self.env,
                        server_id=idx,
                    )
                )
            else:  # pragma: no cover - defensive path
                raise AttributeError(f"baseline {self.config.name} lacks a choose_action method")
        return actions
