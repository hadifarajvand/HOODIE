from __future__ import annotations

from typing import List, Sequence

from environment import Environment

from .adapters import build_baseline, seed_everything
from .config import BaselineConfig


class BaselinePolicy:
    """Thin wrapper that drives a baseline decision maker across all edge agents."""

    def __init__(self, env: Environment, config: BaselineConfig):
        self.env = env
        self.config = config
        self.seed = config.seed
        if self.seed is not None:
            seed_everything(self.seed)
        self.decision_makers = [
            build_baseline(config, env) for _ in range(env.number_of_servers)
        ]

    def reset(self) -> None:
        """Reset any per-episode state on the underlying decision makers."""

        for decision_maker in self.decision_makers:
            reset = getattr(decision_maker, "reset_lstm_history", None)
            if callable(reset):
                reset()

    def choose_actions(
        self, local_observations: Sequence[Sequence[float]], queue_observations: Sequence[Sequence[float]]
    ) -> List[int]:
        """Return one action per edge agent."""

        actions: List[int] = []
        for idx, decision_maker in enumerate(self.decision_makers):
            local_obs = local_observations[idx]
            queue_obs = queue_observations[idx]
            if hasattr(decision_maker, "choose_action"):
                actions.append(decision_maker.choose_action(local_obs, queue_obs))
            else:  # pragma: no cover - defensive path
                raise AttributeError(f"baseline {self.config.name} lacks a choose_action method")
        return actions
