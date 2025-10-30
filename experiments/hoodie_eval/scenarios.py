from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .config import ScenarioConfig


@dataclass
class ScenarioSweep:
    """Container for a family of scenarios produced by a parametrised sweep."""

    name: str
    scenarios: Sequence[ScenarioConfig]


def load_sweep(
    base_config: ScenarioConfig,
    arrival_probabilities: Iterable[float],
    *,
    num_servers: Optional[int] = None,
) -> ScenarioSweep:
    """Generate a load sweep by varying the Bernoulli task arrival probability."""

    effective_servers = num_servers or base_config.overrides.get("number_of_servers")
    if effective_servers is None:
        effective_servers = 1

    scenarios: List[ScenarioConfig] = []
    for probability in arrival_probabilities:
        overrides: Dict[str, List[float]] = {
            "task_arrive_probabilities": [probability] * int(effective_servers)
        }
        scenario = ScenarioConfig(
            name=f"{base_config.name}_p{probability:.2f}",
            hyperparameters_path=base_config.hyperparameters_path,
            overrides={**base_config.overrides, **overrides},
            seeds=base_config.seeds,
            horizon=base_config.horizon,
        )
        scenarios.append(scenario)
    return ScenarioSweep(name=f"{base_config.name}_load", scenarios=scenarios)


def scalability_sweep(
    base_config: ScenarioConfig,
    edge_counts: Iterable[int],
    *,
    base_prob: Optional[float] = None,
) -> ScenarioSweep:
    """Sweep over the number of edge agents to evaluate scalability."""

    scenarios: List[ScenarioConfig] = []
    for count in edge_counts:
        overrides = dict(base_config.overrides)
        overrides["number_of_servers"] = count
        if base_prob is not None:
            overrides["task_arrive_probabilities"] = [base_prob] * count
        scenario = ScenarioConfig(
            name=f"{base_config.name}_n{count}",
            hyperparameters_path=base_config.hyperparameters_path,
            overrides=overrides,
            seeds=base_config.seeds,
            horizon=base_config.horizon,
        )
        scenarios.append(scenario)
    return ScenarioSweep(name=f"{base_config.name}_scalability", scenarios=scenarios)


def link_sweep(
    base_config: ScenarioConfig,
    horizontal_rates: Iterable[float],
    vertical_rates: Iterable[float],
) -> ScenarioSweep:
    """Vary horizontal/vertical capacities to observe communication sensitivity."""

    scenarios: List[ScenarioConfig] = []
    for rh in horizontal_rates:
        for rv in vertical_rates:
            overrides = dict(base_config.overrides)
            overrides["horizontal_capacity_override"] = rh
            overrides["vertical_capacity_override"] = rv
            scenario = ScenarioConfig(
                name=f"{base_config.name}_rh{rh:.2f}_rv{rv:.2f}",
                hyperparameters_path=base_config.hyperparameters_path,
                overrides=overrides,
                seeds=base_config.seeds,
                horizon=base_config.horizon,
            )
            scenarios.append(scenario)
    return ScenarioSweep(name=f"{base_config.name}_links", scenarios=scenarios)
