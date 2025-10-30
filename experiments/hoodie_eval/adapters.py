from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import torch

from decision_makers import Agent, AllHorizontal, AllLocal, AllVertical, Random, RoundRobin

try:  # Optional dependency; rule based policy is used in certain baselines.
    from decision_makers.rule_based import RuleBased
except Exception:  # pragma: no cover - RuleBased may not exist in minimal setups.
    RuleBased = None  # type: ignore

from environment import Environment

from .config import AgentConfig, BaselineConfig, ScenarioConfig

ENVIRONMENT_KWARGS = [
    "static_frequency",
    "number_of_servers",
    "private_cpu_capacities",
    "public_cpu_capacities",
    "connection_matrix",
    "cloud_computational_capacity",
    "episode_time",
    "task_arrive_probabilities",
    "task_size_mins",
    "task_size_maxs",
    "task_size_distributions",
    "timeout_delay_mins",
    "timeout_delay_maxs",
    "timeout_delay_distributions",
    "priotiry_mins",
    "priotiry_maxs",
    "priotiry_distributions",
    "computational_density_mins",
    "computational_density_maxs",
    "computational_density_distributions",
    "drop_penalty_mins",
    "drop_penalty_maxs",
    "drop_penalty_distributions",
    "number_of_clouds",
]


def _deep_update(target: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_hyperparameters(scenario: ScenarioConfig) -> Dict[str, Any]:
    """
    Load the hyperparameter bundle referenced by ``scenario`` and apply overrides.
    """

    path = scenario.hyperparameters_path or Path("hyperparameters/hyperparameters.json")
    if not path.exists():
        raise FileNotFoundError(f"hyperparameters file not found: {path}")
    with path.open() as stream:
        bundle = json.load(stream)
    overrides = scenario.overrides or {}
    _deep_update(bundle, overrides)
    if scenario.horizon is not None:
        bundle["episode_time"] = scenario.horizon
    return bundle


def make_environment(hyperparameters: Mapping[str, Any]) -> Environment:
    """
    Instantiate :class:`Environment` using a hyperparameter dictionary.  Missing
    optional keys default to reasonable values if possible.
    """

    payload: Dict[str, Any] = {}
    for key in ENVIRONMENT_KWARGS:
        if key == "number_of_clouds":
            payload[key] = hyperparameters.get(key, 1)
        elif key in hyperparameters:
            payload[key] = hyperparameters[key]
    return Environment(**payload)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for deterministic runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


@dataclass
class SystemSpec:
    """Normalized specification produced from agent/baseline configs."""

    name: str
    is_learning: bool
    agent_config: Optional[AgentConfig] = None
    baseline_config: Optional[BaselineConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_agent(
    config: AgentConfig,
    env: Environment,
    hyperparameters: Mapping[str, Any],
    log_dir: Path,
) -> Agent:
    """
    Construct a DRL agent that is compatible with the HOODIE environment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir.mkdir(parents=True, exist_ok=True)

    hp = dict(hyperparameters)
    hp_overrides = config.overrides or {}
    _deep_update(hp, hp_overrides)
    if config.dueling is False:
        hp["dueling"] = False
    if config.lstm is False:
        hp["lstm_layers"] = 0
    if config.double is False:
        # The existing implementation approximates Double DQN via target net updates.
        # When disabled we increase replace_target_iter to a very large value.
        hp["replace_target_iter"] = hp.get("replace_target_iter", 50) * 1000

    scheduler_file = log_dir / "scheduler.pkl"
    if "scheduler_choice" not in hp:
        hp["scheduler_choice"] = "constant"

    state_dimensions, foreign_queues, number_of_actions = env.get_server_dimensions(0)
    lstm_shape = foreign_queues
    agent = Agent(
        id=0,
        state_dimensions=state_dimensions,
        lstm_shape=lstm_shape,
        number_of_actions=number_of_actions,
        hidden_layers=hp["hidden_layers"],
        lstm_layers=hp["lstm_layers"],
        lstm_time_step=hp.get("lstm_time_step", 1),
        dropout_rate=hp["dropout_rate"],
        dueling=hp.get("dueling", True),
        epsilon=config.epsilon_eval,
        epsilon_decrement=hp["epsilon_decrement"],
        epsilon_end=hp["epsilon_end"],
        gamma=hp["gamma"],
        learning_rate=hp["learning_rate"],
        scheduler_file=str(scheduler_file),
        loss_function=getattr(torch.nn, hp["loss_function"]),
        optimizer=getattr(torch.optim, hp["optimizer"]),
        checkpoint_folder=str(log_dir / "agent.pth"),
        save_model_frequency=hp.get("save_model_frequency", 100),
        update_weight_percentage=hp.get("update_weight_percentage", 1.0),
        memory_size=hp["memory_size"],
        batch_size=hp["batch_size"],
        replace_target_iter=hp["replace_target_iter"],
        device=device,
    )
    return agent


def build_baseline(baseline: BaselineConfig, env: Environment) -> Any:
    """
    Instantiate one of the lightweight baseline policies.

    The mapping between requested baseline names and concrete decision makers is
    handled here to keep `baselines.py` focused on high-level orchestration logic.
    """

    name = baseline.name.lower()
    state_dimensions, foreign_queues, number_of_actions = env.get_server_dimensions(0)

    if name in {"flc", "all_local", "local"}:
        return AllLocal()
    if name in {"vo", "vertical", "all_vertical"}:
        return AllVertical(number_of_actions=number_of_actions)
    if name in {"ho", "horizontal"}:
        return AllHorizontal(number_of_actions=number_of_actions)
    if name in {"ro", "random"}:
        return Random(number_of_actions=number_of_actions)
    if name in {"bco", "round_robin"}:
        return RoundRobin(number_of_actions=number_of_actions)
    if name in {"mleo", "rule_based"} and RuleBased is not None:
        local_cpu = env.servers[0].private_queue_computational_capacity
        foreign_cpus = env.get_foreign_cpus(0)
        return RuleBased(
            number_of_actions=number_of_actions,
            local_cpu=local_cpu,
            foreign_cpus=foreign_cpus,
        )
    raise ValueError(f"Unsupported baseline system: {baseline.name}")
