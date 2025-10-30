"""
HOODIE paper evaluation toolkit built on top of the existing simulator.

Public entrypoints are intentionally additive so that core behaviour of the
original packages remains untouched.  All functionality is reachable via
`experiments.hoodie_eval.cli`.
"""

from .config import (
    AgentConfig,
    BaselineConfig,
    HoodieEvalConfig,
    OutputConfig,
    ScenarioConfig,
    load_config,
    loads_config,
)
from .runner import RunResult, run_experiment

__all__ = [
    "AgentConfig",
    "BaselineConfig",
    "HoodieEvalConfig",
    "OutputConfig",
    "ScenarioConfig",
    "RunResult",
    "load_config",
    "loads_config",
    "run_experiment",
]
