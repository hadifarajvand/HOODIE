from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional
    yaml = None  # type: ignore

PathLike = Union[str, Path]


def _coerce_path(value: Optional[PathLike]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    return path.expanduser().resolve()


@dataclass
class AgentConfig:
    """Configuration for the HOODIE DRL agent."""

    name: str = "hoodie"
    episodes: int = 2000
    validate_every: int = 100
    dueling: bool = True
    double: bool = True
    lstm: bool = True
    lookback_window: int = 4
    epsilon_eval: float = 0.05
    checkpoint_path: Optional[Path] = None
    hyperparameters_path: Optional[Path] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.checkpoint_path is not None:
            data["checkpoint_path"] = str(self.checkpoint_path)
        if self.hyperparameters_path is not None:
            data["hyperparameters_path"] = str(self.hyperparameters_path)
        return data


@dataclass
class BaselineConfig:
    """Configuration for a non-learning baseline system."""

    name: str
    seed: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioConfig:
    """
    Specifies an environment/scenario setup used for both HOODIE and baseline runs.

    The scenario is defined by a base hyperparameters file plus optional overrides.
    """

    name: str
    hyperparameters_path: Optional[Path] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    seeds: Sequence[int] = field(default_factory=lambda: [0])
    horizon: Optional[int] = None  # Episode length override per evaluation

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.hyperparameters_path is not None:
            data["hyperparameters_path"] = str(self.hyperparameters_path)
        data["seeds"] = list(self.seeds)
        return data


@dataclass
class OutputConfig:
    """Controls how artifacts and intermediates are stored."""

    artifacts_root: Path = Path("experiments/hoodie_eval/artifacts")
    stamp: Optional[str] = None
    keep_raw_events: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["artifacts_root"] = str(self.artifacts_root)
        return data


@dataclass
class HoodieEvalConfig:
    """Top-level configuration tying together agent, baselines, and scenarios."""

    scenario: ScenarioConfig
    agent: Optional[AgentConfig] = None
    baselines: Sequence[BaselineConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "agent": self.agent.to_dict() if self.agent else None,
            "baselines": [baseline.to_dict() for baseline in self.baselines],
            "output": self.output.to_dict(),
        }

    @property
    def all_systems(self) -> List[str]:
        systems: List[str] = []
        if self.agent is not None:
            systems.append(self.agent.name)
        systems.extend(baseline.name for baseline in self.baselines)
        return systems


def _resolve_agent(data: MutableMapping[str, Any]) -> Optional[AgentConfig]:
    if data is None:
        return None
    checkpoint = _coerce_path(data.get("checkpoint_path"))
    if checkpoint is not None:
        data["checkpoint_path"] = checkpoint
    hyperparameters = _coerce_path(data.get("hyperparameters_path"))
    if hyperparameters is not None:
        data["hyperparameters_path"] = hyperparameters
    return AgentConfig(**data)


def _resolve_baselines(data: Optional[Iterable[Mapping[str, Any]]]) -> List[BaselineConfig]:
    if not data:
        return []
    baselines: List[BaselineConfig] = []
    for item in data:
        baselines.append(BaselineConfig(**dict(item)))
    return baselines


def _resolve_scenario(data: Mapping[str, Any]) -> ScenarioConfig:
    payload = dict(data)
    hyperparameters_path = _coerce_path(payload.get("hyperparameters_path"))
    if hyperparameters_path is not None:
        payload["hyperparameters_path"] = hyperparameters_path
    seeds = payload.get("seeds")
    if seeds is not None and not isinstance(seeds, Sequence):
        payload["seeds"] = list(seeds)
    return ScenarioConfig(**payload)


def _resolve_output(data: Optional[Mapping[str, Any]]) -> OutputConfig:
    if data is None:
        return OutputConfig()
    payload = dict(data)
    if "artifacts_root" in payload:
        payload["artifacts_root"] = _coerce_path(payload["artifacts_root"]) or OutputConfig().artifacts_root
    return OutputConfig(**payload)


def _loads_config_dict(raw: Mapping[str, Any]) -> HoodieEvalConfig:
    scenario = _resolve_scenario(raw["scenario"])
    agent_data = raw.get("agent")
    agent = _resolve_agent(dict(agent_data)) if agent_data else None
    baselines = _resolve_baselines(raw.get("baselines"))
    output = _resolve_output(raw.get("output"))
    return HoodieEvalConfig(scenario=scenario, agent=agent, baselines=baselines, output=output)


def parse_config_payload(payload: Mapping[str, Any]) -> HoodieEvalConfig:
    """Construct a `HoodieEvalConfig` from an in-memory dictionary."""

    if "scenario" not in payload:
        raise ValueError("configuration payload must include a 'scenario' section")
    return _loads_config_dict(dict(payload))


def loads_config(text: Union[str, bytes], fmt: Optional[str] = None) -> HoodieEvalConfig:
    """
    Load configuration from a YAML/JSON string.

    Parameters
    ----------
    text:
        Configuration content.
    fmt:
        Optional explicit format (`\"json\"` or `\"yaml\"`). When omitted the loader
        attempts YAML first (if PyYAML is available) then JSON.
    """

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    if fmt is None and yaml is not None:
        try:
            data = yaml.safe_load(text)
        except Exception:
            data = None
        else:
            if isinstance(data, Mapping):
                return _loads_config_dict(dict(data))
    if fmt == "yaml" and yaml is None:
        raise RuntimeError("pyyaml is not installed; cannot parse YAML configuration")
    if fmt == "yaml" and yaml is not None:
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise TypeError("expected a mapping at the root of the configuration")
        return _loads_config_dict(dict(data))

    data = json.loads(text)
    if not isinstance(data, Mapping):
        raise TypeError("expected a mapping at the root of the configuration")
    return _loads_config_dict(dict(data))


def load_config(path: PathLike) -> HoodieEvalConfig:
    """
    Load configuration from a file path.  YAML is preferred when available,
    otherwise JSON is expected.
    """

    path = _coerce_path(path)
    if path is None:
        raise ValueError("configuration path may not be None")
    content = path.read_text()
    suffix = path.suffix.lower()
    fmt: Optional[str]
    if suffix in {".yaml", ".yml"}:
        fmt = "yaml"
    elif suffix == ".json":
        fmt = "json"
    else:
        fmt = None
    config = loads_config(content, fmt=fmt)
    # Derived defaults: prefer config-local hyperparameters file if relative
    if config.agent and config.agent.hyperparameters_path is None and config.scenario.hyperparameters_path:
        config.agent.hyperparameters_path = config.scenario.hyperparameters_path
    return config
