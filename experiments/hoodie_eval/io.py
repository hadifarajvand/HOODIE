from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from .config import HoodieEvalConfig, OutputConfig, ScenarioConfig

try:  # pragma: no cover - pandas is optional at runtime
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # pragma: no cover - pyarrow is optional
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class ArtifactLayout:
    """Resolved artifact locations for a single scenario run."""

    root: Path
    scenario_root: Path
    plots_root: Path
    config_path: Path

    def system_root(self, system: str, seed: Optional[int] = None) -> Path:
        segment = system.lower()
        base = self.scenario_root / segment
        if seed is not None:
            base = base / f"seed_{seed}"
        return _ensure_dir(base)

    def metrics_path(self, system: str, seed: Optional[int] = None) -> Path:
        return self.system_root(system, seed) / "metrics.json"

    def events_path(self, system: str, seed: Optional[int] = None) -> Path:
        return self.system_root(system, seed) / "events.parquet"


def resolve_layout(
    output: OutputConfig, scenario: ScenarioConfig, stamp: Optional[str] = None
) -> ArtifactLayout:
    """Resolve an :class:`ArtifactLayout` based on the output configuration."""

    root = output.artifacts_root
    _ensure_dir(root)
    stamp_value = output.stamp or stamp or _default_stamp()
    stamp_root = root / stamp_value
    scenario_root = stamp_root / scenario.name
    plots_root = _ensure_dir(scenario_root / "plots")
    config_path = scenario_root / "config.json"
    return ArtifactLayout(
        root=stamp_root,
        scenario_root=_ensure_dir(scenario_root),
        plots_root=plots_root,
        config_path=config_path,
    )


def write_config_snapshot(config: HoodieEvalConfig, layout: ArtifactLayout) -> None:
    """Persist the resolved configuration for reproducibility."""

    payload = config.to_dict()
    layout.config_path.write_text(json.dumps(payload, indent=2))


def write_metrics(metrics: Mapping[str, Any], path: Path) -> None:
    """Write aggregate metrics as JSON."""

    content = json.dumps(metrics, indent=2)
    path.write_text(content)


def _write_parquet(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    if pd is not None:
        frame = pd.DataFrame(list(records))
        if frame.empty:
            # Ensure schema exists by creating an empty file with metadata.
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
            return
        frame.to_parquet(path)
        return
    if pq is not None and pa is not None:
        table = pa.Table.from_pylist(list(records))  # type: ignore[arg-type]
        pq.write_table(table, path)  # type: ignore[arg-type]
        return
    raise RuntimeError(
        "Neither pandas nor pyarrow is available; cannot write Parquet artifacts."
    )


def write_events(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write raw step/task events to Parquet."""

    if not records:
        # Create an empty struct with a simple schema to keep downstream code happy.
        empty_record: MutableMapping[str, Any] = {}
        _write_parquet([empty_record], path)
        return
    _write_parquet(records, path)


def load_metrics(path: Path) -> Mapping[str, Any]:
    """Load metrics JSON into memory."""

    return json.loads(path.read_text())
