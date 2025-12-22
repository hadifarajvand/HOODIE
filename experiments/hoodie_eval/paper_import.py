from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .io import ArtifactLayout, write_events, write_metrics


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open() as fp:
        return json.load(fp)


def import_existing_metrics(
    source: Path,
    layout: ArtifactLayout,
    *,
    scenario_name: str = "paper",
) -> None:
    """
    Import previously published metrics/events into the standard artifact layout.

    The expected directory layout is::

        source/
            metrics.json
            events.jsonl

    ``metrics.json`` must be a JSON object of aggregate metrics, while
    ``events.jsonl`` contains one JSON object per line describing per-task events.
    """

    scenario_root = layout.scenario_root / scenario_name
    scenario_root.mkdir(parents=True, exist_ok=True)
    metrics_dest = scenario_root / "metrics.json"
    events_dest = scenario_root / "events.parquet"

    metrics_payload = _load_json(source / "metrics.json")
    write_metrics(metrics_payload, metrics_dest)

    events_source = source / "events.jsonl"
    if events_source.exists():
        records = []
        for line in events_source.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        write_events(records, events_dest)
    else:
        write_events([], events_dest)
