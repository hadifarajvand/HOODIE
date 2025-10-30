from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


@dataclass
class AggregateMetrics:
    """Structured view of aggregate evaluation statistics."""

    mean_latency: float
    load: float
    p95_latency: float
    drop_rate: float
    throughput: float
    offload_ratio_cloud: float
    offload_ratio_edge: float
    avg_queue_lengths: Dict[str, float]
    peak_queue_lengths: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_latency": self.mean_latency,
            "load": self.load,
            "p95_latency": self.p95_latency,
            "drop_rate": self.drop_rate,
            "throughput": self.throughput,
            "offload_ratio_cloud": self.offload_ratio_cloud,
            "offload_ratio_edge": self.offload_ratio_edge,
            "avg_queue_lengths": self.avg_queue_lengths,
            "peak_queue_lengths": self.peak_queue_lengths,
        }


def _collect(events: Iterable[Mapping[str, Any]], event_type: str) -> Sequence[Mapping[str, Any]]:
    return [event for event in events if event.get("event") == event_type]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def _queue_stats(events: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    running: MutableMapping[str, list] = {}
    for event in events:
        if event.get("event") != "queue_length":
            continue
        queue_name = str(event.get("queue"))
        length = float(event.get("length", 0.0))
        running.setdefault(queue_name, []).append(length)
    avg = {queue: float(np.mean(lengths)) for queue, lengths in running.items()}
    peak = {queue: float(np.max(lengths)) for queue, lengths in running.items()}
    return avg, peak


def compute_metrics(events: Sequence[Mapping[str, Any]], episode_time: int) -> AggregateMetrics:
    """
    Compute aggregate metrics from raw event logs.

    Events are expected to contain, at minimum, the following schema:

    - ``task_arrived``: records task metadata at creation time.
    - ``task_completed``: includes ``latency`` and ``destination`` (``\"cloud\"`` or ``\"edge\"``).
    - ``task_dropped``: task timed out; destination mirrors intended executor.
    - ``queue_length``: instantaneous queue length sample (optional but used for stats).
    """

    arrivals = _collect(events, "task_arrived")
    completions = _collect(events, "task_completed")
    drops = _collect(events, "task_dropped")

    latencies = [float(event.get("latency", 0.0)) for event in completions if "latency" in event]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = _percentile(latencies, 95.0)

    total_arrivals = max(len(arrivals), 1)
    unique_servers = {event.get("server") for event in arrivals if event.get("server") is not None}
    denominator = max(len(unique_servers) * episode_time, 1)
    load = len(arrivals) / denominator
    drop_rate = len(drops) / total_arrivals
    throughput = len(completions) / max(episode_time, 1)

    cloud_count = sum(1 for event in completions if event.get("destination") == "cloud")
    edge_count = sum(1 for event in completions if event.get("destination") == "edge")
    total_completions = max(cloud_count + edge_count, 1)
    offload_ratio_cloud = cloud_count / total_completions
    offload_ratio_edge = edge_count / total_completions

    avg_queue_lengths, peak_queue_lengths = _queue_stats(events)

    return AggregateMetrics(
        mean_latency=mean_latency,
        load=load,
        p95_latency=p95_latency,
        drop_rate=drop_rate,
        throughput=throughput,
        offload_ratio_cloud=offload_ratio_cloud,
        offload_ratio_edge=offload_ratio_edge,
        avg_queue_lengths=avg_queue_lengths,
        peak_queue_lengths=peak_queue_lengths,
    )
