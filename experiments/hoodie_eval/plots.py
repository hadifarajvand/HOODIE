from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt


def _prepare_axis() -> None:
    plt.style.use("seaborn-v0_8-colorblind")  # type: ignore[attr-defined]


def plot_latency_vs_load(points: Sequence[Mapping[str, float]], path: Path) -> None:
    """
    Plot average latency against arrival load for multiple systems.

    Each point mapping must contain ``load``, ``latency``, and ``label`` entries.
    """

    _prepare_axis()
    fig, ax = plt.subplots()
    labels = sorted({point["label"] for point in points})
    for label in labels:
        xs = [point["load"] for point in points if point["label"] == label]
        ys = [point["latency"] for point in points if point["label"] == label]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Arrival probability (load)")
    ax.set_ylabel("Average latency")
    ax.set_title("Latency vs. Load")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_drop_rate_vs_load(points: Sequence[Mapping[str, float]], path: Path) -> None:
    """Plot drop rate as a function of arrival load."""

    _prepare_axis()
    fig, ax = plt.subplots()
    labels = sorted({point["label"] for point in points})
    for label in labels:
        xs = [point["load"] for point in points if point["label"] == label]
        ys = [point["drop_rate"] for point in points if point["label"] == label]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Arrival probability (load)")
    ax.set_ylabel("Drop rate")
    ax.set_title("Drop Rate vs. Load")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_scalability(points: Sequence[Mapping[str, float]], path: Path) -> None:
    """Plot latency against the number of edge agents."""

    _prepare_axis()
    fig, ax = plt.subplots()
    labels = sorted({point["label"] for point in points})
    for label in labels:
        xs = [point["num_edges"] for point in points if point["label"] == label]
        ys = [point["latency"] for point in points if point["label"] == label]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Number of edge agents (N)")
    ax.set_ylabel("Average latency")
    ax.set_title("Scalability")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(curves: Mapping[str, Iterable[float]], path: Path) -> None:
    """Plot convergence curves (reward/loss) for HOODIE agent training."""

    _prepare_axis()
    fig, ax = plt.subplots()
    for label, values in curves.items():
        ax.plot(list(values), label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("Training Convergence")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_ablation(points: Sequence[Mapping[str, float]], path: Path) -> None:
    """Bar plot for ablation studies toggling Double/Dueling/LSTM."""

    _prepare_axis()
    fig, ax = plt.subplots()
    labels = [point["label"] for point in points]
    latencies = [point["latency"] for point in points]
    ax.bar(labels, latencies)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Average latency")
    ax.set_title("Ablation Study")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
