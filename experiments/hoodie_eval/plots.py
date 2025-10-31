from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import json
import matplotlib.pyplot as plt


def plot_convergence(log_path: Path, path: Path) -> None:
    if not log_path.exists():
        return
    episodes = []
    rewards = []
    losses = []
    with log_path.open() as fp:
        for line in fp:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            episodes.append(record.get("episode", len(episodes)))
            rewards.append(record.get("reward", 0.0))
            losses.append(record.get("loss", 0.0))
    if not episodes:
        return
    _prepare_axis()
    fig, ax1 = plt.subplots()
    ax1.plot(episodes, rewards, label="Reward", color="tab:blue")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(episodes, losses, label="Loss", color="tab:red", alpha=0.7)
    ax2.set_ylabel("Loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


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
