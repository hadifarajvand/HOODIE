from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

try:
    import typer
except Exception:  # pragma: no cover - Typer may not be installed
    typer = None  # type: ignore

from . import plots
from .config import HoodieEvalConfig, load_config
from .io import ArtifactLayout, load_metrics, resolve_layout
from .runner import run_experiment


def _filter_systems(config: HoodieEvalConfig, systems: Optional[Sequence[str]]) -> HoodieEvalConfig:
    if not systems:
        return config
    systems_normalised = {system.lower() for system in systems}
    agent = config.agent if config.agent and config.agent.name.lower() in systems_normalised else None
    baselines = [baseline for baseline in config.baselines if baseline.name.lower() in systems_normalised]
    return HoodieEvalConfig(
        scenario=config.scenario,
        agent=agent,
        baselines=baselines,
        output=config.output,
    )


def _render_layout(config: HoodieEvalConfig) -> ArtifactLayout:
    return resolve_layout(config.output, config.scenario)


def _command_run(config_path: Path, systems: Optional[List[str]]) -> None:
    config = load_config(config_path)
    config = _filter_systems(config, systems)
    results = run_experiment(config)
    typer.secho(f"Completed {len(results)} runs", fg=typer.colors.GREEN)


def _command_compare(artifacts_root: Path, systems: List[str]) -> None:
    for system in systems:
        metrics_path = artifacts_root / system / "metrics.json"
        if not metrics_path.exists():
            typer.secho(f"Metrics not found for {system}", fg=typer.colors.RED)
            continue
        data = load_metrics(metrics_path)
        load = data.get("load")
        load_segment = f", load={load:.3f}" if load is not None else ""
        typer.secho(
            f"{system}: mean_latency={data['mean_latency']:.3f}, drop_rate={data['drop_rate']:.3f}{load_segment}"
        )


def _command_plots(artifacts_root: Path) -> None:
    latency_points = []
    drop_points = []
    for system_dir in artifacts_root.iterdir():
        metrics_path = system_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as fp:
            data = json.load(fp)
        load_value = data.get("load", 0.0)
        latency_points.append({"label": system_dir.name, "load": load_value, "latency": data.get("mean_latency", 0.0)})
        drop_points.append({"label": system_dir.name, "load": load_value, "drop_rate": data.get("drop_rate", 0.0)})
    layout = ArtifactLayout(
        root=artifacts_root,
        scenario_root=artifacts_root,
        plots_root=artifacts_root / "plots",
        config_path=artifacts_root / "config.json",
    )
    plots.plot_latency_vs_load(latency_points, layout.plots_root / "latency_vs_load.png")
    plots.plot_drop_rate_vs_load(drop_points, layout.plots_root / "drop_rate_vs_load.png")
    typer.secho(f"Plots saved under {layout.plots_root}", fg=typer.colors.GREEN)


if typer is not None:  # pragma: no cover
    app = typer.Typer(add_completion=False)

    @app.command()
    def run(
        config: Path = typer.Option(..., "--config", help="Path to hoodie evaluation config."),
        system: List[str] = typer.Option(None, "--system", help="Systems to run (hoodie, flc, ...)."),
    ) -> None:
        """Execute the configured evaluation pipeline."""

        _command_run(config, system)

    @app.command()
    def compare(
        artifacts: Path = typer.Option(..., "--artifacts", help="Directory containing metrics.json files."),
        systems: List[str] = typer.Option(..., "--systems", help="Systems to compare."),
    ) -> None:
        """Print a concise comparison table across systems."""

        _command_compare(artifacts, systems)

    @app.command()
    def plots_command(
        artifacts: Path = typer.Option(..., "--artifacts", help="Directory containing metrics.json files."),
    ) -> None:
        """Regenerate paper-style plots from stored metrics."""

        _command_plots(artifacts)

    __all__ = ["app"]

    if __name__ == "__main__":  # pragma: no cover
        app()

else:  # pragma: no cover - fallback simple CLI

    def main() -> None:
        raise RuntimeError("Typer is required for the hoodie evaluation CLI.")

    __all__ = ["main"]

    if __name__ == "__main__":
        main()
