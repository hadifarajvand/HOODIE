#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH=${1:-configs/hoodie_eval/paper_full.json}
PYTHON_BIN=${PYTHON:-python}

if ! ${PYTHON_BIN} -c "import typer" >/dev/null 2>&1; then
  echo "[hoodie-eval] Typer is required. Install it with 'pip install typer[all]'." >&2
  exit 1
fi

readarray -t META < <(
  HOODIE_CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from experiments.hoodie_eval.config import load_config

config_path = os.environ["HOODIE_CONFIG_PATH"]
config = load_config(config_path)
scenario = config.scenario.name
systems = config.all_systems
print(scenario)
print(" ".join(systems))
print(config.output.artifacts_root.as_posix())
PY
)

if [[ ${#META[@]} -lt 3 ]]; then
  echo "[hoodie-eval] Failed to parse configuration metadata." >&2
  exit 1
fi

SCENARIO_NAME=${META[0]}
SYSTEMS_LINE=${META[1]}
ARTIFACT_ROOT=${META[2]}

echo "[hoodie-eval] Running scenario '${SCENARIO_NAME}' using ${CONFIG_PATH}"
${PYTHON_BIN} -m experiments.hoodie_eval.cli run --config "${CONFIG_PATH}"

SCENARIO_ARTIFACT=$(
  HOODIE_ARTIFACT_ROOT="${ARTIFACT_ROOT}" HOODIE_SCENARIO="${SCENARIO_NAME}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["HOODIE_ARTIFACT_ROOT"])
scenario = os.environ["HOODIE_SCENARIO"]
stamp_dirs = sorted(
    (p for p in root.iterdir() if p.is_dir()),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not stamp_dirs:
    raise SystemExit("No artifact directories found under " + str(root))
scenario_root = stamp_dirs[0] / scenario
if not scenario_root.exists():
    raise SystemExit(f"Scenario directory {scenario_root} not found.")
print(scenario_root.as_posix())
PY
)

echo "[hoodie-eval] Artifacts stored in ${SCENARIO_ARTIFACT}"

if [[ -n "${SYSTEMS_LINE}" ]]; then
  read -r -a SYSTEM_ARRAY <<< "${SYSTEMS_LINE}"
else
  SYSTEM_ARRAY=()
fi

COMPARE_ARGS=()
for system in "${SYSTEM_ARRAY[@]}"; do
  COMPARE_ARGS+=(--systems "${system}")
done

if [[ ${#SYSTEM_ARRAY[@]} -gt 0 ]]; then
  echo "[hoodie-eval] Summary metrics:"
  ${PYTHON_BIN} -m experiments.hoodie_eval.cli compare \
    --artifacts "${SCENARIO_ARTIFACT}" \
    "${COMPARE_ARGS[@]}"
fi

echo "[hoodie-eval] Generating plots."
${PYTHON_BIN} -m experiments.hoodie_eval.cli plots \
  --artifacts "${SCENARIO_ARTIFACT}"

echo "[hoodie-eval] Compiling metrics summary CSV."
HOODIE_SCENARIO_ROOT="${SCENARIO_ARTIFACT}" ${PYTHON_BIN} - <<'PY'
from pathlib import Path
import csv
import json
import os

scenario_root = Path(os.environ["HOODIE_SCENARIO_ROOT"])
rows = []
for system_dir in sorted(p for p in scenario_root.iterdir() if p.is_dir() and p.name != "plots"):
    for seed_dir in system_dir.glob("seed_*"):
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as fp:
            data = json.load(fp)
        row = {"system": system_dir.name, "seed": seed_dir.name.replace("seed_", "")}
        row.update({k: v for k, v in data.items() if isinstance(v, (int, float))})
        rows.append(row)

if rows:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    summary_path = scenario_root / "metrics_summary.csv"
    with summary_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[hoodie-eval] Metrics summary saved to {summary_path}")
else:
    print("[hoodie-eval] No metrics found to summarise.")
PY

echo "[hoodie-eval] Workflow complete."
