"""Aggregate repeated runs into machine-readable and LaTeX summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("precision", "recall", "f1", "roc_auc", "mcc")


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "values": values.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for variant in ("full", "no_cp", "no_svd", "no_cp_no_svd", "gcn"):
        files = sorted(args.runs.glob(f"{variant}_seed*.json"))
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        if not runs: continue
        result[variant] = {"seeds": [run["config"]["seed"] for run in runs],
            "internal": {metric: stats([run["internal_test"][metric] for run in runs]) for metric in METRICS},
            "external": {metric: stats([run["external_test"][metric] for run in runs]) for metric in METRICS},
            "wall_seconds": stats([run["wall_seconds"] for run in runs]),
            "parameters": runs[0]["parameter_count"], "epochs": [run["epochs_run"] for run in runs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
