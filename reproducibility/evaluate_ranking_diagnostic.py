"""Evaluate frozen query-positive-negative groups without fitting or tuning.

The diagnostic reports pair classification under the validation-selected
training threshold and, more importantly, whether the positive candidate is
ranked above the negative candidate for each query.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.data import read_manifest
from reproducibility.model import TensorPairClassifier
from reproducibility.train import Artifacts, chunks, seed_everything


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")


def lexical_tokens(path: Path) -> set[str]:
    return set(TOKEN_RE.findall(path.read_text(encoding="utf-8", errors="ignore").lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@torch.inference_mode()
def predict(model, rows: list[dict], artifacts: Artifacts, batch_size: int) -> list[float]:
    model.eval()
    probabilities: list[float] = []
    for batch in chunks(rows, batch_size):
        logits = model(artifacts.batch(batch))
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return probabilities


def classification(labels: list[int], probabilities: list[float], threshold: float) -> dict:
    labels_array = np.asarray(labels)
    probability_array = np.asarray(probabilities)
    predictions = (probability_array >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels_array, predictions),
        "precision": precision_score(labels_array, predictions, zero_division=0),
        "recall": recall_score(labels_array, predictions, zero_division=0),
        "f1": f1_score(labels_array, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels_array, predictions),
        "roc_auc": roc_auc_score(labels_array, probability_array),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("full", "no_cp", "no_svd", "no_cp_no_svd", "gcn"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_manifest(args.manifest)
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    threshold = float(training["selected_threshold"])

    model = TensorPairClassifier(
        input_dim=768,
        dim=64,
        baseline="gcn" if args.variant == "gcn" else "gat",
        use_unsup=True,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    artifacts = Artifacts(args.features, args.prepared, device)
    probabilities = predict(model, rows, artifacts, args.batch_size)

    token_cache: dict[str, set[str]] = {}
    for row, probability in zip(rows, probabilities):
        for ref in (row["a"], row["b"]):
            if ref not in token_cache:
                token_cache[ref] = lexical_tokens(args.datasets / args.corpus / ref)
        row["lexical_jaccard"] = jaccard(token_cache[row["a"]], token_cache[row["b"]])
        row["probability"] = probability
        row["prediction"] = int(probability >= threshold)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["group"])].append(row)
    groups = []
    for group_id in sorted(grouped):
        group_rows = grouped[group_id]
        positive = next(row for row in group_rows if int(row["label"]) == 1)
        negative = next(row for row in group_rows if int(row["label"]) == 0)
        groups.append({
            "group": group_id,
            "query": positive["a"],
            "positive": positive,
            "negative": negative,
            "margin": positive["probability"] - negative["probability"],
            "ranking_correct": positive["probability"] > negative["probability"],
            "lexical_conflict": negative["lexical_jaccard"] > positive["lexical_jaccard"],
        })

    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    positive_median = float(np.median([row["lexical_jaccard"] for row in positives]))
    negative_median = float(np.median([row["lexical_jaccard"] for row in negatives]))
    low_positive = [row for row in positives if row["lexical_jaccard"] <= positive_median]
    high_negative = [row for row in negatives if row["lexical_jaccard"] >= negative_median]
    conflict_groups = [group for group in groups if group["lexical_conflict"]]

    result = {
        "config": vars(args) | {"device": str(device), "selected_threshold": threshold},
        "n_pairs": len(rows),
        "n_groups": len(groups),
        "classification": classification(
            [int(row["label"]) for row in rows], probabilities, threshold
        ),
        "ranking_accuracy": sum(group["ranking_correct"] for group in groups) / len(groups),
        "ranking_correct": sum(group["ranking_correct"] for group in groups),
        "mean_margin": float(np.mean([group["margin"] for group in groups])),
        "lexical": {
            "tokenization": "lowercase identifier-and-number token-set Jaccard",
            "positive_median": positive_median,
            "negative_median": negative_median,
            "low_lexical_positive_n": len(low_positive),
            "low_lexical_positive_accuracy": sum(row["prediction"] == 1 for row in low_positive) / len(low_positive),
            "high_lexical_negative_n": len(high_negative),
            "high_lexical_negative_false_positive_rate": sum(row["prediction"] == 1 for row in high_negative) / len(high_negative),
            "conflict_group_n": len(conflict_groups),
            "conflict_group_ranking_accuracy": (
                sum(group["ranking_correct"] for group in conflict_groups) / len(conflict_groups)
                if conflict_groups else None
            ),
        },
        "groups": groups,
    }
    result["config"] = {key: str(value) if isinstance(value, Path) else value
                        for key, value in result["config"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "n_pairs", "n_groups", "classification", "ranking_accuracy",
        "ranking_correct", "mean_margin", "lexical")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
