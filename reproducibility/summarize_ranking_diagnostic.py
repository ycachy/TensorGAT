"""Aggregate the frozen Reviewer 1.2 ranking diagnostic across seeds."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


VARIANTS = ("full", "no_cp", "no_svd", "no_cp_no_svd", "gcn")
LABELS = {
    "full": "TensorGAT",
    "no_cp": "without CP",
    "no_svd": "without t-SVD",
    "no_cp_no_svd": "without both",
    "gcn": "TensorGCN",
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seeds = (1234, 2025, 3407)
    raw = {
        variant: [
            json.loads((args.results / f"{variant}_seed{seed}.json").read_text(encoding="utf-8"))
            for seed in seeds
        ]
        for variant in VARIANTS
    }
    reference = raw["full"][0]
    groups = reference["groups"]
    lexical_correct = sum(
        group["positive"]["lexical_jaccard"] > group["negative"]["lexical_jaccard"]
        for group in groups
    )
    lexical_ties = sum(
        group["positive"]["lexical_jaccard"] == group["negative"]["lexical_jaccard"]
        for group in groups
    )

    rows = []
    for variant in VARIANTS:
        runs = raw[variant]
        row = {"variant": variant, "label": LABELS[variant]}
        metrics = {
            "accuracy": [run["classification"]["accuracy"] for run in runs],
            "f1": [run["classification"]["f1"] for run in runs],
            "roc_auc": [run["classification"]["roc_auc"] for run in runs],
            "mcc": [run["classification"]["mcc"] for run in runs],
            "ranking_accuracy": [run["ranking_accuracy"] for run in runs],
            "conflict_ranking_accuracy": [
                run["lexical"]["conflict_group_ranking_accuracy"] for run in runs
            ],
            "low_lexical_positive_accuracy": [
                run["lexical"]["low_lexical_positive_accuracy"] for run in runs
            ],
            "high_lexical_negative_fpr": [
                run["lexical"]["high_lexical_negative_false_positive_rate"] for run in runs
            ],
        }
        for metric, values in metrics.items():
            mean, sd = mean_sd(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_by_seed"] = values
        rows.append(row)

    majority = []
    for group in groups:
        group_id = group["group"]
        record = {
            "group": group_id,
            "query": group["query"],
            "positive": group["positive"]["b"],
            "negative": group["negative"]["b"],
            "positive_lexical_jaccard": group["positive"]["lexical_jaccard"],
            "negative_lexical_jaccard": group["negative"]["lexical_jaccard"],
            "lexical_conflict": group["lexical_conflict"],
        }
        for variant in VARIANTS:
            outcomes = [
                next(item for item in run["groups"] if item["group"] == group_id)["ranking_correct"]
                for run in raw[variant]
            ]
            record[f"{variant}_correct_seeds"] = sum(outcomes)
            record[f"{variant}_majority_correct"] = sum(outcomes) >= 2
        majority.append(record)

    summary = {
        "design": {
            "pairs": reference["n_pairs"],
            "groups": reference["n_groups"],
            "seeds": list(seeds),
            "lexical_metric": reference["lexical"]["tokenization"],
            "lexical_ranking_correct": lexical_correct,
            "lexical_ranking_ties": lexical_ties,
            "lexical_ranking_accuracy": lexical_correct / reference["n_groups"],
            "lexical_conflict_groups": reference["lexical"]["conflict_group_n"],
            "low_lexical_positive_n": reference["lexical"]["low_lexical_positive_n"],
            "high_lexical_negative_n": reference["lexical"]["high_lexical_negative_n"],
        },
        "variants": rows,
        "groups": majority,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    flat_columns = [
        "variant", "label", "accuracy_mean", "accuracy_sd", "f1_mean", "f1_sd",
        "roc_auc_mean", "roc_auc_sd", "mcc_mean", "mcc_sd",
        "ranking_accuracy_mean", "ranking_accuracy_sd",
        "conflict_ranking_accuracy_mean", "conflict_ranking_accuracy_sd",
        "low_lexical_positive_accuracy_mean", "low_lexical_positive_accuracy_sd",
        "high_lexical_negative_fpr_mean", "high_lexical_negative_fpr_sd",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Reviewer 1 Comment 2 frozen ranking diagnostic",
        "",
        f"The diagnostic contains {reference['n_groups']} queries and {reference['n_pairs']} pairs. "
        "Each query has one archived corresponding candidate and one different-operation candidate. "
        "The list was frozen before new model inference; no diagnostic example was used for training, "
        "checkpoint selection, or threshold selection.",
        "",
        "The earlier discussion described a mixed AWS/Paho candidate pack, but that file was not "
        "available locally. This diagnostic is a disclosed reconstruction from the archived AWS "
        "Python--Java pairs: the first 21 positive queries with an available same-query negative, "
        "in archived order. It is not independently double-annotated and is not called a gold set. "
        "A 900-node ceiling retains all frozen candidates; this exceeds the 600-node ceiling used "
        "in the main experiment and is treated as a diagnostic extrapolation.",
        "",
        "| Model | Pair F1 | ROC-AUC | Group ranking | Lexical-conflict ranking |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['f1_mean']:.3f}±{row['f1_sd']:.3f} | "
            f"{row['roc_auc_mean']:.3f}±{row['roc_auc_sd']:.3f} | "
            f"{row['ranking_accuracy_mean']:.3f}±{row['ranking_accuracy_sd']:.3f} | "
            f"{row['conflict_ranking_accuracy_mean']:.3f}±{row['conflict_ranking_accuracy_sd']:.3f} |"
        )
    lines.extend([
        "",
        f"The transparent lexical baseline ranks {lexical_correct}/{reference['n_groups']} groups "
        f"correctly ({lexical_correct / reference['n_groups']:.3f}); {lexical_ties} groups are tied. "
        "The three lexical-conflict groups are the cases in which the different-operation candidate "
        "has higher identifier/number token-set Jaccard than the corresponding candidate.",
        "",
        "The full TensorGAT model is near chance on group ranking, and no ablation causes a stable "
        "decrease. These observations do not demonstrate semantic understanding beyond surface "
        "similarity. They delimit this model to an evaluated pair-matching representation whose "
        "structural inputs and external generalization remain empirically weak in this setting.",
        "",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["design"], indent=2))
    for row in rows:
        print(row["label"], row["ranking_accuracy_mean"], row["ranking_accuracy_sd"])


if __name__ == "__main__":
    main()
