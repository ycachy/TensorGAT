"""Train and evaluate corrected TensorGAT/TensorGCN variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.data import GraphCorpusIndex, filter_pairs_by_size, read_manifest
from reproducibility.model import PreparedGraph, TensorPairClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_balance(rows: list[dict], seed: int) -> list[dict]:
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    key = lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()
    count = min(len(positives), len(negatives))
    return sorted(sorted(positives, key=key)[:count] + sorted(negatives, key=key)[:count], key=key)


class Artifacts:
    def __init__(self, features: Path, prepared: Path, device: torch.device) -> None:
        self.feature_index = json.loads(features.read_text(encoding="utf-8"))["graphs"]
        self.graph_index = json.loads(prepared.read_text(encoding="utf-8"))["graphs"]
        self.device = device
        self.features, self.graphs = {}, {}

    def feature(self, ref: str) -> torch.Tensor:
        if ref not in self.features:
            saved = torch.load(self.feature_index[ref]["path"], map_location="cpu", weights_only=True)
            self.features[ref] = saved["features"].float()
        return self.features[ref].to(self.device)

    def graph(self, ref: str) -> PreparedGraph:
        if ref not in self.graphs:
            value = torch.load(self.graph_index[ref], map_location="cpu", weights_only=True)
            self.graphs[ref] = PreparedGraph.from_dict(value)
        return self.graphs[ref]

    def batch(self, rows: list[dict]) -> list[dict]:
        return [{"a": row["a"], "b": row["b"],
                 "features_a": self.feature(row["a"]), "features_b": self.feature(row["b"]),
                 "graph_a": self.graph(row["a"]), "graph_b": self.graph(row["b"])} for row in rows]


def chunks(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


@torch.inference_mode()
def evaluate(model, rows, artifacts, batch_size, threshold=0.5):
    model.eval(); labels, logits = [], []
    for rows_batch in chunks(rows, batch_size):
        logits.extend(model(artifacts.batch(rows_batch)).cpu().tolist())
        labels.extend(int(row["label"]) for row in rows_batch)
    probability = 1 / (1 + np.exp(-np.asarray(logits)))
    prediction = (probability >= threshold).astype(int)
    result = {"n": len(labels), "positive": int(sum(labels)), "threshold": float(threshold),
              "accuracy": accuracy_score(labels, prediction),
              "precision": precision_score(labels, prediction, zero_division=0),
              "recall": recall_score(labels, prediction, zero_division=0),
              "f1": f1_score(labels, prediction, zero_division=0),
              "mcc": matthews_corrcoef(labels, prediction)}
    result["roc_auc"] = roc_auc_score(labels, probability) if len(set(labels)) == 2 else None
    result["logits"] = logits
    result["labels"] = labels
    return result


def select_threshold(metrics: dict) -> float:
    labels = np.asarray(metrics["labels"])
    probabilities = 1 / (1 + np.exp(-np.asarray(metrics["logits"])))
    candidates = np.unique(np.r_[0.5, probabilities])
    scored = []
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(int)
        scored.append((f1_score(labels, prediction, zero_division=0),
                       accuracy_score(labels, prediction), -abs(float(threshold) - 0.5), float(threshold)))
    return max(scored)[-1]


def load_split(path, graphs, max_nodes, seed):
    rows, excluded = filter_pairs_by_size(read_manifest(path), graphs, max_nodes)
    rows = stable_balance(rows, seed)
    return rows, len(excluded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--variant", choices=("full", "no_cp", "no_svd", "no_cp_no_svd", "gcn"), default="full")
    parser.add_argument("--external-corpus")
    parser.add_argument("--external-test", type=Path)
    parser.add_argument("--external-features", type=Path)
    parser.add_argument("--external-prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--unsup-weight", type=float, default=1e-3)
    parser.add_argument("--max-nodes", type=int, default=600)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_graphs = GraphCorpusIndex(args.datasets / args.corpus)
    train, train_excluded = load_split(args.train, corpus_graphs, args.max_nodes, args.seed)
    val, val_excluded = load_split(args.val, corpus_graphs, args.max_nodes, args.seed)
    test, test_excluded = load_split(args.test, corpus_graphs, args.max_nodes, args.seed)
    artifacts = Artifacts(args.features, args.prepared, device)
    baseline = "gcn" if args.variant == "gcn" else "gat"
    model = TensorPairClassifier(baseline=baseline, use_unsup=args.unsup_weight > 0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{args.variant}_seed{args.seed}.pt"
    history, best_score, stale = [], (-1.0, float("-inf")), 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); shuffled = train.copy(); random.Random(args.seed + epoch).shuffle(shuffled)
        losses = []
        for rows_batch in chunks(shuffled, args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = artifacts.batch(rows_batch)
            labels = torch.tensor([row["label"] for row in rows_batch], dtype=torch.float32, device=device)
            if args.unsup_weight > 0:
                logits, unsup = model(batch, return_unsup=True)
                loss = F.binary_cross_entropy_with_logits(logits, labels) + args.unsup_weight * unsup
            else:
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        validation = evaluate(model, val, artifacts, args.batch_size)
        val_logits = torch.tensor(validation["logits"])
        val_labels = torch.tensor(validation["labels"], dtype=torch.float32)
        validation["bce"] = float(F.binary_cross_entropy_with_logits(val_logits, val_labels))
        scheduler.step(validation["roc_auc"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "lr": optimizer.param_groups[0]["lr"], "validation": validation})
        print(f"epoch={epoch} loss={np.mean(losses):.5f} val_f1={validation['f1']:.4f}", flush=True)
        score = (float(validation["roc_auc"]), -validation["bce"])
        if score > best_score:
            best_score, stale = score, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    best_validation = evaluate(model, val, artifacts, args.batch_size)
    threshold = select_threshold(best_validation)
    internal_fixed = evaluate(model, test, artifacts, args.batch_size)
    internal = evaluate(model, test, artifacts, args.batch_size, threshold=threshold)
    result = {
        "config": vars(args) | {"datasets": str(args.datasets), "train": str(args.train),
                                "val": str(args.val), "test": str(args.test),
                                "features": str(args.features), "prepared": str(args.prepared),
                                "output": str(args.output)},
        "device": str(device), "torch": torch.__version__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "excluded_over_limit": {"train": train_excluded, "val": val_excluded, "test": test_excluded},
        "best_val_roc_auc": best_score[0], "selected_threshold": threshold,
        "best_validation": best_validation, "epochs_run": len(history),
        "wall_seconds": time.perf_counter() - start_time,
        "internal_test": internal, "internal_test_at_0_5": internal_fixed, "history": history,
    }
    if args.external_test:
        external_graphs = GraphCorpusIndex(args.datasets / args.external_corpus)
        external, excluded = load_split(args.external_test, external_graphs, args.max_nodes, args.seed)
        external_artifacts = Artifacts(args.external_features, args.external_prepared, device)
        result["external_test_at_0_5"] = evaluate(model, external, external_artifacts, args.batch_size)
        result["external_test"] = evaluate(model, external, external_artifacts, args.batch_size, threshold=threshold)
        result["external_test"]["excluded_over_limit"] = excluded
    target = args.output / f"{args.variant}_seed{args.seed}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"result": str(target.resolve()), "internal": internal,
                      "external": result.get("external_test")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
