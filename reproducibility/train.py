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
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
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


def dynamic_pair_batches(rows: list[dict], size: int, seed: int, epoch: int):
    """Yield anchor-grouped pairs with a difficulty-matched negative each epoch.

    Candidate C fragments come only from positive pairs already assigned to the
    training split.  n1 samples within one operation family, n2 within one CWE
    but across operation families, and n3 across CWEs.  This preserves the
    family-disjoint train/validation/test boundary.
    """
    if size < 2 or size % 2:
        raise ValueError("dynamic pair training requires an even batch size >= 2")
    positives = [row for row in rows if int(row["label"]) == 1]
    candidates = sorted({(row["b"], row["cwe"], row["family"]) for row in positives})
    pairs = []
    for positive in positives:
        difficulty = positive["difficulty"]
        if difficulty == "n1":
            eligible = [ref for ref, _, candidate_family in candidates
                        if candidate_family == positive["family"] and ref != positive["b"]]
        elif difficulty == "n2":
            eligible = [ref for ref, candidate_cwe, candidate_family in candidates
                        if candidate_cwe == positive["cwe"]
                        and candidate_family != positive["family"]]
        elif difficulty == "n3":
            eligible = [ref for ref, candidate_cwe, _ in candidates
                        if candidate_cwe != positive["cwe"]]
        else:
            raise ValueError(f"unsupported dynamic-negative difficulty: {difficulty}")
        if not eligible:
            raise ValueError(f"no {difficulty} negative candidate for {positive['id']}")
        digest = hashlib.sha256(
            f"{seed}:{epoch}:{positive['id']}".encode("utf-8")
        ).digest()
        negative_ref = eligible[int.from_bytes(digest[:8], "big") % len(eligible)]
        negative = dict(positive)
        negative.update({
            "id": f"{positive['id']}/dynamic-negative/{epoch}",
            "b": negative_ref,
            "label": 0,
            "negative_source": "dynamic_within_train_cross_cwe",
        })
        pairs.append((positive, negative))
    random.Random(seed + epoch).shuffle(pairs)
    pairs_per_batch = size // 2
    for start in range(0, len(pairs), pairs_per_batch):
        yield [row for pair in pairs[start:start + pairs_per_batch] for row in pair]


def paired_manifest_batches(rows: list[dict], size: int, seed: int, epoch: int):
    """Keep each manifest positive beside its same-anchor negative.

    The Comment-3 manifests contain one positive and one independently labelled
    negative for each left-hand fragment.  Grouping them supports a paired
    ranking term without changing the model or generating new examples.
    """
    if size < 2 or size % 2:
        raise ValueError("paired ranking requires an even batch size >= 2")
    positives, negatives = {}, {}
    for row in rows:
        if int(row["label"]) == 1:
            positives[row["id"]] = row
        else:
            pair_id = row.get("paired_positive_id")
            if pair_id is None:
                pair_id = row["a"]
            if pair_id in negatives:
                raise ValueError(f"multiple negatives for pair {pair_id}")
            negatives[pair_id] = row
    # Legacy manifests used the unique left reference as the pair key.
    if set(positives) != set(negatives):
        by_anchor = {row["a"]: row for row in positives.values()}
        if len(by_anchor) == len(positives) and set(by_anchor) == set(negatives):
            positives = by_anchor
    if set(positives) != set(negatives):
        raise ValueError("positive/negative pair ids do not match")
    pairs = [(positives[key], negatives[key]) for key in positives]
    random.Random(seed + epoch).shuffle(pairs)
    pairs_per_batch = size // 2
    for start in range(0, len(pairs), pairs_per_batch):
        yield [row for pair in pairs[start:start + pairs_per_batch] for row in pair]


@torch.inference_mode()
def evaluate(model, rows, artifacts, batch_size, threshold=0.5, amp=False):
    model.eval(); labels, logits = [], []
    for rows_batch in chunks(rows, batch_size):
        device_type = artifacts.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=amp and device_type == "cuda",
        ):
            values = model(artifacts.batch(rows_batch))
        logits.extend(values.float().cpu().tolist())
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
    result["average_precision"] = (
        average_precision_score(labels, probability)
        if len(set(labels)) == 2 else None
    )
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
    parser.add_argument(
        "--variant",
        choices=("full", "feature_only", "structure_only", "gcn",
                 "no_cp", "no_svd", "no_cp_no_svd"),
        default="full",
    )
    parser.add_argument("--external-corpus")
    parser.add_argument("--external-test", type=Path)
    parser.add_argument("--external-features", type=Path)
    parser.add_argument("--external-prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        help="Optional same-architecture checkpoint used to initialize transfer learning.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-auc-improvement", type=float, default=0.0)
    parser.add_argument("--selection-metric",
                        choices=("roc_auc", "average_precision", "f1"),
                        default="roc_auc")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--unsup-weight", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.2)
    parser.add_argument(
        "--focal-gamma", type=float, default=0.0,
        help="Use focal BCE with this gamma; zero preserves ordinary BCE.",
    )
    parser.add_argument("--dynamic-negatives", "--dynamic-n3-negatives",
                        dest="dynamic_negatives", action="store_true")
    parser.add_argument("--paired-ranking", action="store_true")
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--shared-encoder", action="store_true")
    parser.add_argument("--pair-interactions", action="store_true")
    parser.add_argument("--normalize-input", action="store_true")
    parser.add_argument("--symmetric-pair-head", action="store_true")
    parser.add_argument("--normalize-pair-embeddings", action="store_true")
    parser.add_argument("--head-layer-norm", action="store_true")
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true",
                        help="Use CUDA bfloat16 autocast for training and evaluation.")
    parser.add_argument("--max-nodes", type=int, default=600)
    args = parser.parse_args()

    if args.dynamic_negatives and args.paired_ranking:
        parser.error("choose only one paired-batch strategy")
    if args.rank_weight > 0 and not (args.dynamic_negatives or args.paired_ranking):
        parser.error("--rank-weight requires paired batches")
    if args.symmetric_pair_head and not args.shared_encoder:
        parser.error("--symmetric-pair-head requires --shared-encoder")
    if not 0 <= args.head_dropout < 1:
        parser.error("--head-dropout must be in [0, 1)")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_graphs = GraphCorpusIndex(args.datasets / args.corpus)
    train, train_excluded = load_split(args.train, corpus_graphs, args.max_nodes, args.seed)
    val, val_excluded = load_split(args.val, corpus_graphs, args.max_nodes, args.seed)
    test, test_excluded = load_split(args.test, corpus_graphs, args.max_nodes, args.seed)
    artifacts = Artifacts(args.features, args.prepared, device)
    # ``structure_only`` deliberately replaces learned GAT feature attention
    # with the same GCN propagation used by the TensorGCN baseline while
    # retaining the t-SVD/CP structural branch.  It is an ablation, not a
    # renamed TensorGCN model.  ``gcn`` has neither decomposition branch.
    baseline = "gcn" if args.variant in ("gcn", "structure_only") else "gat"
    use_structural_attention = args.variant not in (
        "gcn", "feature_only", "no_cp", "no_cp_no_svd"
    )
    model = TensorPairClassifier(
        baseline=baseline,
        use_unsup=args.unsup_weight > 0,
        shared_encoder=args.shared_encoder,
        pair_interactions=args.pair_interactions,
        use_structural_attention=use_structural_attention,
        normalize_input=args.normalize_input,
        symmetric_pair_head=args.symmetric_pair_head,
        normalize_pair_embeddings=args.normalize_pair_embeddings,
        head_layer_norm=args.head_layer_norm,
        head_dropout=args.head_dropout,
    ).to(device)
    if args.initial_checkpoint:
        model.load_state_dict(torch.load(
            args.initial_checkpoint, map_location=device, weights_only=True
        ))
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{args.variant}_seed{args.seed}.pt"
    history, best_score, stale = [], (-1.0, float("-inf")), 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.dynamic_negatives:
            training_batches = dynamic_pair_batches(
                train, args.batch_size, args.seed, epoch
            )
        elif args.paired_ranking:
            training_batches = paired_manifest_batches(
                train, args.batch_size, args.seed, epoch
            )
        else:
            shuffled = train.copy(); random.Random(args.seed + epoch).shuffle(shuffled)
            training_batches = chunks(shuffled, args.batch_size)
        losses, bce_losses, rank_losses, contrastive_losses = [], [], [], []
        for rows_batch in training_batches:
            optimizer.zero_grad(set_to_none=True)
            batch = artifacts.batch(rows_batch)
            labels = torch.tensor([row["label"] for row in rows_batch], dtype=torch.float32, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.amp and device.type == "cuda",
            ):
                if args.unsup_weight > 0 or args.contrastive_weight > 0:
                    if args.contrastive_weight > 0:
                        logits, unsup, left_embedding, right_embedding = model(
                            batch, return_unsup=args.unsup_weight > 0,
                            return_embeddings=True,
                        )
                    else:
                        logits, unsup = model(batch, return_unsup=True)
                else:
                    logits = model(batch)
                    unsup = logits.new_zeros(())
                per_example_bce = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                if args.focal_gamma > 0:
                    correct_class_probability = torch.exp(-per_example_bce)
                    bce = (
                        (1.0 - correct_class_probability).pow(args.focal_gamma)
                        * per_example_bce
                    ).mean()
                else:
                    bce = per_example_bce.mean()
                if args.rank_weight > 0:
                    # dynamic_pair_batches orders every pair as positive, negative.
                    ranking = F.softplus(-(logits[0::2] - logits[1::2])).mean()
                else:
                    ranking = logits.new_zeros(())
                if args.contrastive_weight > 0:
                    targets = labels.mul(2).sub(1)
                    contrastive = F.cosine_embedding_loss(
                        left_embedding, right_embedding, targets,
                        margin=args.contrastive_margin,
                    )
                else:
                    contrastive = logits.new_zeros(())
                loss = (bce + args.rank_weight * ranking
                        + args.unsup_weight * unsup
                        + args.contrastive_weight * contrastive)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); losses.append(float(loss.detach()))
            bce_losses.append(float(bce.detach()))
            rank_losses.append(float(ranking.detach()))
            contrastive_losses.append(float(contrastive.detach()))
        validation = evaluate(model, val, artifacts, args.batch_size, amp=args.amp)
        val_logits = torch.tensor(validation["logits"])
        val_labels = torch.tensor(validation["labels"], dtype=torch.float32)
        validation["bce"] = float(F.binary_cross_entropy_with_logits(val_logits, val_labels))
        validation["selected_threshold"] = select_threshold(validation)
        validation_probability = torch.sigmoid(val_logits).numpy()
        validation_prediction = (
            validation_probability >= validation["selected_threshold"]
        ).astype(int)
        validation["f1_at_selected_threshold"] = f1_score(
            validation["labels"], validation_prediction, zero_division=0
        )
        selection_value = (
            validation["f1_at_selected_threshold"]
            if args.selection_metric == "f1"
            else validation[args.selection_metric]
        )
        scheduler.step(selection_value)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "bce": float(np.mean(bce_losses)),
                        "ranking": float(np.mean(rank_losses)),
                        "contrastive": float(np.mean(contrastive_losses)),
                        "lr": optimizer.param_groups[0]["lr"], "validation": validation})
        print(f"epoch={epoch} loss={np.mean(losses):.5f} bce={np.mean(bce_losses):.5f} "
              f"rank={np.mean(rank_losses):.5f} val_f1={validation['f1']:.4f} "
              f"contrastive={np.mean(contrastive_losses):.5f} "
              f"val_auc={validation['roc_auc']:.4f} "
              f"val_selected_f1={validation['f1_at_selected_threshold']:.4f}",
              flush=True)
        score = (float(selection_value), -validation["bce"])
        if score[0] > best_score[0] + args.min_auc_improvement:
            best_score, stale = score, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    best_validation = evaluate(model, val, artifacts, args.batch_size, amp=args.amp)
    threshold = select_threshold(best_validation)
    internal_fixed = evaluate(model, test, artifacts, args.batch_size, amp=args.amp)
    internal = evaluate(model, test, artifacts, args.batch_size,
                        threshold=threshold, amp=args.amp)
    result = {
        "config": vars(args) | {"datasets": str(args.datasets), "train": str(args.train),
                                "val": str(args.val), "test": str(args.test),
                                "features": str(args.features), "prepared": str(args.prepared),
                                "output": str(args.output)},
        "device": str(device), "torch": torch.__version__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "excluded_over_limit": {"train": train_excluded, "val": val_excluded, "test": test_excluded},
        "best_val_roc_auc": best_validation["roc_auc"],
        "best_val_selection_score": best_score[0],
        "selected_threshold": threshold,
        "best_validation": best_validation, "epochs_run": len(history),
        "wall_seconds": time.perf_counter() - start_time,
        "internal_test": internal, "internal_test_at_0_5": internal_fixed, "history": history,
    }
    if args.external_test:
        external_graphs = GraphCorpusIndex(args.datasets / args.external_corpus)
        external, excluded = load_split(args.external_test, external_graphs, args.max_nodes, args.seed)
        external_artifacts = Artifacts(args.external_features, args.external_prepared, device)
        result["external_test_at_0_5"] = evaluate(
            model, external, external_artifacts, args.batch_size, amp=args.amp
        )
        result["external_test"] = evaluate(
            model, external, external_artifacts, args.batch_size,
            threshold=threshold, amp=args.amp
        )
        result["external_test"]["excluded_over_limit"] = excluded
    target = args.output / f"{args.variant}_seed{args.seed}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    compact = lambda value: (
        {key: item for key, item in value.items()
         if key not in ("logits", "labels")}
        if value is not None else None
    )
    print(json.dumps({"result": str(target.resolve()),
                      "internal": compact(internal),
                      "external": compact(result.get("external_test"))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
