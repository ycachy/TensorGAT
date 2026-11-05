"""Build leakage-controlled Juliet n1/n2/n3 experiments.

The split unit is a CWE/operation family, not an individual pair.  Each
difficulty is written as an independent balanced binary experiment.  The
provided Juliet negative is retained when it belongs to the anchor's split;
otherwise a deterministic same-difficulty replacement is selected from that
split.  This prevents a source family from occurring in more than one split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.data import GraphCorpusIndex


VARIANT_RE = re.compile(r"_(?:\d+[a-z]?)$")


def family(ref: str) -> str:
    stem = Path(ref).stem
    cwe = stem.split("_", 1)[0]
    operation = VARIANT_RE.sub("", stem.split("__", 1)[1])
    return f"{cwe}:{operation}"


def cwe(ref: str) -> str:
    return family(ref).split(":", 1)[0]


def digest(seed: int, *values: str) -> str:
    return hashlib.sha256((str(seed) + ":" + ":".join(values)).encode()).hexdigest()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            left, right = line.strip().split(",")
            rows.append((left.replace("\\", "/"), right.replace("\\", "/")))
    return rows


def matches(level: str, anchor: str, candidate: str) -> bool:
    if candidate == anchor:
        return False
    if level == "n1":
        return family(anchor) == family(candidate)
    if level == "n2":
        return cwe(anchor) == cwe(candidate) and family(anchor) != family(candidate)
    if level == "n3":
        return cwe(anchor) != cwe(candidate)
    raise ValueError(level)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--corpus", default="Juliet")
    parser.add_argument("--pair", default="java-c")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--max-nodes", type=int, default=600)
    args = parser.parse_args()

    corpus = args.datasets / args.corpus
    pair_dir = corpus / args.pair
    positives = read_pairs(pair_dir / "1.txt")
    originals = {level: dict(read_pairs(pair_dir / f"{level}.txt")) for level in ("n1", "n2", "n3")}
    graphs = GraphCorpusIndex(corpus)

    positive_families = sorted(
        {family(a) for a, _ in positives}, key=lambda value: digest(args.seed, value)
    )
    train_end = round(0.6 * len(positive_families))
    val_end = round(0.8 * len(positive_families))
    assignment = {
        value: ("train" if index < train_end else "val" if index < val_end else "test")
        for index, value in enumerate(positive_families)
    }

    def split_for(ref: str) -> str:
        group = family(ref)
        if group in assignment:
            return assignment[group]
        score = int(digest(args.seed, group)[:16], 16) / 16**16
        return "train" if score < 0.6 else "val" if score < 0.8 else "test"

    candidate_pool = [
        ref for ref in sorted(graphs._records)
        if ref.startswith("c/") and graphs[ref].num_nodes <= args.max_nodes
    ]
    candidates_by_split = defaultdict(list)
    for ref in candidate_pool:
        candidates_by_split[split_for(ref)].append(ref)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "corpus": args.corpus,
        "pair": args.pair,
        "seed": args.seed,
        "max_nodes": args.max_nodes,
        "split_unit": "CWE:operation family",
        "positive_family_counts": dict(Counter(assignment.values())),
        "levels": {},
    }

    all_refs_by_level = {}
    for level in ("n1", "n2", "n3"):
        rows_by_split = defaultdict(list)
        source_counts = Counter()
        for anchor, positive in positives:
            split = split_for(anchor)
            if graphs[anchor].num_nodes > args.max_nodes or graphs[positive].num_nodes > args.max_nodes:
                source_counts["positive_over_limit"] += 1
                continue
            original = originals[level].get(anchor)
            if (original is not None and original in graphs
                    and graphs[original].num_nodes <= args.max_nodes
                    and split_for(original) == split and matches(level, positive, original)):
                negative = original
                source_counts["provided_negative"] += 1
            else:
                eligible = [
                    ref for ref in candidates_by_split[split]
                    if matches(level, positive, ref)
                ]
                if not eligible:
                    source_counts["no_within_split_negative"] += 1
                    continue
                negative = min(eligible, key=lambda ref: digest(args.seed, level, anchor, ref))
                source_counts["deterministic_replacement"] += 1

            base = {
                "split": split,
                "difficulty": level,
                "family": family(anchor),
                "cwe": cwe(anchor),
                "provided_negative": original,
            }
            rows_by_split[split].append(base | {
                "id": f"Juliet/{args.pair}/{level}/{Path(anchor).stem}/positive",
                "a": anchor, "b": positive, "label": 1,
            })
            rows_by_split[split].append(base | {
                "id": f"Juliet/{args.pair}/{level}/{Path(anchor).stem}/negative",
                "a": anchor, "b": negative, "label": 0,
                "negative_source": "provided" if negative == original else "within_split_replacement",
            })

        level_summary = {"negative_sources": dict(source_counts), "splits": {}}
        refs_by_split = {}
        for split in ("train", "val", "test"):
            rows = sorted(rows_by_split[split], key=lambda row: digest(args.seed + 1, row["id"]))
            target = args.output / f"Juliet_{args.pair}_{level}_{split}.jsonl"
            target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            refs = {row[key] for row in rows for key in ("a", "b")}
            refs_by_split[split] = refs
            level_summary["splits"][split] = {
                "positive": sum(int(row["label"]) for row in rows),
                "negative": sum(1 - int(row["label"]) for row in rows),
                "total": len(rows),
                "unique_sources": len(refs),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        overlaps = {
            "train_val": len(refs_by_split["train"] & refs_by_split["val"]),
            "train_test": len(refs_by_split["train"] & refs_by_split["test"]),
            "val_test": len(refs_by_split["val"] & refs_by_split["test"]),
        }
        if any(overlaps.values()):
            raise AssertionError(f"source leakage in {level}: {overlaps}")
        level_summary["source_overlap"] = overlaps
        summary["levels"][level] = level_summary
        all_refs_by_level[level] = refs_by_split

    summary_path = args.output / "Juliet_three_difficulties_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
