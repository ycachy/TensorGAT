"""Create class-disjoint splits with deterministic within-split negatives.

Positive pairs define semantic classes.  Different classes are non-equivalent
by the benchmark definition, so a cyclic derangement of the second-language
fragments yields auditable negatives without crossing a split boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


class UnionFind:
    def __init__(self): self.parent = {}
    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value: self.parent[value] = self.find(self.parent[value])
        return self.parent[value]
    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right: self.parent[right] = left


def digest(seed, value): return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    positives_by_pair = defaultdict(list); uf = UnionFind()
    for path in sorted(args.source.glob(f"{args.corpus}_*_1.jsonl")):
        pair = path.stem[len(args.corpus)+1:-2]
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); positives_by_pair[pair].append(row); uf.union(row["a"], row["b"])
    roots = sorted({uf.find(value) for value in uf.parent}, key=lambda value: digest(args.seed, value))
    train_end, val_end = round(0.6*len(roots)), round(0.8*len(roots))
    assignment = {root: ("train" if i < train_end else "val" if i < val_end else "test")
                  for i, root in enumerate(roots)}
    args.output.mkdir(parents=True, exist_ok=True); summary = {"corpus": args.corpus, "seed": args.seed,
        "negative_rule": "cyclic derangement of language-B fragments from different positive classes",
        "classes": len(roots), "pairs": {}}
    for pair, positives in sorted(positives_by_pair.items()):
        summary["pairs"][pair] = {}
        for split in ("train", "val", "test"):
            selected = [dict(row, split=split, class_id=uf.find(row["a"])) for row in positives
                        if assignment[uf.find(row["a"])] == split]
            selected.sort(key=lambda row: digest(args.seed+1, row["id"]))
            if len(selected) < 2: raise ValueError(f"Need at least two classes in {pair}/{split}")
            negatives = []
            for index, positive in enumerate(selected):
                donor = selected[(index + 1) % len(selected)]
                if positive["class_id"] == donor["class_id"]: raise AssertionError("derangement failed")
                negatives.append({"id": f"{args.corpus}/{pair}/derived-negative/{split}/{index+1}",
                    "a": positive["a"], "b": donor["b"], "label": 0, "split": split,
                    "class_a": positive["class_id"], "class_b": donor["class_id"],
                    "negative_source": "within_split_class_derangement",
                    "derived_from": [positive["id"], donor["id"]]})
            final = sorted(selected + negatives, key=lambda row: digest(args.seed+2, row["id"]))
            target = args.output / f"{args.corpus}_{pair}_{split}.jsonl"
            target.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in final), encoding="utf-8")
            summary["pairs"][pair][split] = {"positive": len(selected), "negative": len(negatives),
                "total": len(final), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (args.output/f"{args.corpus}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
