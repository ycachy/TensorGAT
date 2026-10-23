"""Precompute frozen CodeBERT node features for one or more pair manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codebert_features import CodeBERTFeatureCache, CodeBERTNodeEncoder
from reproducibility.data import GraphCorpusIndex, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/codebert-base")
    parser.add_argument("--max-nodes", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    graphs = GraphCorpusIndex(args.datasets / args.corpus)
    refs = set()
    for manifest in args.manifest:
        for row in read_manifest(manifest):
            refs.update((row["a"], row["b"]))
    refs = sorted(ref for ref in refs if graphs[ref].num_nodes <= args.max_nodes)

    encoder = CodeBERTNodeEncoder(model_name=args.model)
    cache = CodeBERTFeatureCache(args.cache, encoder)
    index = {}
    totals = {"graphs": 0, "nodes": 0, "matched_nodes": 0, "lexical_value_nodes": 0,
              "exact_graphs": 0, "skipped_over_limit": 0}
    for number, ref in enumerate(refs, 1):
        graph = graphs[ref]
        features, report, path = cache.get_or_encode(
            graph.ast, graph.source_path.read_bytes(), graph.language, batch_size=args.batch_size
        )
        if features.shape != (graph.num_nodes, encoder.output_dim):
            raise ValueError(f"Feature shape mismatch for {ref}: {tuple(features.shape)}")
        index[ref] = {"path": str(path.resolve()), "alignment": report.__dict__,
                      "nodes": graph.num_nodes, "sha256": path.stem}
        totals["graphs"] += 1
        totals["nodes"] += graph.num_nodes
        totals["matched_nodes"] += report.matched_nodes
        totals["lexical_value_nodes"] += report.lexical_value_nodes
        totals["exact_graphs"] += int(report.exact)
        if number % 25 == 0 or number == len(refs):
            print(f"[{number}/{len(refs)}] {ref}", flush=True)

    all_refs = set()
    for manifest in args.manifest:
        for row in read_manifest(manifest):
            all_refs.update((row["a"], row["b"]))
    totals["skipped_over_limit"] = len(all_refs) - len(refs)
    payload = {"corpus": args.corpus, "model": encoder.model_name,
               "feature_dim": encoder.output_dim, "max_nodes": args.max_nodes,
               "totals": totals, "graphs": index}
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
