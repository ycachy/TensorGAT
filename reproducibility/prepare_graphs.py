"""Cache per-fragment t-SVD/CP graph preprocessing for one experiment variant."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.data import GraphCorpusIndex, read_manifest
from reproducibility.model import prepare_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("full", "no_cp", "no_svd", "no_cp_no_svd", "gcn"), default="full")
    parser.add_argument("--drop-view", choices=("ast", "cfg", "ddg", "ncs"))
    parser.add_argument("--max-nodes", type=int, default=600)
    args = parser.parse_args()

    graphs = GraphCorpusIndex(args.datasets / args.corpus)
    all_refs = set()
    for path in args.manifest:
        for row in read_manifest(path):
            all_refs.update((row["a"], row["b"]))
    refs = sorted(ref for ref in all_refs if graphs[ref].num_nodes <= args.max_nodes)
    enabled = [True] * 4
    if args.drop_view:
        enabled[("ast", "cfg", "ddg", "ncs").index(args.drop_view)] = False
    use_t_svd = args.variant not in ("no_svd", "no_cp_no_svd", "gcn")
    use_cp = args.variant not in ("no_cp", "no_cp_no_svd", "gcn")
    config = {"variant": args.variant, "drop_view": args.drop_view, "enabled_views": enabled,
              "max_nodes": args.max_nodes, "svd_rank": 32, "topk": 32,
              "cp_rank": 16, "cp_iters": 3}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = args.output / config_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for number, ref in enumerate(refs, 1):
        graph = graphs[ref]
        key = hashlib.sha256((args.corpus + ":" + ref).encode()).hexdigest()
        target = cache_dir / f"{key}.pt"
        if not target.exists():
            prepared = prepare_graph(graph.edge_lists(enabled), graph.num_nodes,
                                     use_t_svd=use_t_svd, use_cp=use_cp)
            torch.save(prepared.cpu_dict(), target)
        index[ref] = str(target.resolve())
        if number % 25 == 0 or number == len(refs):
            print(f"[{number}/{len(refs)}] {ref}", flush=True)
    payload = {"corpus": args.corpus, "config": config, "config_hash": config_hash,
               "graphs": index, "skipped_over_limit": len(all_refs) - len(index)}
    index_path = cache_dir / "index.json"
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(index_path.resolve())


if __name__ == "__main__":
    main()
