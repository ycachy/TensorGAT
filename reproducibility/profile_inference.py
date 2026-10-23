"""Profile cached-model inference with synchronized CUDA timing."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.data import GraphCorpusIndex, filter_pairs_by_size, read_manifest
from reproducibility.model import TensorPairClassifier
from reproducibility.train import Artifacts, chunks, stable_balance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("full", "no_cp", "no_svd", "no_cp_no_svd", "gcn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda")
    graphs = GraphCorpusIndex(args.datasets / args.corpus)
    rows, excluded = filter_pairs_by_size(read_manifest(args.manifest), graphs, 600)
    rows = stable_balance(rows, 1234)
    artifacts = Artifacts(args.features, args.prepared, device)
    batches = [artifacts.batch(part) for part in chunks(rows, args.batch_size)]
    model = TensorPairClassifier(baseline="gcn" if args.variant == "gcn" else "gat").to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    with torch.inference_mode():
        for batch in batches[:2]: model(batch)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        for batch in batches: model(batch)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    result = {"variant": args.variant, "pairs": len(rows), "excluded": len(excluded),
              "batch_size": args.batch_size, "seconds": elapsed,
              "milliseconds_per_pair": 1000*elapsed/len(rows),
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "parameters": sum(p.numel() for p in model.parameters()),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__": main()
