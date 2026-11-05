"""Convert an existing dense CP-attention cache to an equivalent top-k cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert(source: Path, target: Path, topk: int) -> None:
    value = torch.load(source, map_location="cpu", weights_only=True)
    attention = value.get("structural_attention")
    if attention is None:
        raise ValueError(f"dense structural attention missing: {source}")
    attention = attention.float()
    n, _, views = attention.shape
    k = min(topk, n)
    rows = torch.arange(n)[:, None].expand(n, k)
    edge_indices, weights = [], []
    for view in range(views):
        selected, columns = torch.topk(attention[:, :, view], k=k, dim=1)
        positive = selected > 0
        selected = selected / selected.sum(dim=1, keepdim=True).clamp_min(1e-12)
        edge_indices.append(torch.stack((rows[positive], columns[positive])))
        weights.append(selected[positive].half())
    value["structural_edge_index"] = edge_indices
    value["structural_weight"] = weights
    value["structural_attention"] = None
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--topk", type=int, required=True)
    args = parser.parse_args()
    files = sorted(args.source.glob("*.pt"))
    converted = 0
    for number, source in enumerate(files, 1):
        target = args.target / source.name
        if not target.exists():
            convert(source, target, args.topk)
            converted += 1
        if number % 100 == 0 or number == len(files):
            print(f"[{number}/{len(files)}] converted={converted}", flush=True)


if __name__ == "__main__":
    main()
