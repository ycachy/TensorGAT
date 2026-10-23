"""Create a deterministic balanced external-test manifest without new labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    read = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    positive, negative = read(args.positive), read(args.negative)
    key = lambda row: hashlib.sha256(f"{args.seed}:{row['id']}".encode()).hexdigest()
    count = min(len(positive), len(negative))
    rows = sorted(sorted(positive, key=key)[:count] + sorted(negative, key=key)[:count], key=key)
    rows = [dict(row, split="external_test") for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"positive": count, "negative": count, "total": 2*count,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
