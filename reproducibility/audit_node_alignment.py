"""Audit Tree-sitter/source alignment before extracting CodeBERT features."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codebert_features import LANGUAGE_ALIASES, node_texts_from_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="per corpus/language; 0 audits all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for corpus in ("AWS", "TransLibEval", "SALLM", "Juliet"):
        corpus_dir = args.datasets / corpus
        graphs_dir = corpus_dir / "graphs"
        if not graphs_dir.exists():
            continue
        for map_file in sorted(graphs_dir.glob("*-ast.txt")):
            raw_language = map_file.name.removesuffix("-ast.txt")
            language = LANGUAGE_ALIASES.get(raw_language, raw_language)
            json_file = map_file.with_suffix(".json")
            names = map_file.read_text(encoding="utf-8").splitlines()
            records = json_file.read_text(encoding="utf-8").splitlines()
            count = min(len(names), len(records))
            if args.limit:
                count = min(count, args.limit)
            for index in range(count):
                name = names[index].strip().replace("\\", "/")
                source_path = corpus_dir / name
                record = json.loads(records[index])
                if not source_path.exists():
                    rows.append({"corpus": corpus, "language": language, "name": name,
                                 "exists": False, "exact": False})
                    continue
                _, report = node_texts_from_source(record, source_path.read_bytes(), language)
                rows.append({"corpus": corpus, "language": language, "name": name,
                             "exists": True, **report.__dict__})

    counts = Counter((r["corpus"], r["language"], r.get("exact", False)) for r in rows)
    summary = {
        "files": len(rows),
        "exact": sum(bool(r.get("exact")) for r in rows),
        "missing_source": sum(not r.get("exists", False) for r in rows),
        "graph_nodes": sum(r.get("graph_nodes", 0) for r in rows),
        "matched_nodes": sum(r.get("matched_nodes", 0) for r in rows),
        "lexical_value_nodes": sum(r.get("lexical_value_nodes", 0) for r in rows),
        "by_corpus_language": [
            {"corpus": key[0], "language": key[1], "exact": key[2], "count": value}
            for key, value in sorted(counts.items())
        ],
        "mismatches": [r for r in rows if not r.get("exact", False)][:100],
    }
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
