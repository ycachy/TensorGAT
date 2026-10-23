"""Dataset readers for the corrected TensorGAT experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


VIEW_KEYS = ("astEdges", "cfgEdges", "dfgEdges", "ncsEdges")


@dataclass
class GraphRecord:
    key: str
    language: str
    source_path: Path
    ast: dict

    @property
    def num_nodes(self) -> int:
        return len(self.ast["nodeList"])

    def edge_lists(self, enabled_views=(True, True, True, True)) -> list[torch.Tensor]:
        result = []
        for view, (key, enabled) in enumerate(zip(VIEW_KEYS, enabled_views)):
            if not enabled:
                result.append(torch.empty((2, 0), dtype=torch.long))
                continue
            # Some generators called data-dependence edges ``ddgEdges`` and
            # others called them ``dfgEdges``.  Use the available field; an
            # absent view stays empty and is never replaced with AST edges.
            if key == "dfgEdges":
                edges = self.ast.get("ddgEdges", self.ast.get("dfgEdges", []))
            else:
                edges = self.ast.get(key, [])
            valid = [
                (int(src), int(dst)) for src, dst in edges
                if 0 <= int(src) < self.num_nodes and 0 <= int(dst) < self.num_nodes
            ]
            result.append(
                torch.tensor(valid, dtype=torch.long).T.contiguous()
                if valid else torch.empty((2, 0), dtype=torch.long)
            )
        return result


class GraphCorpusIndex:
    """Random access to JSONL graph records using their companion name files."""

    def __init__(self, corpus_dir: str | Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        self._records: dict[str, GraphRecord] = {}
        graph_dir = self.corpus_dir / "graphs"
        for names_path in sorted(graph_dir.glob("*-ast.txt")):
            language = names_path.name.removesuffix("-ast.txt")
            json_path = names_path.with_suffix(".json")
            names = names_path.read_text(encoding="utf-8").splitlines()
            records = json_path.read_text(encoding="utf-8").splitlines()
            if len(names) != len(records):
                raise ValueError(f"Graph index length mismatch: {names_path}")
            for name, raw in zip(names, records):
                key = name.strip().replace("\\", "/")
                self._records[key] = GraphRecord(
                    key=key,
                    language=language,
                    source_path=self.corpus_dir / key,
                    ast=json.loads(raw),
                )

    def __getitem__(self, key: str) -> GraphRecord:
        return self._records[key.replace("\\", "/")]

    def __contains__(self, key: str) -> bool:
        return key.replace("\\", "/") in self._records


def read_manifest(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def filter_pairs_by_size(rows: list[dict], graphs: GraphCorpusIndex, max_nodes: int) -> tuple[list[dict], list[dict]]:
    kept, excluded = [], []
    for row in rows:
        if graphs[row["a"]].num_nodes <= max_nodes and graphs[row["b"]].num_nodes <= max_nodes:
            kept.append(row)
        else:
            excluded.append(row)
    return kept, excluded

