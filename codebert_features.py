"""CodeBERT node features aligned to the AST stored by TensorGAT.

The original repository used language-specific one-hot node-type vectors even
though the paper describes CodeBERT features.  This module implements the
paper's feature definition reproducibly: each graph node is represented by its
Tree-sitter type and, for leaf nodes, the exact source text covered by that
node.  A frozen ``microsoft/codebert-base`` encoder produces one 768-dimensional
vector per node.  Features can be cached because the encoder is not fine-tuned.
"""
from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "java": "java",
    "c": "c",
    "cc": "cpp",
    "cpp": "cpp",
    "c++": "cpp",
}

NCS_EDGE_VERSION = "source-token-leaves-v1"


@dataclass(frozen=True)
class AlignmentReport:
    exact: bool
    graph_nodes: int
    parsed_nodes: int
    matched_prefix: int
    matched_nodes: int
    lexical_value_nodes: int
    contextual_nodes: int = 0


def _preorder(root) -> list:
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(reversed(node.children))
    return result


def _parser_nodes(source: bytes, language: str):
    language = LANGUAGE_ALIASES[language.lower()]
    from tree_sitter import Language, Parser
    if language == "python":
        import tree_sitter_python as grammar
    elif language == "java":
        import tree_sitter_java as grammar
    elif language == "c":
        import tree_sitter_c as grammar
    elif language == "cpp":
        import tree_sitter_cpp as grammar
    else:
        raise ValueError(f"Unsupported language: {language}")
    return _preorder(Parser(Language(grammar.language())).parse(source).root_node)


def _align_graph_nodes(ast_record: dict, source: bytes, language: str):
    graph_types = [str(node["type"]) for node in ast_record["nodeList"]]
    parsed = _parser_nodes(source, language)
    parsed_types = [node.type for node in parsed]
    original_ids = [node.get("_original_id") for node in ast_record["nodeList"]]
    if original_ids and all(isinstance(index, int) for index in original_ids):
        metadata = ast_record.get("_reviewer1_comment3", {})
        original_types = metadata.get("original_node_types")
        if original_types:
            matcher = SequenceMatcher(a=original_types, b=parsed_types,
                                      autojunk=False)
            original_to_parsed = {}
            for block in matcher.get_matching_blocks():
                for offset in range(block.size):
                    original_to_parsed[block.a + offset] = block.b + offset
            graph_to_parsed = {
                graph_index: original_to_parsed[original_id]
                for graph_index, original_id in enumerate(original_ids)
                if original_id in original_to_parsed
            }
        else:
            graph_to_parsed = {
                graph_index: parsed_index
                for graph_index, parsed_index in enumerate(original_ids)
                if 0 <= parsed_index < len(parsed)
                and graph_types[graph_index] == parsed_types[parsed_index]
            }
        aligned = [
            parsed[graph_to_parsed[index]] if index in graph_to_parsed else None
            for index in range(len(graph_types))
        ]
        matched_prefix = 0
        for graph_index, parsed_index in enumerate(original_ids):
            if parsed_index != graph_index or graph_index not in graph_to_parsed:
                break
            matched_prefix += 1
        exact = (len(graph_types) == len(parsed_types)
                 and len(graph_to_parsed) == len(graph_types))
        return (graph_types, aligned, exact, len(parsed), matched_prefix,
                len(graph_to_parsed))
    matched_prefix = 0
    for graph_type, parsed_type in zip(graph_types, parsed_types):
        if graph_type != parsed_type:
            break
        matched_prefix += 1
    exact = graph_types == parsed_types
    if exact:
        graph_to_parsed = {index: index for index in range(len(graph_types))}
    else:
        matcher = SequenceMatcher(a=graph_types, b=parsed_types, autojunk=False)
        graph_to_parsed = {}
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                graph_to_parsed[block.a + offset] = block.b + offset
    aligned = [parsed[graph_to_parsed[index]] if index in graph_to_parsed else None
               for index in range(len(graph_types))]
    return graph_types, aligned, exact, len(parsed), matched_prefix, len(graph_to_parsed)


def natural_code_sequence_edges(
    ast_record: dict,
    source: bytes,
    language: str,
) -> tuple[list[tuple[int, int]], AlignmentReport]:
    """Connect consecutive lexical AST leaves in source-code order.

    TensorGAT graph records number every AST node in preorder.  Consecutive
    node ids therefore describe an AST traversal, not the program's natural
    token sequence.  NCS contains only terminal syntax nodes that can be
    aligned to source spans and orders them by their byte positions.
    """
    graph_types, aligned, exact, parsed_count, matched_prefix, matched_count = _align_graph_nodes(
        ast_record, source, language
    )
    lexical = sorted(
        (
            int(node.start_byte),
            int(node.end_byte),
            graph_index,
        )
        for graph_index, node in enumerate(aligned)
        if node is not None and node.child_count == 0 and node.end_byte > node.start_byte
    )
    indices = [graph_index for _, _, graph_index in lexical]
    edges = list(zip(indices, indices[1:]))
    report = AlignmentReport(
        exact=exact,
        graph_nodes=len(graph_types),
        parsed_nodes=parsed_count,
        matched_prefix=matched_prefix,
        matched_nodes=matched_count,
        lexical_value_nodes=len(indices),
    )
    return edges, report


def node_texts_from_source(
    ast_record: dict,
    source: bytes,
    language: str,
) -> tuple[list[str], AlignmentReport]:
    """Return ``type </s> value`` labels in graph-node order.

    Exact Tree-sitter preorder alignment is required to attach source spans. If
    parser versions disagree, node types remain available and are used as a
    deterministic fallback; the report makes that fallback auditable.
    """
    graph_types, aligned, exact, parsed_count, matched_prefix, matched_count = _align_graph_nodes(
        ast_record, source, language
    )

    labels = []
    lexical_value_nodes = 0
    for node_type, node in zip(graph_types, aligned):
        # Internal-node spans contain whole subtrees and would duplicate the
        # complete function many times.  The lexical value is well-defined for
        # leaves; internal nodes are represented by their syntax type.
        if node is not None and node.child_count == 0:
            value = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            value = " ".join(value.split())
            labels.append(f"{node_type} </s> {value}" if value else node_type)
            lexical_value_nodes += bool(value)
        else:
            labels.append(node_type)
    report = AlignmentReport(
        exact,
        len(graph_types),
        parsed_count,
        matched_prefix,
        matched_count,
        int(lexical_value_nodes),
    )
    return labels, report


class CodeBERTNodeEncoder:
    """Frozen batched CodeBERT encoder for AST-node labels."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: str | torch.device | None = None,
        max_length: int = 32,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.output_dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, labels: Sequence[str], batch_size: int = 128) -> torch.Tensor:
        chunks = []
        for start in range(0, len(labels), batch_size):
            batch = list(labels[start : start + batch_size])
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            chunks.append(pooled.cpu())
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0, self.output_dim))

    @torch.inference_mode()
    def encode_graph(self, ast_record: dict, source: bytes, language: str,
                     window_batch_size: int = 16) -> tuple[torch.Tensor, AlignmentReport]:
        """Contextualize the full source and pool token spans into AST nodes."""
        graph_types, aligned, exact, parsed_count, prefix, matched = _align_graph_nodes(
            ast_record, source, language
        )
        text = source.decode("utf-8", errors="replace")
        # Tree-sitter uses byte offsets; fast tokenizers return character offsets.
        byte_to_char = [0] * (len(source) + 1)
        byte_position = 0
        for char_position, character in enumerate(text):
            encoded = character.encode("utf-8", errors="replace")
            for offset in range(len(encoded)):
                if byte_position + offset < len(byte_to_char):
                    byte_to_char[byte_position + offset] = char_position
            byte_position += len(encoded)
        byte_to_char[min(byte_position, len(source))] = len(text)
        for position in range(1, len(byte_to_char)):
            if byte_to_char[position] == 0 and position != 0:
                byte_to_char[position] = byte_to_char[position - 1]

        tokenized = self.tokenizer(
            text, truncation=True, max_length=512, stride=128,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding=True, return_tensors="pt",
        )
        offsets = tokenized.pop("offset_mapping")
        tokenized.pop("overflow_to_sample_mapping", None)
        span_sum, span_count = {}, {}
        windows = int(tokenized["input_ids"].shape[0])
        for start in range(0, windows, window_batch_size):
            end = min(start + window_batch_size, windows)
            inputs = {key: value[start:end].to(self.device) for key, value in tokenized.items()}
            hidden = self.model(**inputs).last_hidden_state.cpu()
            for window in range(end - start):
                for token, (left, right) in enumerate(offsets[start + window].tolist()):
                    if right <= left:
                        continue
                    key = (int(left), int(right))
                    span_sum[key] = span_sum.get(key, 0) + hidden[window, token]
                    span_count[key] = span_count.get(key, 0) + 1
        spans = sorted(span_sum)
        contextual = {span: span_sum[span] / span_count[span] for span in spans}

        unique_types = sorted(set(graph_types))
        type_matrix = self.encode(unique_types, batch_size=128)
        type_vectors = dict(zip(unique_types, type_matrix))
        features, contextual_nodes, lexical_nodes = [], 0, 0
        for node_type, node in zip(graph_types, aligned):
            type_vector = type_vectors[node_type]
            if node is None:
                features.append(type_vector)
                continue
            left = byte_to_char[min(node.start_byte, len(source))]
            right = byte_to_char[min(node.end_byte, len(source))]
            covered = [contextual[span] for span in spans if span[0] < right and span[1] > left]
            if covered:
                context_vector = torch.stack(covered).mean(dim=0)
                features.append(0.5 * (context_vector + type_vector))
                contextual_nodes += 1
                lexical_nodes += int(node.child_count == 0)
            else:
                features.append(type_vector)
        report = AlignmentReport(exact, len(graph_types), parsed_count, prefix, matched,
                                 lexical_nodes, contextual_nodes)
        return torch.stack(features), report


class CodeBERTFeatureCache:
    """Content-addressed cache for frozen CodeBERT node features."""

    def __init__(self, cache_dir: str | Path, encoder: CodeBERTNodeEncoder) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder

    def _key(self, ast_record: dict, source: bytes, language: str) -> str:
        payload = json.dumps(ast_record, sort_keys=True, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(b"contextual-source-v1")
        digest.update(self.encoder.model_name.encode("utf-8"))
        digest.update(str(self.encoder.max_length).encode("ascii"))
        digest.update(LANGUAGE_ALIASES[language.lower()].encode("ascii"))
        digest.update(payload)
        digest.update(source)
        return digest.hexdigest()

    def get_or_encode(
        self,
        ast_record: dict,
        source: bytes,
        language: str,
        batch_size: int = 128,
    ) -> tuple[torch.Tensor, AlignmentReport, Path]:
        key = self._key(ast_record, source, language)
        target = self.cache_dir / f"{key}.pt"
        if target.exists():
            saved = torch.load(target, map_location="cpu", weights_only=True)
            report = AlignmentReport(**saved["alignment"])
            return saved["features"].float(), report, target

        features, report = self.encoder.encode_graph(
            ast_record, source, language, window_batch_size=max(1, batch_size // 16)
        )
        features = features.to(torch.float16)
        torch.save(
            {"features": features, "alignment": report.__dict__},
            target,
        )
        return features.float(), report, target
