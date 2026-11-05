"""Batch-safe TensorGAT matching the released final architecture.

Tensor decomposition is graph-only and deterministic, so it is cached outside
the trainable network.  This preserves the released equations while preventing
unrelated graphs in a mini-batch from changing one another's decompositions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, Set2Set


NUM_VIEWS = 4


def _adjacency(edge_lists: Sequence[torch.Tensor], n: int, device) -> torch.Tensor:
    tensor = torch.zeros((n, n, NUM_VIEWS), device=device)
    for view, edge in enumerate(edge_lists):
        if edge.numel():
            edge = edge.to(device)
            tensor[edge[0], edge[1], view] = 1.0
    return tensor


@torch.inference_mode()
def _t_svd(tensor: torch.Tensor, rank: int,
           method: str = "exact") -> torch.Tensor:
    transformed = torch.fft.fft(tensor, dim=2)
    rank = min(rank, tensor.shape[0], tensor.shape[1])
    if method == "exact":
        # cuSOLVER accepts a leading batch dimension.  Decomposing the four
        # Fourier slices in one call avoids four independent solver launches
        # while producing the same rank-truncated t-SVD.
        batched = transformed.permute(2, 0, 1)
        u, s, vh = torch.linalg.svd(batched, full_matrices=False)
        reconstructed = ((u[:, :, :rank] * s[:, None, :rank])
                         @ vh[:, :rank, :])
        stacked = reconstructed.permute(1, 2, 0)
        return torch.fft.ifft(stacked, dim=2).real.clamp_min_(0)
    if method != "randomized":
        raise ValueError(f"unsupported t-SVD method: {method}")
    slices = []
    devices = ([tensor.device.index] if tensor.device.type == "cuda" else [])
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        if tensor.device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        for view in range(transformed.shape[2]):
            matrix = transformed[:, :, view]
            # Only the leading ``rank`` components are used downstream.  A
            # small oversampled subspace avoids the cubic cost of a full SVD.
            q = min(rank + 8, min(matrix.shape))
            u, s, v = torch.svd_lowrank(matrix, q=q, niter=2)
            reconstructed = ((u[:, :rank] * s[:rank].unsqueeze(0))
                             @ v[:, :rank].mH)
            slices.append(reconstructed)
    return torch.fft.ifft(torch.stack(slices, dim=2), dim=2).real.clamp_min_(0)


def _normalize(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    degree = tensor.sum(dim=1).clamp_min(eps)
    inv = degree.pow(-0.5)
    return inv.unsqueeze(1) * tensor * inv.unsqueeze(0)


def _sparse_views(normalized: torch.Tensor, base: Sequence[torch.Tensor], topk: int, eps: float):
    n = normalized.shape[0]
    row_index = torch.arange(n, device=normalized.device)
    result = []
    for view in range(NUM_VIEWS):
        values, cols = torch.topk(normalized[:, :, view], k=min(topk, n), dim=1)
        mask = values > eps
        rows = row_index[:, None].expand_as(cols)[mask]
        edge = torch.stack((rows, cols[mask])) if rows.numel() else torch.empty((2, 0), dtype=torch.long, device=normalized.device)
        if base[view].numel():
            edge = torch.cat((edge, base[view].to(normalized.device)), dim=1)
        result.append(torch.unique(edge, dim=1))
    return result


def _cyclic_edges(views: Sequence[torch.Tensor], n: int) -> torch.Tensor:
    a1, a2, a3, a4 = views
    layout = (
        (a1, 0, 0), (a2, n, 0), (a3, 2*n, 0), (a4, 3*n, 0),
        (a4, 0, n), (a1, n, n), (a2, 2*n, n), (a3, 3*n, n),
        (a3, 0, 2*n), (a4, n, 2*n), (a1, 2*n, 2*n), (a2, 3*n, 2*n),
        (a2, 0, 3*n), (a3, n, 3*n), (a4, 2*n, 3*n), (a1, 3*n, 3*n),
    )
    shifted = []
    for edge, src_offset, dst_offset in layout:
        item = edge.clone()
        item[0] += src_offset
        item[1] += dst_offset
        shifted.append(item)
    return torch.cat(shifted, dim=1)


def _khatri_rao(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ir,jr->ijr", left, right).reshape(left.shape[0] * right.shape[0], -1)


@torch.inference_mode()
def _cp_factors(tensor: torch.Tensor, rank: int, iterations: int, eps: float):
    i, j, t = tensor.shape
    rank = min(rank, i, j)
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(0)
    o = torch.rand((i, rank), device=tensor.device, generator=generator).clamp_min(eps)
    p = torch.rand((j, rank), device=tensor.device, generator=generator).clamp_min(eps)
    q = torch.rand((t, rank), device=tensor.device, generator=generator).clamp_min(eps)
    eye = torch.eye(rank, device=tensor.device)
    for _ in range(iterations):
        kr = _khatri_rao(p, q); o = (tensor.reshape(i, j*t) @ kr) @ torch.linalg.pinv(kr.T @ kr + eps*eye)
        kr = _khatri_rao(o, q); p = (tensor.permute(1, 0, 2).reshape(j, i*t) @ kr) @ torch.linalg.pinv(kr.T @ kr + eps*eye)
        kr = _khatri_rao(o, p); q = (tensor.permute(2, 0, 1).reshape(t, i*j) @ kr) @ torch.linalg.pinv(kr.T @ kr + eps*eye)
        o = F.normalize(o.clamp_min(eps), p=2, dim=0)
        p = F.normalize(p.clamp_min(eps), p=2, dim=0)
        q = F.normalize(q.clamp_min(eps), p=2, dim=0)
    return o, p, q


@dataclass
class PreparedGraph:
    num_nodes: int
    loop_edge_index: torch.Tensor
    structural_edge_index: list[torch.Tensor]
    structural_weight: list[torch.Tensor]
    structural_attention: torch.Tensor | None = None

    def cpu_dict(self) -> dict:
        return {"num_nodes": self.num_nodes,
                # Node ids are below 4N (at most 2400 in these experiments),
                # so int32 is lossless on disk.  Forward converts back to the
                # int64 index type required by PyG.
                "loop_edge_index": self.loop_edge_index.cpu().int(),
                "structural_edge_index": [x.cpu() for x in self.structural_edge_index],
                "structural_weight": [x.cpu().half() for x in self.structural_weight],
                "structural_attention": (
                    self.structural_attention.cpu().half()
                    if self.structural_attention is not None else None
                )}

    @classmethod
    def from_dict(cls, value: dict) -> "PreparedGraph":
        return cls(value["num_nodes"], value["loop_edge_index"],
                   value.get("structural_edge_index", []),
                   value.get("structural_weight", []),
                   value.get("structural_attention"))


@torch.inference_mode()
def prepare_graph(edge_lists: Sequence[torch.Tensor], n: int, device="cuda", *,
                  use_t_svd=True, use_cp=True, svd_rank=32, topk=32,
                  cp_rank=16, cp_iters=3, eps=1e-8,
                  svd_method="exact", cp_attention_topk=0) -> PreparedGraph:
    device = torch.device(device)
    base = [edge.long().to(device) for edge in edge_lists]
    # With both decomposition branches disabled, normalization followed by
    # top-k sparsification contributes no new edges: _sparse_views appends the
    # complete base relation after selecting normalized nonzeros.  Returning
    # the cyclic base graph directly is therefore exactly equivalent and keeps
    # the original TensorGCN baseline free of unnecessary dense tensors.
    if not use_t_svd and not use_cp:
        return PreparedGraph(n, _cyclic_edges(base, n), [], [])
    tensor = _adjacency(base, n, device)
    if use_t_svd:
        tensor = _t_svd(tensor, svd_rank, method=svd_method)
    normalized = _normalize(tensor, eps)
    sparse = _sparse_views(normalized, base, topk, eps)
    loop = _cyclic_edges(sparse, n)
    structural_edges, structural_weights = [], []
    structural_attention = None
    if use_cp:
        o, p, q = _cp_factors(normalized, cp_rank, cp_iters, eps)
        scores = torch.einsum("ir,jr,tr->ijt", o, p, q)
        mask = normalized > eps
        mask |= torch.eye(n, dtype=torch.bool, device=device).unsqueeze(-1)
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        if cp_attention_topk > 0:
            k = min(cp_attention_topk, n)
            rows = torch.arange(n, device=device)[:, None].expand(n, k)
            for view in range(NUM_VIEWS):
                values, cols = torch.topk(masked_scores[:, :, view], k=k,
                                          dim=1)
                valid = torch.isfinite(values)
                # Every row contains at least its diagonal.  Masked softmax is
                # nevertheless used so small/degenerate graphs remain safe.
                safe_values = values.masked_fill(~valid, -1e9)
                weights = torch.softmax(safe_values, dim=1) * valid
                structural_edges.append(torch.stack((rows[valid], cols[valid])))
                structural_weights.append(weights[valid])
        else:
            structural_attention = torch.softmax(masked_scores, dim=1)
            structural_attention = torch.nan_to_num(structural_attention)
    return PreparedGraph(
        n, loop, structural_edges, structural_weights, structural_attention
    )


class MultiViewLayer(nn.Module):
    def __init__(self, dim: int, baseline: str = "gat",
                 use_structural_attention: bool = True) -> None:
        super().__init__()
        self.use_structural_attention = use_structural_attention
        if baseline == "gcn":
            self.conv1 = GCNConv(dim, dim, add_self_loops=False)
            self.conv2 = GCNConv(dim, dim, add_self_loops=False)
        else:
            self.conv1 = GATConv(dim, dim // NUM_VIEWS, heads=NUM_VIEWS, concat=True, add_self_loops=False)
            self.conv2 = GATConv(dim, dim // NUM_VIEWS, heads=NUM_VIEWS, concat=True, add_self_loops=False)

    def _combine_structural(self, views: torch.Tensor,
                            graph: PreparedGraph) -> torch.Tensor:
        """Apply the cached structural branch to one graph's view features."""
        if self.use_structural_attention and graph.structural_attention is not None:
            attention = graph.structural_attention.to(
                views.device, dtype=views.dtype
            )
            attended = torch.einsum("ijt,tjd->tid", attention, views)
            views = 0.5 * (views + attended)
        elif self.use_structural_attention and graph.structural_edge_index:
            # Backward-compatible reader for caches produced by earlier
            # versions of the reproducibility pipeline.
            attended = []
            for view, (indices, weights) in enumerate(zip(graph.structural_edge_index, graph.structural_weight)):
                indices = indices.to(views.device)
                weights = weights.to(views.device, dtype=views.dtype)
                out = torch.zeros_like(views[view])
                out.index_add_(0, indices[0], weights[:, None] * views[view, indices[1]])
                attended.append(out)
            views = 0.5 * (views + torch.stack(attended))
        return views.sum(dim=0)

    def _combine_sparse_many(self, view_blocks: Sequence[torch.Tensor],
                             graphs: Sequence[PreparedGraph]) -> list[torch.Tensor]:
        """Apply sparse CP attention to a graph batch with four index-adds."""
        sizes = [graph.num_nodes for graph in graphs]
        offsets, total = [], 0
        for size in sizes:
            offsets.append(total)
            total += size
        combined_views = []
        for view in range(NUM_VIEWS):
            values = torch.cat([block[view] for block in view_blocks], dim=0)
            shifted, weights = [], []
            for offset, graph in zip(offsets, graphs):
                edge = graph.structural_edge_index[view].to(
                    values.device, dtype=torch.long
                )
                shifted.append(edge + offset)
                weights.append(graph.structural_weight[view].to(
                    values.device, dtype=values.dtype
                ))
            attended = torch.zeros_like(values)
            if shifted:
                edge = torch.cat(shifted, dim=1)
                weight = torch.cat(weights)
                attended.index_add_(
                    0, edge[0], weight[:, None] * values[edge[1]]
                )
            combined_views.append(0.5 * (values + attended))
        combined = torch.stack(combined_views).sum(dim=0)
        return list(combined.split(sizes, dim=0))

    def forward_many(self, values: Sequence[torch.Tensor],
                     graphs: Sequence[PreparedGraph]) -> list[torch.Tensor]:
        """Run mathematically independent graphs as one block-diagonal batch."""
        if len(values) != len(graphs):
            raise ValueError("values/graphs length mismatch")
        if not values:
            return []
        repeated, shifted_edges, sizes = [], [], []
        offset = 0
        for value, graph in zip(values, graphs):
            if value.shape[0] != graph.num_nodes:
                raise ValueError("feature/graph node-count mismatch")
            repeated.append(value.repeat(NUM_VIEWS, 1))
            edge = graph.loop_edge_index.to(
                device=value.device, dtype=torch.long
            )
            shifted_edges.append(edge + offset)
            size = NUM_VIEWS * graph.num_nodes
            sizes.append(size)
            offset += size
        batched_values = torch.cat(repeated, dim=0)
        batched_edges = torch.cat(shifted_edges, dim=1)
        encoded = self.conv2(
            self.conv1(batched_values, batched_edges), batched_edges
        )
        outputs = []
        for block, graph in zip(encoded.split(sizes, dim=0), graphs):
            views = block.view(NUM_VIEWS, graph.num_nodes, -1)
            outputs.append(self._combine_structural(views, graph))
        return outputs

    def forward(self, x: torch.Tensor, graph: PreparedGraph) -> torch.Tensor:
        return self.forward_many([x], [graph])[0]


class FragmentEncoder(nn.Module):
    def __init__(self, input_dim=768, dim=64, baseline="gat",
                 use_structural_attention=True, normalize_input=False) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim) if normalize_input else nn.Identity()
        self.projection = nn.Linear(input_dim, dim)
        self.conv = MultiViewLayer(
            dim, baseline=baseline,
            use_structural_attention=use_structural_attention,
        )
        self.gru = nn.GRUCell(dim, dim)
        self.pool = Set2Set(dim, processing_steps=3)

    def forward_many(self, features: Sequence[torch.Tensor],
                     graphs: Sequence[PreparedGraph]):
        if len(features) != len(graphs):
            raise ValueError("features/graphs length mismatch")
        if not features:
            return []
        states = [F.relu(self.projection(self.input_norm(value))) for value in features]
        for _ in range(3):
            messages = self.conv.forward_many(states, graphs)
            states = [
                self.gru(F.relu(message), state)
                for message, state in zip(messages, states)
            ]
        # Set2Set is inexpensive relative to graph propagation.  Pool each
        # disconnected graph separately so batching changes neither reduction
        # order nor floating-point output for the graph-level representation.
        pooled = [
            self.pool(
                state,
                torch.zeros(
                    state.shape[0], dtype=torch.long, device=state.device
                ),
            ).squeeze(0)
            for state in states
        ]
        return list(zip(pooled, states))

    def forward(self, features: torch.Tensor, graph: PreparedGraph):
        return self.forward_many([features], [graph])[0]


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(input_dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU())
        self.shortcut = nn.Linear(input_dim, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value) + self.shortcut(value)


def local_global_jsd(local: list[torch.Tensor], global_: torch.Tensor,
                     local_projector: nn.Module, global_projector: nn.Module) -> torch.Tensor:
    """Device-safe form of the repository's JSD local-global objective."""
    if len(local) <= 1:
        return global_.new_zeros(())
    projected_local = torch.cat([local_projector(value) for value in local], dim=0)
    projected_global = global_projector(global_)
    owners = torch.cat([
        torch.full((value.shape[0],), index, dtype=torch.long, device=value.device)
        for index, value in enumerate(local)
    ])
    scores = projected_local @ projected_global.T
    positive = F.softplus(-scores).neg().add(torch.log(scores.new_tensor(2.0)))
    negative = F.softplus(-scores) + scores - torch.log(scores.new_tensor(2.0))
    mask = F.one_hot(owners, num_classes=len(local)).bool()
    return negative[~mask].mean() - positive[mask].mean()


class TensorPairClassifier(nn.Module):
    def __init__(self, input_dim=768, dim=64, baseline="gat", use_unsup=True,
                 shared_encoder=False, pair_interactions=False,
                 use_structural_attention=True, normalize_input=False,
                 symmetric_pair_head=False, normalize_pair_embeddings=False,
                 head_layer_norm=False, head_dropout=0.0) -> None:
        super().__init__()
        self.use_unsup = use_unsup
        self.pair_interactions = pair_interactions
        self.symmetric_pair_head = symmetric_pair_head
        self.encoder1 = FragmentEncoder(
            input_dim, dim, baseline, use_structural_attention, normalize_input
        )
        self.encoder2 = self.encoder1 if shared_encoder else FragmentEncoder(
            input_dim, dim, baseline, use_structural_attention, normalize_input
        )
        embedding_dim = 2 * dim
        self.pair_embedding_norm = (
            nn.LayerNorm(embedding_dim)
            if normalize_pair_embeddings else nn.Identity()
        )
        pair_input_dim = (
            2 * embedding_dim if symmetric_pair_head
            else (4 * embedding_dim if pair_interactions else 2 * embedding_dim)
        )
        self.fc1 = nn.Linear(pair_input_dim, dim)
        self.head_norm = nn.LayerNorm(dim) if head_layer_norm else nn.Identity()
        self.head_dropout = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(dim, 1)
        if use_unsup:
            self.local_d1, self.local_d2 = FeedForward(dim, dim), FeedForward(dim, dim)
            self.global_d1, self.global_d2 = FeedForward(2 * dim, dim), FeedForward(2 * dim, dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: Sequence[dict], return_unsup=False,
                return_embeddings=False):
        # Encode each distinct fragment once per mini-batch.  Decompositions
        # were prepared per graph, so graph membership in the batch is inert.
        left_items, right_items = {}, {}
        for item in batch:
            left_items.setdefault(
                item["a"], (item["features_a"], item["graph_a"])
            )
            right_items.setdefault(
                item["b"], (item["features_b"], item["graph_b"])
            )
        left_keys, right_keys = list(left_items), list(right_items)
        left_encoded = self.encoder1.forward_many(
            [left_items[key][0] for key in left_keys],
            [left_items[key][1] for key in left_keys],
        )
        right_encoded = self.encoder2.forward_many(
            [right_items[key][0] for key in right_keys],
            [right_items[key][1] for key in right_keys],
        )
        left = dict(zip(left_keys, left_encoded))
        right = dict(zip(right_keys, right_encoded))
        paired_rows, paired_left, paired_right = [], [], []
        for item in batch:
            left_embedding = self.pair_embedding_norm(left[item["a"]][0])
            right_embedding = self.pair_embedding_norm(right[item["b"]][0])
            paired_left.append(left_embedding)
            paired_right.append(right_embedding)
            if self.symmetric_pair_head:
                # Clone similarity is invariant to pair order.  Removing raw
                # language-side coordinates prevents the small classifier from
                # learning C/C++-specific marginal shortcuts while retaining
                # the shared TensorGAT encoder and its CP/t-SVD graph views.
                values = [(left_embedding - right_embedding).abs(),
                          left_embedding * right_embedding]
            else:
                values = [left_embedding, right_embedding]
            if self.pair_interactions and not self.symmetric_pair_head:
                values.extend(((left_embedding - right_embedding).abs(),
                               left_embedding * right_embedding))
            paired_rows.append(torch.cat(values))
        paired = torch.stack(paired_rows)
        hidden = self.head_dropout(F.relu(self.head_norm(self.fc1(paired))))
        logits = self.fc2(hidden).squeeze(-1)
        if not return_unsup and not return_embeddings:
            return logits
        # Compute local-global discrimination over unique fragments.  A grouped
        # positive/negative batch repeats its left anchor; treating that repeat
        # as a separate owner would incorrectly make an identical embedding a
        # negative example of itself.
        if return_unsup and self.use_unsup:
            left_values = list(left.values())
            right_values = list(right.values())
            left_global = torch.stack([value[0] for value in left_values])
            right_global = torch.stack([value[0] for value in right_values])
            left_local = [value[1] for value in left_values]
            right_local = [value[1] for value in right_values]
            unsup = local_global_jsd(left_local, left_global, self.local_d1, self.global_d1)
            unsup = unsup + local_global_jsd(right_local, right_global, self.local_d2, self.global_d2)
        else:
            unsup = logits.new_zeros(())
        if return_embeddings:
            return logits, unsup, torch.stack(paired_left), torch.stack(paired_right)
        return logits, unsup
