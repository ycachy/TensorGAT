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
def _t_svd(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    transformed = torch.fft.fft(tensor, dim=2)
    slices = []
    rank = min(rank, tensor.shape[0], tensor.shape[1])
    for view in range(transformed.shape[2]):
        u, s, vh = torch.linalg.svd(transformed[:, :, view], full_matrices=False)
        slices.append((u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :])
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

    def cpu_dict(self) -> dict:
        return {"num_nodes": self.num_nodes, "loop_edge_index": self.loop_edge_index.cpu(),
                "structural_edge_index": [x.cpu() for x in self.structural_edge_index],
                "structural_weight": [x.cpu().half() for x in self.structural_weight]}

    @classmethod
    def from_dict(cls, value: dict) -> "PreparedGraph":
        return cls(value["num_nodes"], value["loop_edge_index"],
                   value["structural_edge_index"], value["structural_weight"])


@torch.inference_mode()
def prepare_graph(edge_lists: Sequence[torch.Tensor], n: int, device="cuda", *,
                  use_t_svd=True, use_cp=True, svd_rank=32, topk=32,
                  cp_rank=16, cp_iters=3, eps=1e-8) -> PreparedGraph:
    device = torch.device(device)
    base = [edge.long().to(device) for edge in edge_lists]
    tensor = _adjacency(base, n, device)
    if use_t_svd:
        tensor = _t_svd(tensor, svd_rank)
    normalized = _normalize(tensor, eps)
    sparse = _sparse_views(normalized, base, topk, eps)
    loop = _cyclic_edges(sparse, n)
    structural_edges, structural_weights = [], []
    if use_cp:
        o, p, q = _cp_factors(normalized, cp_rank, cp_iters, eps)
        scores = torch.einsum("ir,jr,tr->ijt", o, p, q)
        mask = normalized > eps
        mask |= torch.eye(n, dtype=torch.bool, device=device).unsqueeze(-1)
        attention = torch.softmax(scores.masked_fill(~mask, -1e9), dim=1)
        attention = torch.nan_to_num(attention)
        for view in range(NUM_VIEWS):
            edge = mask[:, :, view].nonzero(as_tuple=False).T.contiguous()
            structural_edges.append(edge)
            structural_weights.append(attention[edge[0], edge[1], view])
    return PreparedGraph(n, loop, structural_edges, structural_weights)


class MultiViewLayer(nn.Module):
    def __init__(self, dim: int, baseline: str = "gat") -> None:
        super().__init__()
        if baseline == "gcn":
            self.conv1 = GCNConv(dim, dim, add_self_loops=False)
            self.conv2 = GCNConv(dim, dim, add_self_loops=False)
        else:
            self.conv1 = GATConv(dim, dim // NUM_VIEWS, heads=NUM_VIEWS, concat=True, add_self_loops=False)
            self.conv2 = GATConv(dim, dim // NUM_VIEWS, heads=NUM_VIEWS, concat=True, add_self_loops=False)

    def forward(self, x: torch.Tensor, graph: PreparedGraph) -> torch.Tensor:
        edge = graph.loop_edge_index.to(x.device).clone()
        views = self.conv2(self.conv1(x.repeat(NUM_VIEWS, 1), edge), edge).view(NUM_VIEWS, graph.num_nodes, -1)
        if graph.structural_edge_index:
            attended = []
            for view, (indices, weights) in enumerate(zip(graph.structural_edge_index, graph.structural_weight)):
                indices = indices.to(x.device).clone()
                weights = weights.to(x.device, dtype=x.dtype).clone()
                out = torch.zeros_like(views[view])
                out.index_add_(0, indices[0], weights[:, None] * views[view, indices[1]])
                attended.append(out)
            views = 0.5 * (views + torch.stack(attended))
        return views.sum(dim=0)


class FragmentEncoder(nn.Module):
    def __init__(self, input_dim=768, dim=64, baseline="gat") -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, dim)
        self.conv = MultiViewLayer(dim, baseline=baseline)
        self.gru = nn.GRUCell(dim, dim)
        self.pool = Set2Set(dim, processing_steps=3)

    def forward(self, features: torch.Tensor, graph: PreparedGraph):
        state = F.relu(self.projection(features))
        for _ in range(3):
            state = self.gru(F.relu(self.conv(state, graph)), state)
        batch = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
        return self.pool(state, batch).squeeze(0), state


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
    def __init__(self, input_dim=768, dim=64, baseline="gat", use_unsup=True) -> None:
        super().__init__()
        self.use_unsup = use_unsup
        self.encoder1 = FragmentEncoder(input_dim, dim, baseline)
        self.encoder2 = FragmentEncoder(input_dim, dim, baseline)
        self.fc1 = nn.Linear(4 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        if use_unsup:
            self.local_d1, self.local_d2 = FeedForward(dim, dim), FeedForward(dim, dim)
            self.global_d1, self.global_d2 = FeedForward(2 * dim, dim), FeedForward(2 * dim, dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: Sequence[dict], return_unsup=False):
        # Encode each distinct fragment once per mini-batch.  Decompositions
        # were prepared per graph, so graph membership in the batch is inert.
        left, right = {}, {}
        for item in batch:
            if item["a"] not in left:
                left[item["a"]] = self.encoder1(item["features_a"], item["graph_a"])
            if item["b"] not in right:
                right[item["b"]] = self.encoder2(item["features_b"], item["graph_b"])
        paired = torch.stack([torch.cat((left[item["a"]][0], right[item["b"]][0])) for item in batch])
        logits = self.fc2(F.relu(self.fc1(paired))).squeeze(-1)
        if not return_unsup or not self.use_unsup:
            return logits
        # Repeat representations in pair order, as the original paired loader
        # did, so frequently occurring fragments receive the same weighting.
        left_global = torch.stack([left[item["a"]][0] for item in batch])
        right_global = torch.stack([right[item["b"]][0] for item in batch])
        left_local = [left[item["a"]][1] for item in batch]
        right_local = [right[item["b"]][1] for item in batch]
        unsup = local_global_jsd(left_local, left_global, self.local_d1, self.global_d1)
        unsup = unsup + local_global_jsd(right_local, right_global, self.local_d2, self.global_d2)
        return logits, unsup
