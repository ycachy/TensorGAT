from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from model.layers import MPLayer, InterConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List


class Tensor_GGNN_GCN(MessagePassing):
    def __init__(self,
                 num_edge_types,
                 # in_features,
                 out_features,
                 embedding_features,
                 # classifier_features,
                 # embedding_num_classes,
                 dropout=0,
                 max_node_per_graph=600,
                 # max_variable_candidates=5,
                 add_self_loops=False,
                 bias=True,
                 aggr="mean",
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 output_model="learning",
                 use_t_svd=True,
                 t_svd_rank=32,
                 t_svd_topk=32,
                 use_cp_attention=True,
                 cp_rank=16,
                 cp_iters=3,
                 tensor_eps=1e-8):
        super(Tensor_GGNN_GCN, self).__init__(aggr=aggr)
        # params set
        self.num_edge_types = num_edge_types
        self.device = device
        self.output_model = output_model.lower()
        self.dropout = dropout
        self.max_node_per_graph = max_node_per_graph
        self.add_self_loops = add_self_loops
        self.use_t_svd = use_t_svd
        self.t_svd_rank = t_svd_rank
        self.t_svd_topk = t_svd_topk
        self.use_cp_attention = use_cp_attention
        self.cp_rank = cp_rank
        self.cp_iters = cp_iters
        self.tensor_eps = tensor_eps
        # self.max_variable_candidates = max_variable_candidates
        # 先对值进行embedding
        # self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
        # in_features,
        # embedding_features,
        # device=device)

        self.MessagePassingNN = nn.ModuleList(
            [
                MPLayer(in_features=embedding_features, out_features=out_features, device=device) for _ in
                range(self.num_edge_types)
            ]
        )

        self.gru_cell = torch.nn.GRUCell(input_size=embedding_features, hidden_size=out_features)

        # self.conv1_list = nn.ModuleList(
        #     [GCNConv(out_features, out_features, add_self_loops=add_self_loops) for _ in range(self.num_edge_types)])
        # self.interConv = InterConv(out_features, out_features, num_edge_types=self.num_edge_types)
        # self.thirdNorm = torch.nn.InstanceNorm2d(out_features)  # 这个就是自己要找的
        #
        # self.lin = nn.Linear(out_features * 2, out_features)
        # self.conv1 = GCNConv(out_features, out_features, add_self_loops=add_self_loops)
        # self.conv2 = GCNConv(out_features, out_features, add_self_loops=add_self_loops)
        self.conv1_list = nn.ModuleList(
            [GATConv(out_features, out_features // num_edge_types, add_self_loops=add_self_loops, heads=num_edge_types,
                     concat=True) for _ in range(self.num_edge_types)])
        self.interConv = InterConv(out_features, out_features, num_edge_types=self.num_edge_types)
        self.thirdNorm = torch.nn.InstanceNorm2d(out_features)  # 这个就是自己要找的

        self.lin = nn.Linear(out_features * 2, out_features)
        self.conv1 = GATConv(out_features, out_features // num_edge_types, add_self_loops=add_self_loops,
                             heads=num_edge_types, concat=True)
        self.conv2 = GATConv(out_features, out_features // num_edge_types, add_self_loops=add_self_loops,
                             heads=num_edge_types, concat=True)

        # self.varmisuse_output_layer = nn.Linear(out_features* 2 + 1, 1)
        # self.varnaming_output_layer = nn.Linear(out_features, classifier_features)

    # def forward(self, x1, edge_list1: List[torch.tensor], x2, edge_list2: List[torch.tensor]):
    '''
    def forward(self, x1, edge_list1: List[torch.tensor]):
        # x_embedding = x1
        # #x_embedding = self.value_embeddingLayer(x)
        # # Tensor GGNN 
        # last_node_states = x_embedding
        # for _ in range(4):
        #     out_list = []
        #     cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
        #     for i in range(len(edge_list1)):
        #         edge = edge_list1[i]
        #         print(edge)
        #         if edge.shape[0] != 0 :
        #             # 该种类型的边存在边
        #             out_list.append(self.MessagePassingNN[i](cur_node_states, edge))
        #     cur_node_states = sum(out_list)
        #     new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
        #     last_node_states = new_node_states
        # 
        # ggnn_out1 = last_node_states # shape: V, D
        # 
        # # tensor GCN new:
        # assert self.num_edge_types == 4   #4
        # cur_x = torch.cat([ggnn_out1,ggnn_out1,ggnn_out1,ggnn_out1], dim=0)  # 4V, D
        # loop_edge_list = self.matrix_loop(edge_list1) # 4V, 4V
        # out = self.conv1(cur_x, loop_edge_list) # 4V, D
        # out = self.conv2(out, loop_edge_list)  # 4V, D
        # out = out.view( 4, x_embedding.shape[0], out.shape[-1])  # V, 4, D
        # out = torch.sum(out, dim=0)  # V, D
        # n1 = out
  #.............................................................................................

        # x_embedding = x2
        # #x_embedding = self.value_embeddingLayer(x)
        # # Tensor GGNN
        # last_node_states = x_embedding
        # for _ in range(4):
        #     out_list = []
        #     cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
        #     for i in range(len(edge_list2)):
        #         edge = edge_list2[i]
        #         if edge.shape[0] != 0 :
        #             # 该种类型的边存在边
        #             out_list.append(self.MessagePassingNN[i](cur_node_states, edge))
        #     cur_node_states = sum(out_list)
        #     new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
        #     last_node_states = new_node_states
        #
        # ggnn_out2 = last_node_states # shape: V, D
        #
        # # tensor GCN new:
        # assert self.num_edge_types == 4   #4
        # cur_x = torch.cat([ggnn_out2, ggnn_out2, ggnn_out2, ggnn_out2], dim=0)  # 4V, D
        # loop_edge_list = self.matrix_loop(edge_list2) # 4V, 4V
        # out = self.conv1(cur_x, loop_edge_list) # 4V, D
        # out = self.conv2(out, loop_edge_list)  # 4V, D
        # out = out.view(4, x_embedding.shape[0], out.shape[-1])  # V, 4, D
        # out = torch.sum(out, dim=0)  # V, D
        # n2 = out
        #cos = torch.cosine_similarity(n1, n2, dim=0)

        return n1
    '''

    # ............................................................................

    def forward(self, x, edge_list: torch.tensor, **kwargs):
        edge_types = edge_list[2].long()

        A1 = edge_list[:2, edge_types == 0]
        A2 = edge_list[:2, edge_types == 1]
        A3 = edge_list[:2, edge_types == 2]  # 边类型为 2
        A4 = edge_list[:2, edge_types == 3]  # 边类型为 3

        edge_lists = []
        edge_lists.append(A1)
        edge_lists.append(A2)
        edge_lists.append(A3)
        edge_lists.append(A4)

        x_embedding = x
        last_node_states = x_embedding
        # for _ in range(4):
        #     out_list = []
        #     cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
        #     for i in range(len(edge_lists)):
        #         edge = edge_lists[i]
        #
        #         if edge.shape[0] != 0:
        #             # 该种类型的边存在边
        #             out_list.append(self.MessagePassingNN[i](cur_node_states, edge))
        #     cur_node_states = sum(out_list)
        #     new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
        #     last_node_states = new_node_states

        ggnn_out1 = last_node_states  # shape: V, D

        # tensor GCN new:
        assert self.num_edge_types == 4  # 4
        num_nodes = x_embedding.shape[0]
        adjacency_tensor = self.edge_lists_to_adjacency_tensor(edge_lists, num_nodes, x_embedding.device)
        if self.use_t_svd:
            # t-SVD denoising: A -> A_r, then keep sparse top-k edges for GATConv.
            adjacency_tensor = self.t_svd_low_rank(adjacency_tensor, self.t_svd_rank)
        normalized_tensor = self.normalize_adjacency_tensor(adjacency_tensor)
        tensor_edge_lists = self.adjacency_tensor_to_edge_lists(
            normalized_tensor,
            base_edge_lists=edge_lists,
            topk=self.t_svd_topk
        )

        cur_x = torch.cat([ggnn_out1, ggnn_out1, ggnn_out1, ggnn_out1], dim=0)  # 4V, D
        loop_edge_list = self.matrix_loop(tensor_edge_lists, num_nodes=num_nodes)  # 4V, 4V
        out = self.conv1(cur_x, loop_edge_list)  # 4V, D
        out = self.conv2(out, loop_edge_list)  # 4V, D
        out = out.view(4, x_embedding.shape[0], out.shape[-1])  # 4, V, D
        if self.use_cp_attention:
            out = self.cp_structural_attention(out, normalized_tensor)
        out = torch.sum(out, dim=0)  # V, D

        return out

    def edge_lists_to_adjacency_tensor(self, edge_lists, num_nodes, device):
        adjacency_tensor = torch.zeros(
            num_nodes,
            num_nodes,
            self.num_edge_types,
            device=device,
            dtype=torch.float32
        )
        for edge_type, edge in enumerate(edge_lists):
            if edge.numel() == 0:
                continue
            edge = edge.long().to(device)
            valid_mask = (
                (edge[0] >= 0) & (edge[0] < num_nodes) &
                (edge[1] >= 0) & (edge[1] < num_nodes)
            )
            if valid_mask.any():
                src = edge[0, valid_mask]
                dst = edge[1, valid_mask]
                adjacency_tensor[src, dst, edge_type] = 1.0
        return adjacency_tensor

    def t_svd_low_rank(self, adjacency_tensor, rank):
        if rank is None or rank <= 0:
            return adjacency_tensor

        with torch.no_grad():
            rank = min(rank, adjacency_tensor.shape[0], adjacency_tensor.shape[1])
            adjacency_hat = torch.fft.fft(adjacency_tensor, dim=2)
            low_rank_slices = []
            for view_id in range(adjacency_hat.shape[2]):
                u, s, vh = torch.linalg.svd(adjacency_hat[:, :, view_id], full_matrices=False)
                cur_rank = min(rank, s.shape[0])
                low_rank_slice = (u[:, :cur_rank] * s[:cur_rank].unsqueeze(0)) @ vh[:cur_rank, :]
                low_rank_slices.append(low_rank_slice)
            low_rank_hat = torch.stack(low_rank_slices, dim=2)
            low_rank_tensor = torch.fft.ifft(low_rank_hat, dim=2).real
            return low_rank_tensor.clamp_min(0.0)

    def normalize_adjacency_tensor(self, adjacency_tensor):
        adjacency_tensor = adjacency_tensor.clamp_min(0.0)
        if self.add_self_loops:
            eye = torch.eye(
                adjacency_tensor.shape[0],
                device=adjacency_tensor.device,
                dtype=adjacency_tensor.dtype
            ).unsqueeze(-1)
            adjacency_tensor = adjacency_tensor + eye

        degree = adjacency_tensor.sum(dim=1).clamp_min(self.tensor_eps)
        degree_inv_sqrt = degree.pow(-0.5)
        return degree_inv_sqrt.unsqueeze(1) * adjacency_tensor * degree_inv_sqrt.unsqueeze(0)

    def adjacency_tensor_to_edge_lists(self, adjacency_tensor, base_edge_lists=None, topk=None):
        edge_lists = []
        num_nodes = adjacency_tensor.shape[0]
        row_index = torch.arange(num_nodes, device=adjacency_tensor.device)
        if topk is None or topk <= 0:
            topk = num_nodes
        topk = min(topk, num_nodes)

        for edge_type in range(adjacency_tensor.shape[2]):
            adj = adjacency_tensor[:, :, edge_type]
            values, cols = torch.topk(adj, k=topk, dim=1)
            mask = values > self.tensor_eps
            rows = row_index.unsqueeze(1).expand_as(cols)[mask]
            cols = cols[mask]
            edge_index = torch.stack([rows, cols], dim=0) if rows.numel() > 0 else adj.new_empty((2, 0), dtype=torch.long)

            if base_edge_lists is not None and base_edge_lists[edge_type].numel() > 0:
                base_edge = base_edge_lists[edge_type].long().to(adjacency_tensor.device)
                valid_mask = (
                    (base_edge[0] >= 0) & (base_edge[0] < num_nodes) &
                    (base_edge[1] >= 0) & (base_edge[1] < num_nodes)
                )
                if valid_mask.any():
                    edge_index = torch.cat([edge_index, base_edge[:, valid_mask]], dim=1)

            if edge_index.numel() > 0:
                edge_index = torch.unique(edge_index, dim=1)
            edge_lists.append(edge_index)
        return edge_lists

    def khatri_rao(self, left, right):
        return torch.einsum('ir,jr->ijr', left, right).reshape(left.shape[0] * right.shape[0], -1)

    def cp_decomposition(self, tensor, rank=None, n_iter=None):
        with torch.no_grad():
            tensor = tensor.detach().clamp_min(0.0)
            dim_i, dim_j, dim_t = tensor.shape
            rank = min(rank or self.cp_rank, dim_i, dim_j)
            n_iter = n_iter or self.cp_iters
            generator_device = tensor.device if tensor.device.type == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(0)
            o_factor = torch.rand(dim_i, rank, device=tensor.device, generator=generator).clamp_min(self.tensor_eps)
            p_factor = torch.rand(dim_j, rank, device=tensor.device, generator=generator).clamp_min(self.tensor_eps)
            q_factor = torch.rand(dim_t, rank, device=tensor.device, generator=generator).clamp_min(self.tensor_eps)
            eye = torch.eye(rank, device=tensor.device, dtype=tensor.dtype)

            for _ in range(n_iter):
                kr = self.khatri_rao(p_factor, q_factor)
                gram = kr.T @ kr + self.tensor_eps * eye
                o_factor = (tensor.reshape(dim_i, dim_j * dim_t) @ kr) @ torch.linalg.pinv(gram)

                kr = self.khatri_rao(o_factor, q_factor)
                gram = kr.T @ kr + self.tensor_eps * eye
                p_factor = (tensor.permute(1, 0, 2).reshape(dim_j, dim_i * dim_t) @ kr) @ torch.linalg.pinv(gram)

                kr = self.khatri_rao(o_factor, p_factor)
                gram = kr.T @ kr + self.tensor_eps * eye
                q_factor = (tensor.permute(2, 0, 1).reshape(dim_t, dim_i * dim_j) @ kr) @ torch.linalg.pinv(gram)

                o_factor = F.normalize(o_factor.clamp_min(self.tensor_eps), p=2, dim=0)
                p_factor = F.normalize(p_factor.clamp_min(self.tensor_eps), p=2, dim=0)
                q_factor = F.normalize(q_factor.clamp_min(self.tensor_eps), p=2, dim=0)

            return o_factor, p_factor, q_factor

    def cp_structural_attention(self, view_features, adjacency_tensor):
        if view_features.shape[0] != self.num_edge_types:
            raise ValueError("view_features should have shape [num_edge_types, num_nodes, hidden_dim].")

        o_factor, p_factor, q_factor = self.cp_decomposition(adjacency_tensor, self.cp_rank, self.cp_iters)
        structural_scores = torch.einsum('ir,jr,tr->ijt', o_factor, p_factor, q_factor)
        edge_mask = adjacency_tensor > self.tensor_eps
        eye_mask = torch.eye(
            adjacency_tensor.shape[0],
            device=adjacency_tensor.device,
            dtype=torch.bool
        ).unsqueeze(-1)
        edge_mask = edge_mask | eye_mask

        structural_scores = structural_scores.masked_fill(~edge_mask, -1e9)
        structural_attention = torch.softmax(structural_scores, dim=1)
        structural_attention = torch.nan_to_num(structural_attention, nan=0.0)
        attended_features = torch.einsum('ijt,tjd->tid', structural_attention, view_features)
        return 0.5 * (view_features + attended_features)

    def matrix_transfer(self, edge, i, j):
        # edge: [[i],[j]]
        edge_new = edge.detach().clone()
        edge_new[0] += i
        edge_new[1] += j
        return edge_new

    def matrix_loop(self, edge_list, num_nodes=None):
        # 4个邻接矩阵并列。
        # assert len(edge_list) == 4

        assert len(edge_list) == 4
        A1, A2, A3, A4 = edge_list

        n = num_nodes or self.max_node_per_graph
        loop_edge_list = []
        loop_edge_list.append(A1)
        loop_edge_list.append(self.matrix_transfer(A2, n, 0))
        loop_edge_list.append(self.matrix_transfer(A3, 2 * n, 0))
        loop_edge_list.append(self.matrix_transfer(A4, 3 * n, 0))

        loop_edge_list.append(self.matrix_transfer(A4, 0, n))
        loop_edge_list.append(self.matrix_transfer(A1, n, n))
        loop_edge_list.append(self.matrix_transfer(A2, 2 * n, n))
        loop_edge_list.append(self.matrix_transfer(A3, 3 * n, n))

        loop_edge_list.append(self.matrix_transfer(A3, 0, 2 * n))
        loop_edge_list.append(self.matrix_transfer(A4, n, 2 * n))
        loop_edge_list.append(self.matrix_transfer(A1, 2 * n, 2 * n))
        loop_edge_list.append(self.matrix_transfer(A2, 3 * n, 2 * n))

        loop_edge_list.append(self.matrix_transfer(A2, 0, 3 * n))
        loop_edge_list.append(self.matrix_transfer(A3, n, 3 * n))
        loop_edge_list.append(self.matrix_transfer(A4, 2 * n, 3 * n))
        loop_edge_list.append(self.matrix_transfer(A1, 3 * n, 3 * n))



        return torch.cat(loop_edge_list, dim=1)

    # def variable_detection(self, out):
    #
    #
    #     return self.out
