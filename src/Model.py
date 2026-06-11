import math
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


def _import_dgl():
    if "dgl.graphbolt" not in sys.modules:
        graphbolt = types.ModuleType("dgl.graphbolt")
        graphbolt.__all__ = []
        sys.modules["dgl.graphbolt"] = graphbolt
    try:
        import dgl
    except ImportError as exc:
        raise ImportError(
            "DGL is required by MCCE-MHGCN. Install a DGL build compatible "
            "with your PyTorch/CUDA version, for example the build matching "
            "PyTorch 2.2.3+cu121 or 2.5.0+cu121."
        ) from exc
    return dgl


def feature_to_tensor(feature, device):
    if isinstance(feature, torch.Tensor):
        return feature.float().to(device)
    try:
        feature = feature.astype(float).toarray()
    except Exception:
        try:
            feature = feature.toarray()
        except Exception:
            pass
    return torch.as_tensor(feature, dtype=torch.float32, device=device)


def _layer_type(index):
    return "L{}".format(index)


def _etype_name(source, target):
    return "r_{}_{}".format(source, target)


def _mp_etype_name(metapath, start_position):
    return "mp_{}_from_{}".format("_".join(str(item) for item in metapath), start_position)


def binary_scipy_matrix(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    if matrix.nnz > 0:
        matrix.data[:] = 1.0
    return matrix


def limit_neighbors(matrix, max_neighbors=None):
    if max_neighbors is None or max_neighbors <= 0:
        return matrix
    matrix = matrix.tocsr().astype(np.float32)
    rows, cols, values = [], [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_cols = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if row_cols.size > max_neighbors:
            selected = np.argpartition(-row_values, max_neighbors - 1)[:max_neighbors]
            row_cols = row_cols[selected]
            row_values = row_values[selected]
        rows.extend([row] * row_cols.size)
        cols.extend(row_cols.tolist())
        values.extend(row_values.tolist())
    return sparse.csr_matrix((np.asarray(values, dtype=np.float32), (np.asarray(rows), np.asarray(cols))), shape=matrix.shape)


def normalize_adjacency(matrix, add_self_loop=True):
    matrix = matrix.astype(np.float32).tocsr()
    if add_self_loop and matrix.shape[0] == matrix.shape[1]:
        matrix = matrix + sparse.eye(matrix.shape[0], dtype=np.float32, format="csr")
    rowsum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    rowsum[rowsum == 0.0] = 1.0
    inv_sqrt = np.power(rowsum, -0.5)
    degree = sparse.diags(inv_sqrt.astype(np.float32), format="csr")
    if matrix.shape[0] == matrix.shape[1]:
        return (degree @ matrix @ degree).astype(np.float32).tocsr()
    return matrix


def _matrix_to_edges(matrix, weighted=False):
    matrix = matrix.tocoo().astype(np.float32)
    src = torch.from_numpy(matrix.col.astype(np.int64))
    dst = torch.from_numpy(matrix.row.astype(np.int64))
    if weighted:
        return src, dst, torch.from_numpy(matrix.data.astype(np.float32))
    return src, dst


def build_dgl_heterograph(intra_adj, cross_adj, num_layers, weighted_intra=False):
    dgl = _import_dgl()
    data_dict = {}
    weights = {}
    num_nodes_dict = {}
    for layer in range(num_layers):
        matrix = intra_adj[layer]
        num_nodes_dict[_layer_type(layer)] = int(matrix.shape[0])
        etype = (_layer_type(layer), _etype_name(layer, layer), _layer_type(layer))
        if weighted_intra:
            src, dst, weight = _matrix_to_edges(matrix, weighted=True)
            weights[etype] = weight
        else:
            src, dst = _matrix_to_edges(binary_scipy_matrix(matrix))
        data_dict[etype] = (src, dst)
    for target in range(num_layers):
        for source in range(num_layers):
            if source == target:
                continue
            matrix = cross_adj[target][source]
            if matrix is None or getattr(matrix, "size", 1) == 0:
                continue
            src, dst = _matrix_to_edges(binary_scipy_matrix(matrix))
            data_dict[(_layer_type(source), _etype_name(source, target), _layer_type(target))] = (src, dst)
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    for etype, weight in weights.items():
        graph.edges[etype].data["norm"] = weight
    return graph


def dgl_metapath_reachable_matrix(dgl_graph, metapath, start_position, sample_size=None):
    dgl = _import_dgl()
    suffix = metapath[start_position:]
    etypes = [_etype_name(suffix[i], suffix[i + 1]) for i in range(len(suffix) - 1)]
    reachable = dgl.metapath_reachable_graph(dgl_graph, etypes)
    source_nodes, target_nodes = reachable.edges()
    source_type = suffix[0]
    target_type = suffix[-1]
    matrix = sparse.csr_matrix(
        (
            np.ones(source_nodes.numel(), dtype=np.float32),
            (target_nodes.cpu().numpy().astype(np.int64), source_nodes.cpu().numpy().astype(np.int64)),
        ),
        shape=(dgl_graph.num_nodes(_layer_type(target_type)), dgl_graph.num_nodes(_layer_type(source_type))),
    )
    return limit_neighbors(binary_scipy_matrix(matrix), sample_size)


class DGLGraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, graph, node_type, etype, features):
        import dgl.function as fn
        h = self.linear(features.float())
        with graph.local_scope():
            graph.nodes[node_type].data["h"] = h
            if "norm" in graph.edges[etype].data:
                graph.update_all(fn.u_mul_e("h", "norm", "m"), fn.sum("m", "out"), etype=etype)
            else:
                graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "out"), etype=etype)
            return graph.nodes[node_type].data["out"]


class MECCHMetapathFusion(nn.Module):
    def __init__(self, n_metapaths, in_dim, out_dim, fusion_type="conv"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "mean":
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "weight":
            self.weight = nn.Parameter(torch.full((n_metapaths,), 1.0 / n_metapaths))
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "conv":
            self.conv = nn.Parameter(torch.full((n_metapaths, in_dim), 1.0 / n_metapaths))
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "cat":
            self.linear = nn.Linear(n_metapaths * in_dim, out_dim)
        else:
            raise ValueError("Unknown metapath_fusion '{}'".format(fusion_type))

    def forward(self, h_list):
        if self.fusion_type == "mean":
            fused = torch.mean(torch.stack(h_list), dim=0)
        elif self.fusion_type == "weight":
            fused = torch.sum(torch.stack(h_list) * self.weight[:, None, None], dim=0)
        elif self.fusion_type == "conv":
            fused = torch.sum(torch.stack(h_list).transpose(0, 1) * self.conv, dim=1)
        else:
            fused = torch.hstack(h_list)
        return self.linear(fused), fused


class MetapathContextEncoder(nn.Module):
    def __init__(self, in_dim, encoder_type="mean", use_v=False, n_heads=8):
        super().__init__()
        if in_dim % n_heads != 0:
            raise ValueError("in_dim must be divisible by n_heads")
        self.encoder_type = encoder_type
        self.use_v = use_v
        self.n_heads = n_heads
        self.d_k = in_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        if encoder_type == "attention":
            self.q_linear = nn.Linear(in_dim, in_dim, bias=False)
            self.k_linear = nn.Linear(in_dim, in_dim, bias=False)
            if use_v:
                self.v_linear = nn.Linear(in_dim, in_dim, bias=False)
        elif encoder_type != "mean":
            raise ValueError("Unknown context encoder '{}'".format(encoder_type))

    def forward(self, graph, target_type, target_embedding, embeddings, suffixes):
        if self.encoder_type == "attention":
            return self._attention_forward(graph, target_type, target_embedding, embeddings, suffixes)
        return self._mean_forward(graph, target_type, target_embedding, embeddings, suffixes)

    def _mean_forward(self, graph, target_type, target_embedding, embeddings, suffixes):
        import dgl.function as fn
        funcs = {}
        degree_sum = target_embedding.new_zeros(target_embedding.size(0))
        with graph.local_scope():
            for source_type, etype in suffixes:
                graph.nodes[_layer_type(source_type)].data["h_src"] = embeddings[source_type]
                funcs[etype] = (fn.copy_u("h_src", "m"), fn.sum("m", "h_neigh"))
                degree_sum = degree_sum + graph.in_degrees(etype=etype).to(target_embedding.device).float()
            if funcs:
                graph.multi_update_all(funcs, "sum")
                message_sum = graph.nodes[target_type].data.get("h_neigh", target_embedding.new_zeros(target_embedding.size()))
            else:
                message_sum = target_embedding.new_zeros(target_embedding.size())
        return (message_sum + target_embedding) / (degree_sum.unsqueeze(-1) + 1.0).clamp_min(1.0)

    def _attention_forward(self, graph, target_type, target_embedding, embeddings, suffixes):
        import dgl
        import dgl.function as fn
        from dgl.nn.functional import edge_softmax
        if not suffixes:
            return target_embedding
        n_target = target_embedding.size(0)
        with graph.local_scope():
            graph.nodes[target_type].data["q"] = self.q_linear(target_embedding).view(n_target, self.n_heads, self.d_k)
            target_key = self.k_linear(target_embedding).view(n_target, self.n_heads, self.d_k)
            target_value = self.v_linear(target_embedding).view(n_target, self.n_heads, self.d_k) if self.use_v else target_embedding.view(n_target, self.n_heads, self.d_k)
            etypes = []
            for source_type, etype in suffixes:
                ntype = _layer_type(source_type)
                source_embedding = embeddings[source_type]
                graph.nodes[ntype].data["k"] = self.k_linear(source_embedding).view(-1, self.n_heads, self.d_k)
                graph.nodes[ntype].data["v"] = self.v_linear(source_embedding).view(-1, self.n_heads, self.d_k) if self.use_v else source_embedding.view(-1, self.n_heads, self.d_k)
                graph.apply_edges(fn.u_dot_v("k", "q", "score"), etype=etype)
                graph.edges[etype].data["score"] = graph.edges[etype].data["score"] / self.sqrt_dk
                etypes.append(etype)
            sub_graph = dgl.edge_type_subgraph(graph, etypes=etypes)
            homo_graph = dgl.to_homogeneous(sub_graph, edata=["score"])
            offset_node = sum(sub_graph.num_nodes(sub_graph.ntypes[i]) for i in range(sub_graph.get_ntype_id(target_type)))
            offset_edge = homo_graph.num_edges()
            target_nodes = homo_graph.nodes()[offset_node:offset_node + n_target]
            self_score = torch.sum(target_key * graph.nodes[target_type].data["q"], dim=-1, keepdim=True) / self.sqrt_dk
            homo_graph.add_edges(target_nodes, target_nodes, data={"score": self_score})
            homo_graph.edata["alpha"] = edge_softmax(homo_graph, homo_graph.edata["score"], norm_by="dst")
            self_alpha = homo_graph.edata["alpha"][-n_target:]
            homo_graph.remove_edges(offset_edge + torch.arange(n_target, device=homo_graph.device))
            restored = dgl.to_heterogeneous(homo_graph, sub_graph.ntypes, sub_graph.etypes)
            for etype in restored.canonical_etypes:
                graph.edges[etype].data["alpha"] = restored.edges[etype].data["alpha"]
            funcs = {etype: (fn.u_mul_e("v", "alpha", "m"), fn.sum("m", "h_neigh")) for _, etype in suffixes}
            graph.multi_update_all(funcs, "sum")
            message_sum = graph.nodes[target_type].data.get("h_neigh", target_embedding.new_zeros((n_target, self.n_heads, self.d_k)))
            return (message_sum + self_alpha * target_value).reshape(n_target, -1)


class MCCE_MHGCN(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_layers, gnn_layers=2, cross_order=1, intra_sample_size=None,
                 dropout=0.5, use_gate=True, metapaths=None, metapath_fusion="conv", context_encoder="mean",
                 context_use_v=False, context_heads=8, context_sample_size=None, metapath_length=3, number_layers=1,
                 fusion_mode="both", context_model="mecch", **kwargs):
        super().__init__()
        if isinstance(input_dims, int):
            input_dims = [input_dims for _ in range(num_layers)]
        if len(input_dims) != num_layers:
            raise ValueError("input_dims must match num_layers")
        if gnn_layers < 1:
            raise ValueError("gnn_layers must be >= 1")
        if not metapaths:
            raise ValueError("MCCE-MHGCN requires metapaths")
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_layers = gnn_layers
        self.intra_sample_size = intra_sample_size
        self.dropout = dropout
        self.use_gate = use_gate
        self.metapaths = metapaths
        self.number_layers = number_layers
        self.context_sample_size = context_sample_size
        self.fusion_mode = fusion_mode.lower()
        self.context_model = context_model.lower()
        if self.context_model != "mecch":
            raise ValueError("This DGL implementation currently supports --context-model mecch")
        self.input_projectors = nn.ModuleList([nn.Linear(dim, hidden_dim) for dim in input_dims])
        self.intra_gcn = nn.ModuleList(nn.ModuleList([DGLGraphConvolution(hidden_dim, hidden_dim) for _ in range(gnn_layers)]) for _ in range(num_layers))
        self.context_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(number_layers)])
        self.context_encoders = nn.ModuleList()
        self.metapath_fuse = nn.ModuleList()
        for _ in range(number_layers):
            encoders = nn.ModuleDict()
            fusers = nn.ModuleDict()
            for target in range(num_layers):
                target_metapaths = [(idx, path) for idx, path in enumerate(metapaths) if path[-1] == target]
                for idx, _path in target_metapaths:
                    encoders["{}_{}".format(target, idx)] = MetapathContextEncoder(hidden_dim, context_encoder, context_use_v, context_heads)
                if target_metapaths:
                    fusers[str(target)] = MECCHMetapathFusion(len(target_metapaths), hidden_dim, hidden_dim, metapath_fusion)
            self.context_encoders.append(encoders)
            self.metapath_fuse.append(fusers)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self._context_cache = {}

    def _build_base_graph(self, intra_adj, cross_adj, device):
        normalized = [normalize_adjacency(limit_neighbors(matrix, self.intra_sample_size), add_self_loop=True) for matrix in intra_adj]
        return build_dgl_heterograph(normalized, cross_adj, self.num_layers, weighted_intra=True).to(device)

    def _encode_intra_layer(self, graph, features, layer):
        device = self.input_projectors[layer].weight.device
        h = F.relu(self.input_projectors[layer](feature_to_tensor(features, device)))
        node_type = _layer_type(layer)
        etype = _etype_name(layer, layer)
        for gcn in self.intra_gcn[layer]:
            h = F.relu(gcn(graph, node_type, etype, h))
            h = F.dropout(h, self.dropout, training=self.training)
        return h

    def _cache_key(self, intra_adj, cross_adj, device):
        intra_sig = tuple((matrix.shape, int(matrix.nnz)) for matrix in intra_adj)
        cross_sig = []
        for target in range(self.num_layers):
            row = []
            for source in range(self.num_layers):
                matrix = cross_adj[target][source]
                row.append(None if matrix is None else (matrix.shape, int(matrix.nnz)))
            cross_sig.append(tuple(row))
        return (str(device), tuple(self.metapaths), intra_sig, tuple(cross_sig))

    def _prepare_context_graph(self, intra_adj, cross_adj, device):
        key = self._cache_key(intra_adj, cross_adj, device)
        if key in self._context_cache:
            return self._context_cache[key]
        dgl = _import_dgl()
        base = build_dgl_heterograph([binary_scipy_matrix(limit_neighbors(m, self.intra_sample_size)) for m in intra_adj], cross_adj, self.num_layers)
        data_dict = {}
        num_nodes_dict = {_layer_type(layer): int(intra_adj[layer].shape[0]) for layer in range(self.num_layers)}
        specs = {target: [] for target in range(self.num_layers)}
        for idx, path in enumerate(self.metapaths):
            target = path[-1]
            suffixes = []
            valid = True
            for start in range(0, len(path) - 1):
                try:
                    matrix = dgl_metapath_reachable_matrix(base, path, start, self.context_sample_size)
                except Exception:
                    valid = False
                    break
                source_type = path[start]
                etype = _mp_etype_name(path, start)
                src, dst = _matrix_to_edges(matrix)
                data_dict[(_layer_type(source_type), etype, _layer_type(target))] = (src, dst)
                suffixes.append((source_type, etype))
            if valid:
                specs[target].append((idx, path, suffixes))
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict).to(device)
        self._context_cache[key] = (graph, specs)
        return graph, specs

    def _metapath_cross_embedding(self, embeddings, intra_adj, cross_adj, target, layer_idx):
        device = embeddings[target].device
        graph, specs = self._prepare_context_graph(intra_adj, cross_adj, device)
        contexts = []
        for idx, _path, suffixes in specs[target]:
            encoder = self.context_encoders[layer_idx]["{}_{}".format(target, idx)]
            contexts.append(encoder(graph, _layer_type(target), embeddings[target], embeddings, suffixes))
        if not contexts:
            return embeddings[target].new_zeros(embeddings[target].size())
        projected, _ = self.metapath_fuse[layer_idx][str(target)](contexts)
        return self.context_norms[layer_idx](projected)

    def forward(self, features_by_layer, intra_adj, cross_adj):
        device = self.input_projectors[0].weight.device
        base_graph = self._build_base_graph(intra_adj, cross_adj, device)
        structural = [self._encode_intra_layer(base_graph, features_by_layer[layer], layer) for layer in range(self.num_layers)]
        if self.fusion_mode == "intra":
            return [F.dropout(h, self.dropout, training=self.training) for h in structural]
        context_input = structural
        cross = None
        for layer_idx in range(self.number_layers):
            cross = [self._metapath_cross_embedding(context_input, intra_adj, cross_adj, target, layer_idx) for target in range(self.num_layers)]
            if layer_idx < self.number_layers - 1:
                context_input = [F.dropout(F.relu(h), self.dropout, training=self.training) for h in cross]
        if self.fusion_mode == "context":
            return [F.dropout(h, self.dropout, training=self.training) for h in cross]
        output = []
        for target, cross_embedding in enumerate(cross):
            combined = torch.cat((structural[target], cross_embedding), dim=1)
            if self.use_gate:
                gate = torch.sigmoid(self.gate(combined))
                fused = gate * structural[target] + (1.0 - gate) * cross_embedding
            else:
                fused = F.relu(self.fusion(combined))
            output.append(F.dropout(fused, self.dropout, training=self.training))
        return output


class MCCE_MHGCNLinkPredictor(nn.Module):
    def __init__(self, encoder, target_layer=0, source_layer=None, link_task="intra", predictor="distmult",
                 hidden_dim=None, predictor_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.target_layer = target_layer
        self.source_layer = target_layer if source_layer is None else source_layer
        self.link_task = link_task.lower()
        self.predictor = predictor.lower()
        if self.predictor not in ("distmult", "dot", "mlp"):
            raise ValueError("predictor must be distmult, dot, or mlp")
        hidden_dim = hidden_dim or encoder.hidden_dim
        if self.predictor == "distmult":
            self.relation = nn.Parameter(torch.ones(hidden_dim))
        if self.predictor == "mlp":
            predictor_hidden_dim = predictor_hidden_dim or hidden_dim
            self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 4, predictor_hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(predictor_hidden_dim, 1))

    def forward(self, features_by_layer, intra_adj, cross_adj, edge_index, target_layer=None, source_layer=None):
        target = self.target_layer if target_layer is None else target_layer
        source = self.source_layer if source_layer is None else source_layer
        if self.link_task == "intra":
            source = target
        embeddings = self.encoder(features_by_layer, intra_adj, cross_adj)
        z_source = embeddings[source]
        z_target = embeddings[target]
        src, dst = edge_index
        if src.numel() > 0:
            if int(src.max().detach().cpu()) >= z_source.size(0):
                raise ValueError("edge_index source id exceeds source-layer node count")
            if int(dst.max().detach().cpu()) >= z_target.size(0):
                raise ValueError("edge_index target id exceeds target-layer node count")
        src_z = z_source[src]
        dst_z = z_target[dst]
        if self.predictor == "mlp":
            return self.edge_mlp(torch.cat((src_z, dst_z, src_z * dst_z, torch.abs(src_z - dst_z)), dim=1)).squeeze(-1)
        if self.predictor == "distmult":
            return torch.sum(src_z * self.relation * dst_z, dim=1)
        return torch.sum(F.normalize(src_z, p=2, dim=1) * F.normalize(dst_z, p=2, dim=1), dim=1)
