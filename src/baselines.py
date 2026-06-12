import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_dgl():
    if "dgl.graphbolt" not in sys.modules:
        graphbolt = types.ModuleType("dgl.graphbolt")
        graphbolt.__all__ = []
        sys.modules["dgl.graphbolt"] = graphbolt
    try:
        import dgl
        import dgl.function as fn
        import dgl.nn as dglnn
        from dgl.nn.functional import edge_softmax
    except ImportError as exc:
        raise ImportError("DGL >= 2.1.0 is required with PyTorch 2.3.0.") from exc
    return dgl, fn, dglnn, edge_softmax


def _metapath_key(metapath):
    return "||".join("__".join(etype) for etype in metapath)


def _relation_names(metapath):
    return [etype[1] for etype in metapath]


class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, z):
        weights = self.project(z).mean(0)
        weights = torch.softmax(weights, dim=0)
        return (weights.unsqueeze(0) * z).sum(1)


class HANTypeLayer(nn.Module):
    def __init__(self, metapaths, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for HAN")
        _dgl, _fn, dglnn, _edge_softmax = _import_dgl()
        self.metapaths = list(metapaths)
        self.hidden_dim = hidden_dim
        self.gat_layers = nn.ModuleDict({
            _metapath_key(mp): dglnn.GATConv(
                hidden_dim,
                hidden_dim // num_heads,
                num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.elu,
                allow_zero_in_degree=True,
            )
            for mp in self.metapaths
        })
        self.semantic_attention = SemanticAttention(hidden_dim)
        self._graph_cache = {}

    def _reachable_graph(self, graph, metapath):
        dgl, _fn, _dglnn, _edge_softmax = _import_dgl()
        key = (id(graph), _metapath_key(metapath))
        if key not in self._graph_cache:
            self._graph_cache[key] = dgl.metapath_reachable_graph(graph, _relation_names(metapath)).to(graph.device)
        return self._graph_cache[key]

    def forward(self, graph, features):
        semantic_embeddings = []
        for mp in self.metapaths:
            reachable_graph = self._reachable_graph(graph, mp)
            h = self.gat_layers[_metapath_key(mp)](reachable_graph, features).flatten(1)
            semantic_embeddings.append(h)
        return self.semantic_attention(torch.stack(semantic_embeddings, dim=1))


class HANEncoder(nn.Module):
    """HAN baseline adapted to the repository link-prediction encoder interface."""

    def __init__(self, graph, input_dims, hidden_dim, metapaths, num_heads=4, dropout=0.5):
        super().__init__()
        if not metapaths:
            raise ValueError("HAN requires at least one metapath")
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        grouped = {ntype: [] for ntype in self.ntypes}
        for mp in metapaths:
            if mp[0][0] == mp[-1][2]:
                grouped[mp[-1][2]].append(mp)
        self.layers = nn.ModuleDict({
            ntype: HANTypeLayer(paths, hidden_dim, num_heads, dropout)
            for ntype, paths in grouped.items()
            if paths
        })
        if not self.layers:
            raise ValueError("HAN only supports closed metapaths such as A-P-A or A-V-A")
        self.dropout = dropout

    def forward(self, graph, features):
        projected = {
            ntype: F.dropout(F.relu(self.projectors[ntype](features[ntype].float())), self.dropout, training=self.training)
            for ntype in self.ntypes
        }
        output = dict(projected)
        for ntype, layer in self.layers.items():
            output[ntype] = F.dropout(layer(graph, projected[ntype]), self.dropout, training=self.training)
        return output


class MAGNNMetapathLayer(nn.Module):
    """MAGNN-style metapath-specific encoder for DGL reachable graphs.

    HGB MAGNN uses explicit metapath instances. A DGL metapath-reachable graph
    only stores the endpoint pairs, so this layer encodes each reachable edge as
    a short endpoint-plus-relation sequence and then performs node-level
    attention. This keeps the MAGNN ingredients while fitting the bin graph path.
    """

    def __init__(self, metapath, hidden_dim, num_heads, dropout, rnn_type="gru", alpha=0.01):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for MAGNN")
        self.metapath = metapath
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.relation_embeddings = nn.Parameter(torch.empty(len(metapath), hidden_dim))
        if rnn_type == "gru":
            self.sequence_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif rnn_type == "lstm":
            self.sequence_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        elif rnn_type == "linear":
            self.sequence_encoder = nn.Linear(hidden_dim, hidden_dim)
        elif rnn_type != "average":
            raise ValueError("MAGNN supports --magnn-rnn-type gru, lstm, linear, or average")
        self.attn = nn.Parameter(torch.empty(num_heads, hidden_dim // num_heads))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.attn_drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.relation_embeddings, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)
        if isinstance(self.sequence_encoder, nn.Linear):
            nn.init.xavier_normal_(self.sequence_encoder.weight, gain=1.414)

    def _encode_sequence(self, src_h, dst_h):
        relation_h = self.relation_embeddings.unsqueeze(0).expand(src_h.shape[0], -1, -1)
        sequence = torch.cat([src_h.unsqueeze(1), relation_h, dst_h.unsqueeze(1)], dim=1)
        if self.rnn_type == "gru":
            _out, hidden = self.sequence_encoder(sequence)
            return hidden[-1]
        if self.rnn_type == "lstm":
            _out, (hidden, _cell) = self.sequence_encoder(sequence)
            return hidden[-1]
        if self.rnn_type == "linear":
            return self.sequence_encoder(sequence.mean(dim=1))
        return sequence.mean(dim=1)

    def forward(self, reachable_graph, features):
        _dgl, fn, _dglnn, edge_softmax = _import_dgl()
        with reachable_graph.local_scope():
            reachable_graph.ndata["h"] = features

            def edge_encoder(edges):
                edge_h = self._encode_sequence(edges.src["h"], edges.dst["h"])
                edge_h = edge_h.view(-1, self.num_heads, self.hidden_dim // self.num_heads)
                score = (edge_h * self.attn.unsqueeze(0)).sum(dim=-1, keepdim=True)
                return {"edge_h": edge_h, "score": self.leaky_relu(score)}

            reachable_graph.apply_edges(edge_encoder)
            reachable_graph.edata["alpha"] = self.attn_drop(edge_softmax(reachable_graph, reachable_graph.edata["score"], norm_by="dst"))
            reachable_graph.update_all(fn.u_mul_e("h_head", "alpha", "unused"), fn.sum("unused", "unused_out")) if False else None
            reachable_graph.update_all(
                lambda edges: {"m": edges.data["edge_h"] * edges.data["alpha"]},
                fn.sum("m", "out"),
            )
            if "out" in reachable_graph.ndata:
                return reachable_graph.ndata["out"].reshape(features.shape[0], self.hidden_dim)
            return features.new_zeros(features.shape)


class MAGNNTypeLayer(nn.Module):
    def __init__(self, metapaths, hidden_dim, num_heads, dropout, rnn_type="gru"):
        super().__init__()
        self.metapaths = list(metapaths)
        self.layers = nn.ModuleDict({
            _metapath_key(mp): MAGNNMetapathLayer(mp, hidden_dim, num_heads, dropout, rnn_type=rnn_type)
            for mp in self.metapaths
        })
        self.semantic_attention = SemanticAttention(hidden_dim)
        self._graph_cache = {}

    def _reachable_graph(self, graph, metapath):
        dgl, _fn, _dglnn, _edge_softmax = _import_dgl()
        key = (id(graph), _metapath_key(metapath))
        if key not in self._graph_cache:
            self._graph_cache[key] = dgl.metapath_reachable_graph(graph, _relation_names(metapath)).to(graph.device)
        return self._graph_cache[key]

    def forward(self, graph, features):
        outs = []
        for mp in self.metapaths:
            reachable_graph = self._reachable_graph(graph, mp)
            outs.append(F.elu(self.layers[_metapath_key(mp)](reachable_graph, features)))
        return self.semantic_attention(torch.stack(outs, dim=1))


class MAGNNEncoder(nn.Module):
    """MAGNN baseline adapted from HGB MAGNN to the DGL .bin link-prediction flow."""

    def __init__(self, graph, input_dims, hidden_dim, metapaths, num_heads=4, dropout=0.5, rnn_type="gru"):
        super().__init__()
        if not metapaths:
            raise ValueError("MAGNN requires at least one metapath")
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        grouped = {ntype: [] for ntype in self.ntypes}
        for mp in metapaths:
            if mp[0][0] == mp[-1][2]:
                grouped[mp[-1][2]].append(mp)
        self.layers = nn.ModuleDict({
            ntype: MAGNNTypeLayer(paths, hidden_dim, num_heads, dropout, rnn_type=rnn_type)
            for ntype, paths in grouped.items()
            if paths
        })
        if not self.layers:
            raise ValueError("MAGNN requires closed metapaths for at least one node type")
        self.dropout = dropout

    def forward(self, graph, features):
        projected = {
            ntype: F.dropout(F.relu(self.projectors[ntype](features[ntype].float())), self.dropout, training=self.training)
            for ntype in self.ntypes
        }
        output = dict(projected)
        for ntype, layer in self.layers.items():
            output[ntype] = F.dropout(layer(graph, projected[ntype]), self.dropout, training=self.training)
        return output


class HGTEncoder(nn.Module):
    """HGT baseline using DGL HGTConv on the homogeneous view of the heterograph."""

    def __init__(self, graph, input_dims, hidden_dim, num_layers=2, num_heads=4, dropout=0.5):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for HGT")
        dgl, _fn, dglnn, _edge_softmax = _import_dgl()
        self.dgl = dgl
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        self.layers = nn.ModuleList([
            dglnn.HGTConv(
                hidden_dim,
                hidden_dim // num_heads,
                num_heads,
                len(graph.ntypes),
                len(graph.canonical_etypes),
                dropout=dropout,
                use_norm=True,
            )
            for _ in range(num_layers)
        ])
        self.dropout = dropout

    def _to_homogeneous(self, graph, h_dict):
        with graph.local_scope():
            for ntype, h in h_dict.items():
                graph.nodes[ntype].data["h_hgt"] = h
            return self.dgl.to_homogeneous(graph, ndata=["h_hgt"])

    def _split_by_type(self, graph, homogeneous_graph, h):
        out = {}
        ntype_ids = homogeneous_graph.ndata[self.dgl.NTYPE]
        original_ids = homogeneous_graph.ndata[self.dgl.NID]
        for type_id, ntype in enumerate(self.ntypes):
            mask = ntype_ids == type_id
            values = h[mask]
            ids = original_ids[mask].long()
            typed = h.new_zeros((graph.num_nodes(ntype), h.shape[-1]))
            if values.numel() > 0:
                typed[ids] = values
            out[ntype] = typed
        return out

    def forward(self, graph, features):
        h_dict = {
            ntype: F.dropout(F.relu(self.projectors[ntype](features[ntype].float())), self.dropout, training=self.training)
            for ntype in self.ntypes
        }
        for layer in self.layers:
            homogeneous_graph = self._to_homogeneous(graph, h_dict)
            h = layer(
                homogeneous_graph,
                homogeneous_graph.ndata["h_hgt"],
                homogeneous_graph.ndata[self.dgl.NTYPE],
                homogeneous_graph.edata[self.dgl.ETYPE],
            )
            h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_dict = self._split_by_type(graph, homogeneous_graph, h)
        return h_dict


class HetGNNLayer(nn.Module):
    """DGL HetGNN-style heterogeneous neighbor aggregation layer."""

    def __init__(self, graph, hidden_dim, dropout):
        super().__init__()
        self.ntypes = list(graph.ntypes)
        self.hidden_dim = hidden_dim
        self.type_attn = nn.ModuleDict({ntype: SemanticAttention(hidden_dim) for ntype in self.ntypes})
        self.self_transforms = nn.ModuleDict({ntype: nn.Linear(hidden_dim, hidden_dim) for ntype in self.ntypes})
        self.dropout = dropout

    def _relation_mean(self, graph, etype, h_dict):
        _dgl, fn, _dglnn, _edge_softmax = _import_dgl()
        src_type, _rel, dst_type = etype
        with graph.local_scope():
            graph.nodes[src_type].data["h_hetgnn_src"] = h_dict[src_type]
            graph.update_all(fn.copy_u("h_hetgnn_src", "m"), fn.mean("m", "h_hetgnn_neigh"), etype=etype)
            if "h_hetgnn_neigh" in graph.nodes[dst_type].data:
                return graph.nodes[dst_type].data["h_hetgnn_neigh"]
            return h_dict[dst_type].new_zeros(h_dict[dst_type].shape)

    def forward(self, graph, h_dict):
        contexts = {ntype: [self.self_transforms[ntype](h_dict[ntype])] for ntype in self.ntypes}
        buckets = {}
        for etype in graph.canonical_etypes:
            src_type, _rel, dst_type = etype
            buckets.setdefault((dst_type, src_type), []).append(self._relation_mean(graph, etype, h_dict))
        for (dst_type, _src_type), values in buckets.items():
            contexts[dst_type].append(torch.stack(values, dim=0).mean(dim=0))
        output = {}
        for ntype in self.ntypes:
            stacked = torch.stack(contexts[ntype], dim=1)
            output[ntype] = F.dropout(F.elu(self.type_attn[ntype](stacked)), self.dropout, training=self.training)
        return output


class HetGNNEncoder(nn.Module):
    """HetGNN baseline adapted from HGB to DGL heterographs.

    The original HGB HetGNN relies on generated random-walk neighbor files. This
    DGL version uses the graph's typed neighbors directly, then performs
    type-level attention over self-content and heterogeneous neighbor channels.
    """

    def __init__(self, graph, input_dims, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        self.content_rnns = nn.ModuleDict({
            ntype: nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            for ntype in self.ntypes
        })
        self.layers = nn.ModuleList([HetGNNLayer(graph, hidden_dim, dropout) for _ in range(num_layers)])
        self.dropout = dropout

    def _content_encode(self, ntype, h):
        sequence = h.unsqueeze(1)
        _out, hidden = self.content_rnns[ntype](sequence)
        return hidden[-1]

    def forward(self, graph, features):
        h_dict = {}
        for ntype in self.ntypes:
            projected = F.relu(self.projectors[ntype](features[ntype].float()))
            h_dict[ntype] = F.dropout(self._content_encode(ntype, projected), self.dropout, training=self.training)
        for layer in self.layers:
            h_dict = layer(graph, h_dict)
        return h_dict


class GTLayer(nn.Module):
    """GTN-style soft relation selection over DGL canonical edge types."""

    def __init__(self, graph, hidden_dim, num_channels=2, dropout=0.5):
        super().__init__()
        self.ntypes = list(graph.ntypes)
        self.canonical_etypes = list(graph.canonical_etypes)
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.relation_logits = nn.Parameter(torch.empty(num_channels, len(self.canonical_etypes)))
        self.channel_fuse = nn.Linear(num_channels * hidden_dim, hidden_dim)
        self.self_loop = nn.ModuleDict({ntype: nn.Linear(hidden_dim, hidden_dim) for ntype in self.ntypes})
        self.dropout = dropout
        nn.init.xavier_normal_(self.relation_logits, gain=1.414)

    def _relation_message(self, graph, etype, h_dict):
        _dgl, fn, _dglnn, _edge_softmax = _import_dgl()
        src_type, _rel, dst_type = etype
        with graph.local_scope():
            graph.nodes[src_type].data["h_gtn_src"] = h_dict[src_type]
            graph.update_all(fn.copy_u("h_gtn_src", "m"), fn.mean("m", "h_gtn_neigh"), etype=etype)
            if "h_gtn_neigh" in graph.nodes[dst_type].data:
                return graph.nodes[dst_type].data["h_gtn_neigh"]
            return h_dict[dst_type].new_zeros(h_dict[dst_type].shape)

    def forward(self, graph, h_dict):
        relation_messages = {etype: self._relation_message(graph, etype, h_dict) for etype in self.canonical_etypes}
        weights = torch.softmax(self.relation_logits, dim=1)
        channel_outputs = [{ntype: self.self_loop[ntype](h_dict[ntype]) for ntype in self.ntypes} for _ in range(self.num_channels)]
        for rel_idx, etype in enumerate(self.canonical_etypes):
            dst_type = etype[2]
            message = relation_messages[etype]
            for channel_idx in range(self.num_channels):
                channel_outputs[channel_idx][dst_type] = channel_outputs[channel_idx][dst_type] + weights[channel_idx, rel_idx] * message
        out = {}
        for ntype in self.ntypes:
            fused = self.channel_fuse(torch.cat([channel_outputs[channel_idx][ntype] for channel_idx in range(self.num_channels)], dim=1))
            out[ntype] = F.dropout(F.relu(fused), self.dropout, training=self.training)
        return out


class GTNEncoder(nn.Module):
    """GTN baseline adapted from Graph Transformer Networks to DGL heterographs.

    The original GTN learns soft selections over relation adjacency matrices to
    construct useful meta-path graphs. This DGL version learns channel-wise
    softmax weights over canonical edge types and mixes their message-passing
    outputs before the shared link-prediction head.
    """

    def __init__(self, graph, input_dims, hidden_dim, num_layers=2, num_channels=2, dropout=0.5):
        super().__init__()
        if not graph.canonical_etypes:
            raise ValueError("GTN requires at least one edge type")
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        self.layers = nn.ModuleList([
            GTLayer(graph, hidden_dim, num_channels=num_channels, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = dropout

    def forward(self, graph, features):
        h_dict = {
            ntype: F.dropout(F.relu(self.projectors[ntype](features[ntype].float())), self.dropout, training=self.training)
            for ntype in self.ntypes
        }
        for layer in self.layers:
            h_dict = layer(graph, h_dict)
        return h_dict


class RGCNLayer(nn.Module):
    def __init__(self, graph, hidden_dim, dropout):
        super().__init__()
        _dgl, _fn, dglnn, _edge_softmax = _import_dgl()
        relation_names = [etype[1] for etype in graph.canonical_etypes]
        if len(relation_names) != len(set(relation_names)):
            raise ValueError("RGCN baseline requires unique relation names in canonical_etypes")
        self.conv = dglnn.HeteroGraphConv({
            etype[1]: dglnn.GraphConv(hidden_dim, hidden_dim, norm="right", weight=True, bias=True, allow_zero_in_degree=True)
            for etype in graph.canonical_etypes
        }, aggregate="sum")
        self.dropout = dropout

    def forward(self, graph, h_dict):
        out = self.conv(graph, h_dict)
        result = {}
        for ntype, h in h_dict.items():
            updated = out[ntype] if ntype in out else h
            result[ntype] = F.dropout(F.relu(updated), self.dropout, training=self.training)
        return result


class RGCNEncoder(nn.Module):
    """RGCN baseline using DGL HeteroGraphConv over all canonical edge types."""

    def __init__(self, graph, input_dims, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        self.layers = nn.ModuleList([RGCNLayer(graph, hidden_dim, dropout) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, graph, features):
        h_dict = {
            ntype: F.dropout(F.relu(self.projectors[ntype](features[ntype].float())), self.dropout, training=self.training)
            for ntype in self.ntypes
        }
        for layer in self.layers:
            h_dict = layer(graph, h_dict)
        return h_dict


def build_baseline_encoder(model_name, graph, input_dims, hidden_dim, gnn_layers, dropout, metapaths, target_etype,
                           num_heads=4, num_bases=-1, magnn_rnn_type="gru", gtn_channels=2):
    del target_etype, num_bases
    model_name = model_name.lower()
    if model_name == "han":
        return HANEncoder(graph, input_dims, hidden_dim, metapaths, num_heads=num_heads, dropout=dropout)
    if model_name == "magnn":
        return MAGNNEncoder(graph, input_dims, hidden_dim, metapaths, num_heads=num_heads, dropout=dropout, rnn_type=magnn_rnn_type)
    if model_name == "hgt":
        return HGTEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, num_heads=num_heads, dropout=dropout)
    if model_name == "gtn":
        return GTNEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, num_channels=gtn_channels, dropout=dropout)
    if model_name == "hetgnn":
        return HetGNNEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, dropout=dropout)
    if model_name == "rgcn":
        return RGCNEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, dropout=dropout)
    raise ValueError("Unknown baseline model '{}'".format(model_name))
