import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _import_dgl, _metapath_key, _relation_names, SemanticAttention


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
        if hasattr(self, "sequence_encoder") and isinstance(self.sequence_encoder, nn.Linear):
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
