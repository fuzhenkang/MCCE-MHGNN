import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _import_dgl, _metapath_key, _relation_names, SemanticAttention


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
