import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _import_dgl, _metapath_key, _relation_names, SemanticAttention


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
