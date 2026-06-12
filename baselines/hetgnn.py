import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _import_dgl, _metapath_key, _relation_names, SemanticAttention


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
