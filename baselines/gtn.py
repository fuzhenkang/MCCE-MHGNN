import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _import_dgl, _metapath_key, _relation_names, SemanticAttention


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
