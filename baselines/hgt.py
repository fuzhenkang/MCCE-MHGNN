import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _import_dgl, _metapath_key, _relation_names, SemanticAttention


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
