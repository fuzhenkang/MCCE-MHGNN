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
        from dgl.nn.functional import edge_softmax
    except ImportError as exc:
        raise ImportError("DGL >= 2.1.0 is required with PyTorch 2.3.0.") from exc
    return dgl, fn, edge_softmax


class SimpleHGNConv(nn.Module):
    """SimpleHGN convolution adapted from OpenHGNN to this link-prediction repo."""

    def __init__(self, in_dim, out_dim, num_heads, num_etypes, edge_dim, feat_drop=0.5,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.num_etypes = num_etypes
        self.beta = beta
        self.activation = activation
        self.edge_emb = nn.Parameter(torch.empty(num_etypes, edge_dim))
        self.node_linear = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.edge_linear = nn.Linear(edge_dim, edge_dim * num_heads, bias=False)
        self.a_l = nn.Parameter(torch.empty(1, num_heads, out_dim))
        self.a_r = nn.Parameter(torch.empty(1, num_heads, out_dim))
        self.a_e = nn.Parameter(torch.empty(1, num_heads, edge_dim))
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = nn.Linear(in_dim, out_dim * num_heads) if residual else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.node_linear.weight, gain=1.414)
        nn.init.xavier_uniform_(self.edge_linear.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_l, gain=1.414)
        nn.init.xavier_uniform_(self.a_r, gain=1.414)
        nn.init.xavier_uniform_(self.a_e, gain=1.414)
        if self.residual is not None:
            nn.init.xavier_uniform_(self.residual.weight, gain=1.414)

    def forward(self, graph, h, etype):
        _dgl, fn, edge_softmax = _import_dgl()
        h_in = h
        h = self.node_linear(self.feat_drop(h)).view(-1, self.num_heads, self.out_dim)
        edge_h = self.edge_linear(self.edge_emb[etype]).view(-1, self.num_heads, self.edge_dim)
        src, dst = graph.edges()
        score = (self.a_l * h).sum(dim=-1)[src] + (self.a_r * h).sum(dim=-1)[dst] + (self.a_e * edge_h).sum(dim=-1)
        alpha = edge_softmax(graph, self.leaky_relu(score))
        if "alpha" in graph.edata:
            alpha = alpha * (1.0 - self.beta) + graph.edata["alpha"] * self.beta
        graph.edata["alpha"] = alpha.detach()
        with graph.local_scope():
            graph.srcdata["h_simplehgn"] = h
            graph.edata["alpha_simplehgn"] = alpha.unsqueeze(-1)
            graph.update_all(fn.u_mul_e("h_simplehgn", "alpha_simplehgn", "m"), fn.sum("m", "h_out"))
            h_out = graph.dstdata["h_out"].reshape(-1, self.out_dim * self.num_heads)
        if self.residual is not None:
            h_out = h_out + self.residual(h_in)
        if self.activation is not None:
            h_out = self.activation(h_out)
        return h_out


class SimpleHGNEncoder(nn.Module):
    """SimpleHGN encoder for DGL heterograph link prediction.

    The encoder follows the OpenHGNN SimpleHGN idea: convert the heterograph to a
    homogeneous graph, use edge-type embeddings in graph attention, apply node
    residuals and optional edge-attention residuals, then split node embeddings
    back by type for the shared link-prediction head.
    """

    def __init__(self, graph, input_dims, hidden_dim, num_layers=2, num_heads=4, edge_dim=64,
                 dropout=0.5, negative_slope=0.2, beta=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for SimpleHGN")
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        self.layers = nn.ModuleList([
            SimpleHGNConv(
                hidden_dim,
                hidden_dim // num_heads,
                num_heads,
                len(graph.canonical_etypes),
                edge_dim,
                feat_drop=dropout,
                negative_slope=negative_slope,
                residual=layer_idx > 0,
                activation=F.elu,
                beta=beta,
            )
            for layer_idx in range(num_layers)
        ])
        self.dropout = dropout

    def _to_homogeneous(self, graph, h_dict):
        dgl, _fn, _edge_softmax = _import_dgl()
        with graph.local_scope():
            for ntype, h in h_dict.items():
                graph.nodes[ntype].data["h_simplehgn"] = h
            return dgl.to_homogeneous(graph, ndata=["h_simplehgn"])

    def _split_by_type(self, graph, homogeneous_graph, h):
        dgl, _fn, _edge_softmax = _import_dgl()
        out = {}
        ntype_ids = homogeneous_graph.ndata[dgl.NTYPE]
        original_ids = homogeneous_graph.ndata[dgl.NID]
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
        homogeneous_graph = self._to_homogeneous(graph, h_dict)
        h = homogeneous_graph.ndata["h_simplehgn"]
        etype = homogeneous_graph.edata[_import_dgl()[0].ETYPE]
        for layer in self.layers:
            h = F.dropout(layer(homogeneous_graph, h, etype), self.dropout, training=self.training)
        return self._split_by_type(graph, homogeneous_graph, h)
