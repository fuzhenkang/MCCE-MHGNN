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
        import dgl.nn as dglnn
    except ImportError as exc:
        raise ImportError("DGL >= 2.1.0 is required with PyTorch 2.3.0.") from exc
    return dgl, dglnn


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
        _dgl, dglnn = _import_dgl()
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
        dgl, _dglnn = _import_dgl()
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
    """HAN baseline adapted to the repository link-prediction encoder interface.

    HAN consumes closed metapaths, builds DGL metapath-reachable graphs, then uses
    GAT plus semantic attention for every node type that has at least one closed
    metapath. Node types without HAN metapaths keep the projected input features.
    """

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


class HGTEncoder(nn.Module):
    """HGT baseline using DGL HGTConv on the homogeneous view of the heterograph."""

    def __init__(self, graph, input_dims, hidden_dim, num_layers=2, num_heads=4, dropout=0.5):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for HGT")
        dgl, dglnn = _import_dgl()
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


class RGCNLayer(nn.Module):
    def __init__(self, graph, hidden_dim, dropout):
        super().__init__()
        _dgl, dglnn = _import_dgl()
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
                           num_heads=4, num_bases=-1):
    del target_etype, num_bases
    model_name = model_name.lower()
    if model_name == "han":
        return HANEncoder(graph, input_dims, hidden_dim, metapaths, num_heads=num_heads, dropout=dropout)
    if model_name == "hgt":
        return HGTEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, num_heads=num_heads, dropout=dropout)
    if model_name == "rgcn":
        return RGCNEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, dropout=dropout)
    raise ValueError("Unknown baseline model '{}'".format(model_name))
