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
