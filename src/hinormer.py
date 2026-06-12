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
        import dgl.function as fn
    except ImportError as exc:
        raise ImportError("DGL >= 2.1.0 is required with PyTorch 2.3.0.") from exc
    return fn


class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, z):
        weights = torch.softmax(self.project(z).mean(0), dim=0)
        return (weights.unsqueeze(0) * z).sum(1)


class HINormerAGTLayer(nn.Module):
    """Relation-aware global attention layer adapted from HINormer."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.5, temper=1.0, beta=1.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for HINormer")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temper = temper
        self.beta = beta
        self.linear_l = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)
        self.r_source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.r_target = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_final = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_emb = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, h, relation_h):
        batch_size = h.shape[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        score = self.att_l(self.leaky_relu(fl)) + self.att_r(self.leaky_relu(fr)).permute(0, 1, 3, 2)
        rk = self.r_source(relation_h).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        rq = self.r_target(relation_h).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        score = (score + self.beta * (rk @ rq)) / self.temper
        score = self.dropout_attn(torch.softmax(score, dim=-1))
        context = score @ fr
        h_sa = context.transpose(1, 2).reshape(batch_size, -1, self.hidden_dim)
        return self.norm(h + self.dropout_emb(self.linear_final(h_sa)))


class HINormerLocalLayer(nn.Module):
    """Local heterogeneous structure encoder before global attention."""

    def __init__(self, graph, hidden_dim, dropout):
        super().__init__()
        self.ntypes = list(graph.ntypes)
        self.self_transforms = nn.ModuleDict({ntype: nn.Linear(hidden_dim, hidden_dim) for ntype in self.ntypes})
        self.type_attn = nn.ModuleDict({ntype: SemanticAttention(hidden_dim) for ntype in self.ntypes})
        self.dropout = dropout

    def relation_mean(self, graph, etype, h_dict):
        fn = _import_dgl()
        src_type, _rel, dst_type = etype
        with graph.local_scope():
            graph.nodes[src_type].data["h_hinormer_src"] = h_dict[src_type]
            graph.update_all(fn.copy_u("h_hinormer_src", "m"), fn.mean("m", "h_hinormer_neigh"), etype=etype)
            if "h_hinormer_neigh" in graph.nodes[dst_type].data:
                return graph.nodes[dst_type].data["h_hinormer_neigh"]
            return h_dict[dst_type].new_zeros(h_dict[dst_type].shape)

    def forward(self, graph, h_dict):
        contexts = {ntype: [self.self_transforms[ntype](h_dict[ntype])] for ntype in self.ntypes}
        buckets = {}
        for etype in graph.canonical_etypes:
            src_type, _rel, dst_type = etype
            buckets.setdefault((dst_type, src_type), []).append(self.relation_mean(graph, etype, h_dict))
        for (dst_type, _src_type), values in buckets.items():
            contexts[dst_type].append(torch.stack(values, dim=0).mean(dim=0))
        return {
            ntype: F.dropout(F.relu(self.type_attn[ntype](torch.stack(values, dim=1))), self.dropout, training=self.training)
            for ntype, values in contexts.items()
        }


class HINormerEncoder(nn.Module):
    """HINormer-style encoder for the repository link-prediction framework.

    The original HINormer is a node-classification model with local structure
    encoding plus relation-aware global attention over sampled heterogeneous
    node sequences. This version builds compact per-node sequences from self
    representations and typed neighbor summaries, then returns node embeddings
    for the shared link prediction head.
    """

    def __init__(self, graph, input_dims, hidden_dim, num_local_layers=2, num_transformer_layers=2,
                 num_heads=4, dropout=0.5, beta=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.ntype_to_id = {ntype: idx for idx, ntype in enumerate(self.ntypes)}
        self.projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        self.type_embeddings = nn.Embedding(len(self.ntypes), hidden_dim)
        self.local_layers = nn.ModuleList([HINormerLocalLayer(graph, hidden_dim, dropout) for _ in range(num_local_layers)])
        self.attention_layers = nn.ModuleList([
            HINormerAGTLayer(hidden_dim, num_heads=num_heads, dropout=dropout, beta=beta)
            for _ in range(num_transformer_layers)
        ])
        self.dropout = dropout

    def _type_embedding(self, ntype, count, device):
        ids = torch.full((count,), self.ntype_to_id[ntype], dtype=torch.long, device=device)
        return self.type_embeddings(ids)

    def _relation_mean(self, graph, etype, h_dict):
        fn = _import_dgl()
        src_type, _rel, dst_type = etype
        with graph.local_scope():
            graph.nodes[src_type].data["h_hinormer_seq_src"] = h_dict[src_type]
            graph.update_all(fn.copy_u("h_hinormer_seq_src", "m"), fn.mean("m", "h_hinormer_seq_neigh"), etype=etype)
            if "h_hinormer_seq_neigh" in graph.nodes[dst_type].data:
                return graph.nodes[dst_type].data["h_hinormer_seq_neigh"]
            return h_dict[dst_type].new_zeros(h_dict[dst_type].shape)

    def _build_sequences(self, graph, h_dict, target_type):
        contexts = [h_dict[target_type]]
        relations = [self._type_embedding(target_type, graph.num_nodes(target_type), h_dict[target_type].device)]
        buckets = {}
        for etype in graph.canonical_etypes:
            src_type, _rel, dst_type = etype
            if dst_type != target_type:
                continue
            buckets.setdefault(src_type, []).append(self._relation_mean(graph, etype, h_dict))
        for src_type, values in buckets.items():
            contexts.append(torch.stack(values, dim=0).mean(dim=0))
            relations.append(self._type_embedding(src_type, graph.num_nodes(target_type), h_dict[target_type].device))
        return torch.stack(contexts, dim=1), torch.stack(relations, dim=1)

    def forward(self, graph, features):
        h_dict = {
            ntype: F.dropout(F.relu(self.projectors[ntype](features[ntype].float())), self.dropout, training=self.training)
            for ntype in self.ntypes
        }
        for layer in self.local_layers:
            h_dict = layer(graph, h_dict)
        out = {}
        for ntype in self.ntypes:
            sequence, relation_sequence = self._build_sequences(graph, h_dict, ntype)
            for layer in self.attention_layers:
                sequence = layer(sequence, relation_sequence)
            out[ntype] = F.dropout(sequence[:, 0, :], self.dropout, training=self.training)
        return out
