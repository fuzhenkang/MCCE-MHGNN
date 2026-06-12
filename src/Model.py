import math
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
    except ImportError as exc:
        raise ImportError("DGL >= 2.1.0 is required with PyTorch 2.3.0.") from exc
    return dgl


def _etype_key(etype):
    return "__".join(etype)


def _metapath_key(metapath):
    return "||".join(_etype_key(etype) for etype in metapath)


class DGLGraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, graph, ntype, etypes, features):
        import dgl.function as fn
        h = self.linear(features.float())
        if not etypes:
            return h
        funcs = {}
        with graph.local_scope():
            graph.nodes[ntype].data["h_gcn"] = h
            for etype in etypes:
                if "norm" in graph.edges[etype].data:
                    funcs[etype] = (fn.u_mul_e("h_gcn", "norm", "m"), fn.sum("m", "h_out"))
                else:
                    funcs[etype] = (fn.copy_u("h_gcn", "m"), fn.mean("m", "h_out"))
            graph.multi_update_all(funcs, "sum")
            return graph.nodes[ntype].data.get("h_out", h)


class MECCHMetapathFusion(nn.Module):
    def __init__(self, n_metapaths, in_dim, out_dim, fusion_type="conv"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "mean":
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "weight":
            self.weight = nn.Parameter(torch.full((n_metapaths,), 1.0 / n_metapaths))
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "conv":
            self.conv = nn.Parameter(torch.full((n_metapaths, in_dim), 1.0 / n_metapaths))
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "cat":
            self.linear = nn.Linear(n_metapaths * in_dim, out_dim)
        else:
            raise ValueError("Unknown metapath_fusion '{}'".format(fusion_type))

    def forward(self, h_list):
        if self.fusion_type == "mean":
            fused = torch.mean(torch.stack(h_list), dim=0)
        elif self.fusion_type == "weight":
            fused = torch.sum(torch.stack(h_list) * self.weight[:, None, None], dim=0)
        elif self.fusion_type == "conv":
            fused = torch.sum(torch.stack(h_list).transpose(0, 1) * self.conv, dim=1)
        else:
            fused = torch.hstack(h_list)
        return self.linear(fused), fused


class MetapathContextEncoder(nn.Module):
    def __init__(self, in_dim, encoder_type="gcn", use_v=False, n_heads=8):
        super().__init__()
        if in_dim % n_heads != 0:
            raise ValueError("in_dim must be divisible by n_heads")
        if encoder_type == "conv":
            encoder_type = "gcn"
        self.encoder_type = encoder_type
        self.use_v = use_v
        self.n_heads = n_heads
        self.d_k = in_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        if encoder_type == "attention":
            self.q_linear = nn.Linear(in_dim, in_dim, bias=False)
            self.k_linear = nn.Linear(in_dim, in_dim, bias=False)
            if use_v:
                self.v_linear = nn.Linear(in_dim, in_dim, bias=False)
        elif encoder_type == "gcn":
            self.source_linear = nn.Linear(in_dim, in_dim, bias=False)
            self.self_linear = nn.Linear(in_dim, in_dim, bias=True)
        elif encoder_type != "mean":
            raise ValueError("Unknown context encoder '{}'".format(encoder_type))

    def forward(self, target_embedding, embeddings, suffix_graphs):
        if self.encoder_type == "attention":
            return self._attention_forward(target_embedding, embeddings, suffix_graphs)
        if self.encoder_type == "gcn":
            return self._gcn_forward(target_embedding, embeddings, suffix_graphs)
        return self._mean_forward(target_embedding, embeddings, suffix_graphs)

    def _mean_forward(self, target_embedding, embeddings, suffix_graphs):
        import dgl.function as fn
        message_sum = target_embedding.new_zeros(target_embedding.shape)
        degree_sum = target_embedding.new_zeros(target_embedding.shape[0])
        for graph, source_type, target_type in suffix_graphs:
            with graph.local_scope():
                graph.nodes[source_type].data["h_src"] = embeddings[source_type]
                graph.update_all(fn.copy_u("h_src", "m"), fn.sum("m", "h_neigh"))
                message_sum = message_sum + graph.nodes[target_type].data.get("h_neigh", target_embedding.new_zeros(target_embedding.shape))
                degree_sum = degree_sum + graph.in_degrees().to(target_embedding.device).float()
        return (message_sum + target_embedding) / (degree_sum.unsqueeze(-1) + 1.0).clamp_min(1.0)

    def _gcn_forward(self, target_embedding, embeddings, suffix_graphs):
        import dgl.function as fn
        message_sum = target_embedding.new_zeros(target_embedding.shape)
        degree_sum = target_embedding.new_zeros(target_embedding.shape[0])
        for graph, source_type, target_type in suffix_graphs:
            with graph.local_scope():
                graph.nodes[source_type].data["h_src"] = self.source_linear(embeddings[source_type])
                graph.update_all(fn.copy_u("h_src", "m"), fn.sum("m", "h_neigh"))
                message_sum = message_sum + graph.nodes[target_type].data.get("h_neigh", target_embedding.new_zeros(target_embedding.shape))
                degree_sum = degree_sum + graph.in_degrees().to(target_embedding.device).float()
        neigh = message_sum / degree_sum.unsqueeze(-1).clamp_min(1.0)
        return F.relu(self.self_linear(target_embedding) + neigh)

    def _attention_forward(self, target_embedding, embeddings, suffix_graphs):
        import dgl.function as fn
        from dgl.nn.functional import edge_softmax
        n_target = target_embedding.shape[0]
        q = self.q_linear(target_embedding).view(n_target, self.n_heads, self.d_k)
        outputs = []
        for graph, source_type, target_type in suffix_graphs:
            source_embedding = embeddings[source_type]
            with graph.local_scope():
                graph.nodes[source_type].data["k"] = self.k_linear(source_embedding).view(-1, self.n_heads, self.d_k)
                graph.nodes[source_type].data["v"] = self.v_linear(source_embedding).view(-1, self.n_heads, self.d_k) if self.use_v else source_embedding.view(-1, self.n_heads, self.d_k)
                graph.nodes[target_type].data["q"] = q
                graph.apply_edges(fn.u_dot_v("k", "q", "score"))
                graph.edata["score"] = graph.edata["score"] / self.sqrt_dk
                graph.edata["alpha"] = edge_softmax(graph, graph.edata["score"], norm_by="dst")
                graph.update_all(fn.u_mul_e("v", "alpha", "m"), fn.sum("m", "h_neigh"))
                outputs.append(graph.nodes[target_type].data.get("h_neigh", target_embedding.new_zeros((n_target, self.n_heads, self.d_k))).reshape(n_target, -1))
        if not outputs:
            return target_embedding
        return torch.mean(torch.stack(outputs + [target_embedding]), dim=0)


class MCCE_MHGCN(nn.Module):
    def __init__(self, graph, input_dims, hidden_dim, gnn_layers=2, dropout=0.5, use_gate=True,
                 metapaths=None, metapath_fusion="conv", context_encoder="gcn", context_use_v=False,
                 context_heads=8, number_layers=1, fusion_mode="both", context_model="mecch"):
        super().__init__()
        if context_model != "mecch":
            raise ValueError("The bin-based implementation currently supports --context-model mecch")
        if not metapaths:
            raise ValueError("MCCE-MHGCN requires at least one metapath")
        self.graph = graph
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.dropout = dropout
        self.use_gate = use_gate
        self.metapaths = metapaths
        self.context_metapaths = [mp for mp in metapaths if all(etype[0] != etype[2] for etype in mp)]
        if fusion_mode != "intra" and not self.context_metapaths:
            raise ValueError("MCCE cross-layer context requires at least one metapath made only of heterogeneous edges")
        dropped = len(self.metapaths) - len(self.context_metapaths)
        if dropped > 0:
            print("MCCE context ignores {} metapath(s) containing same-type edges.".format(dropped))
        self.number_layers = number_layers
        self.fusion_mode = fusion_mode
        self.input_projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        self.intra_etypes = {
            ntype: [etype for etype in graph.canonical_etypes if etype[0] == ntype and etype[2] == ntype]
            for ntype in self.ntypes
        }
        self.intra_gcn = nn.ModuleDict({
            ntype: nn.ModuleList([DGLGraphConvolution(hidden_dim, hidden_dim) for _ in range(gnn_layers)])
            for ntype in self.ntypes
        })
        self.context_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(number_layers)])
        self.context_encoders = nn.ModuleList()
        self.metapath_fuse = nn.ModuleList()
        for _ in range(number_layers):
            encoders = nn.ModuleDict()
            fusers = nn.ModuleDict()
            for target in self.ntypes:
                target_metapaths = [mp for mp in self.context_metapaths if mp[-1][2] == target]
                for mp in target_metapaths:
                    encoders[_metapath_key(mp)] = MetapathContextEncoder(hidden_dim, context_encoder, context_use_v, context_heads)
                if target_metapaths:
                    fusers[target] = MECCHMetapathFusion(len(target_metapaths), hidden_dim, hidden_dim, metapath_fusion)
            self.context_encoders.append(encoders)
            self.metapath_fuse.append(fusers)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self._suffix_cache = None

    def _encode_intra(self, graph, features):
        embeddings = {ntype: F.relu(self.input_projectors[ntype](features[ntype].float())) for ntype in self.ntypes}
        for ntype in self.ntypes:
            for gcn in self.intra_gcn[ntype]:
                embeddings[ntype] = F.relu(gcn(graph, ntype, self.intra_etypes[ntype], embeddings[ntype]))
                embeddings[ntype] = F.dropout(embeddings[ntype], self.dropout, training=self.training)
        return embeddings

    def _prepare_suffix_graphs(self, graph):
        dgl = _import_dgl()
        if self._suffix_cache is not None:
            return self._suffix_cache
        cache = {}
        for mp in self.context_metapaths:
            suffixes = []
            for start in range(0, len(mp)):
                etype_suffix = [etype[1] for etype in mp[start:]]
                try:
                    suffix_graph = dgl.metapath_reachable_graph(graph, etype_suffix)
                except Exception:
                    continue
                suffix_graph = suffix_graph.to(graph.device)
                suffixes.append((suffix_graph, mp[start][0], mp[-1][2]))
            cache[_metapath_key(mp)] = suffixes
        self._suffix_cache = cache
        return cache

    def _context_embedding(self, graph, embeddings, target, layer_idx):
        suffix_cache = self._prepare_suffix_graphs(graph)
        contexts = []
        target_metapaths = [mp for mp in self.context_metapaths if mp[-1][2] == target]
        for mp in target_metapaths:
            key = _metapath_key(mp)
            encoder = self.context_encoders[layer_idx][key]
            contexts.append(encoder(embeddings[target], embeddings, suffix_cache[key]))
        if not contexts:
            return embeddings[target].new_zeros(embeddings[target].shape)
        projected, _ = self.metapath_fuse[layer_idx][target](contexts)
        return self.context_norms[layer_idx](projected)

    def forward(self, graph, features):
        structural = self._encode_intra(graph, features)
        if self.fusion_mode == "intra":
            return structural
        context_input = structural
        cross = None
        for layer_idx in range(self.number_layers):
            cross = {ntype: self._context_embedding(graph, context_input, ntype, layer_idx) for ntype in self.ntypes}
            if layer_idx < self.number_layers - 1:
                context_input = {ntype: F.dropout(F.relu(h), self.dropout, training=self.training) for ntype, h in cross.items()}
        if self.fusion_mode == "context":
            return cross
        output = {}
        for ntype in self.ntypes:
            combined = torch.cat((structural[ntype], cross[ntype]), dim=1)
            if self.use_gate:
                gate = torch.sigmoid(self.gate(combined))
                fused = gate * structural[ntype] + (1.0 - gate) * cross[ntype]
            else:
                fused = F.relu(self.fusion(combined))
            output[ntype] = F.dropout(fused, self.dropout, training=self.training)
        return output


class MCCE_MHGCNLinkPredictor(nn.Module):
    def __init__(self, encoder, target_etype, predictor="distmult", hidden_dim=None, predictor_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.target_etype = target_etype
        self.predictor = predictor
        hidden_dim = hidden_dim or encoder.hidden_dim
        if predictor == "distmult":
            self.relation = nn.Parameter(torch.ones(hidden_dim))
        elif predictor == "mlp":
            predictor_hidden_dim = predictor_hidden_dim or hidden_dim
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 4, predictor_hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(predictor_hidden_dim, 1)
            )
        elif predictor != "dot":
            raise ValueError("predictor must be distmult, dot, or mlp")

    def score_edges(self, embeddings, edge_index):
        src_type, _, dst_type = self.target_etype
        src, dst = edge_index
        src_z = embeddings[src_type][src]
        dst_z = embeddings[dst_type][dst]
        if self.predictor == "mlp":
            edge_features = torch.cat((src_z, dst_z, src_z * dst_z, torch.abs(src_z - dst_z)), dim=1)
            return self.edge_mlp(edge_features).squeeze(-1)
        if self.predictor == "distmult":
            return torch.sum(src_z * self.relation * dst_z, dim=1)
        return torch.sum(F.normalize(src_z, p=2, dim=1) * F.normalize(dst_z, p=2, dim=1), dim=1)

    def forward(self, graph, features, edge_index):
        embeddings = self.encoder(graph, features)
        return self.score_edges(embeddings, edge_index)

