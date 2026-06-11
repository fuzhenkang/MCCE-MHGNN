import sys
import types


def import_dgl():
    if "dgl.graphbolt" not in sys.modules:
        graphbolt = types.ModuleType("dgl.graphbolt")
        graphbolt.__all__ = []
        sys.modules["dgl.graphbolt"] = graphbolt
    try:
        import dgl
    except ImportError as exc:
        raise ImportError(
            "DGL >= 2.1.0 is required. Install a build compatible with PyTorch 2.3.0."
        ) from exc
    return dgl


def load_dgl_bin_graph(path, graph_index=0):
    dgl = import_dgl()
    graphs, metadata = dgl.load_graphs(path)
    if not graphs:
        raise ValueError("No graphs found in {}".format(path))
    if graph_index < 0 or graph_index >= len(graphs):
        raise ValueError("graph_index {} out of range for {} graphs".format(graph_index, len(graphs)))
    return graphs[graph_index], metadata


def parse_canonical_etype(graph, spec=None):
    if spec is None:
        for etype in graph.canonical_etypes:
            data = graph.edges[etype].data
            if "train_mask" in data and ("test_mask" in data or "val_mask" in data or "valid_mask" in data):
                return etype
        raise ValueError(
            "--target-etype was not provided and no edge type with train_mask plus validation/test mask was found."
        )

    parts = [part.strip() for part in spec.replace(",", ":").split(":") if part.strip()]
    if len(parts) == 3:
        etype = tuple(parts)
        if etype not in graph.canonical_etypes:
            raise ValueError("Target etype {} is not in graph.canonical_etypes".format(etype))
        return etype
    if len(parts) == 1:
        matches = [etype for etype in graph.canonical_etypes if etype[1] == parts[0]]
        if len(matches) == 1:
            return matches[0]
        raise ValueError("Edge type name '{}' matched {} canonical etypes: {}".format(parts[0], len(matches), matches))
    raise ValueError("--target-etype must be 'src:rel:dst' or a unique relation name")


def get_edge_mask(graph, etype, split):
    data = graph.edges[etype].data
    aliases = {
        "train": ["train_mask"],
        "valid": ["valid_mask", "val_mask"],
        "val": ["val_mask", "valid_mask"],
        "test": ["test_mask"],
    }
    for key in aliases.get(split, [split]):
        if key in data:
            return data[key].bool()
    raise KeyError("Edge type {} does not contain a {} mask".format(etype, split))


def get_node_features(graph, feat_key="feat"):
    features = {}
    input_dims = {}
    for ntype in graph.ntypes:
        if feat_key not in graph.nodes[ntype].data:
            raise KeyError("Node type '{}' does not contain feature key '{}'".format(ntype, feat_key))
        feat = graph.nodes[ntype].data[feat_key].float()
        features[ntype] = feat
        input_dims[ntype] = int(feat.shape[1])
    return features, input_dims
