import numpy as np
import torch
from scipy import sparse
from scipy.io import loadmat


def _unwrap_cell_value(value):
    while isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 0:
            return value
        value = value.ravel()[0]
    return value


def _cell_to_list(value):
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_unwrap_cell_value(value.ravel()[i]) for i in range(value.size)]
    if isinstance(value, (list, tuple)):
        return [_unwrap_cell_value(item) for item in value]
    return [_unwrap_cell_value(value)]


def _empty_to_none(value):
    if value is None:
        return None
    while isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.ndim == 0:
            value = value.item()
        elif not sparse.issparse(value):
            return sparse.csr_matrix(value)
    return value


def _cell_to_matrix_grid(value, num_layers):
    if not (isinstance(value, np.ndarray) and value.dtype == object):
        raise ValueError("cross_adj must be a MATLAB cell/object array")

    grid = [[None for _ in range(num_layers)] for _ in range(num_layers)]
    if value.shape[0] == num_layers and value.shape[1] == num_layers:
        for target in range(num_layers):
            for source in range(num_layers):
                grid[target][source] = _empty_to_none(value[target, source])
        return grid

    flat = value.ravel()
    if len(flat) != num_layers * num_layers:
        raise ValueError("cross_adj size must be num_layers x num_layers")
    for target in range(num_layers):
        for source in range(num_layers):
            grid[target][source] = _empty_to_none(flat[target * num_layers + source])
    return grid


def _feature_dim(feature):
    try:
        return int(feature.shape[1])
    except Exception:
        return int(feature.toarray().shape[1])


def _parse_layer_names(data, num_layers):
    if "layer_names" not in data:
        return ["layer_{}".format(index) for index in range(num_layers)]
    raw = data["layer_names"].ravel()
    names = []
    for value in raw:
        if isinstance(value, np.ndarray):
            value = value.ravel()[0] if value.size else ""
        names.append(str(value))
    if len(names) != num_layers:
        return ["layer_{}".format(index) for index in range(num_layers)]
    return names


def load_static_mat_data(path):
    """
    Load a static multi-layer heterogeneous graph .mat file.

    Expected keys:
        features_by_layer or features: MATLAB cell/list of layer-specific feature matrices
        intra_adj: MATLAB cell/list of intra-layer adjacency matrices
        cross_adj: MATLAB cell matrix where cross_adj[target, source] maps source nodes to target nodes

    Optional task keys:
        edge_index: candidate node pairs in the target layer, shape [2, E] or [E, 2]
        edge_label: 1 for observed links and 0 for sampled non-links
        target_layer: scalar layer index for link prediction
    """
    data = loadmat(path)

    if "features_by_layer" in data:
        features_by_layer = _cell_to_list(data["features_by_layer"])
    elif "features" in data:
        features_by_layer = _cell_to_list(data["features"])
    else:
        raise KeyError("Expected key 'features_by_layer' or 'features' in the .mat file")

    if "intra_adj" not in data:
        raise KeyError("Expected key 'intra_adj' in the .mat file")
    intra_adj = _cell_to_list(data["intra_adj"])

    num_layers = len(features_by_layer)
    if len(intra_adj) != num_layers:
        raise ValueError("features_by_layer and intra_adj must have the same number of layers")

    if "cross_adj" in data:
        cross_adj = _cell_to_matrix_grid(data["cross_adj"], num_layers)
    else:
        cross_adj = [[None for _ in range(num_layers)] for _ in range(num_layers)]

    result = {
        "features_by_layer": features_by_layer,
        "intra_adj": intra_adj,
        "cross_adj": cross_adj,
        "input_dims": [_feature_dim(feature) for feature in features_by_layer],
        "num_layers": num_layers,
        "target_layer": int(data["target_layer"].ravel()[0]) if "target_layer" in data else 0,
        "layer_names": _parse_layer_names(data, num_layers),
    }

    if "edge_index" in data:
        edge_index = torch.LongTensor(data["edge_index"].astype(np.int64))
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        result["edge_index"] = (edge_index[0], edge_index[1])

    if "edge_label" in data:
        result["edge_label"] = torch.FloatTensor(data["edge_label"].ravel())

    return result
