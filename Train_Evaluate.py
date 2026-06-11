import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import auc, f1_score, precision_recall_curve

from src.Model import MCCE_MHGCN, MCCE_MHGCNLinkPredictor
from src.Utils import load_static_mat_data


def _metapath_matches_closure(path, closure):
    is_closed = path[0] == path[-1]
    if closure == "closed":
        return is_closed
    if closure == "open":
        return not is_closed
    if closure == "both":
        return True
    raise ValueError("Unknown metapath closure '{}'".format(closure))


def parse_metapaths(spec, layer_names, closure="closed"):
    if not spec:
        return None
    aliases = {
        "p": "paper", "paper": "paper",
        "a": "author", "author": "author",
        "f": "affiliation", "aff": "affiliation", "affiliation": "affiliation",
        "o": "affiliation", "org": "affiliation", "organization": "affiliation",
    }
    normalized = [aliases.get(str(name).lower(), str(name).lower()) for name in layer_names]
    name_to_index = {name: idx for idx, name in enumerate(normalized)}
    paths = []
    for raw_path in spec.split(","):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        tokens = [token.strip() for token in raw_path.replace(">", "-").replace("_", "-").split("-") if token.strip()]
        if len(tokens) == 1 and not tokens[0].isdigit():
            tokens = list(tokens[0])
        path = []
        for token in tokens:
            key = token.lower()
            index = int(key) if key.isdigit() else name_to_index.get(aliases.get(key, key))
            if index is None or index < 0 or index >= len(layer_names):
                raise ValueError("Unknown metapath token '{}' in '{}'".format(token, raw_path))
            path.append(index)
        if len(path) < 2:
            raise ValueError("Metapath '{}' is too short".format(raw_path))
        if not _metapath_matches_closure(path, closure):
            raise ValueError("Metapath '{}' does not match closure '{}'".format(raw_path, closure))
        paths.append(tuple(path))
    return paths or None


def matrix_has_edges(matrix):
    if matrix is None:
        return False
    if hasattr(matrix, "nnz"):
        return int(matrix.nnz) > 0
    return bool(np.asarray(matrix).size)


def enumerate_metapaths(data, target_layer, max_length, closure="closed"):
    if max_length < 1:
        raise ValueError("--metapath-length must be >= 1")
    num_layers = data["num_layers"]
    intra_adj = data["intra_adj"]
    cross_adj = data["cross_adj"]

    def relation_exists(source, target):
        if source == target:
            return matrix_has_edges(intra_adj[source])
        return matrix_has_edges(cross_adj[target][source])

    paths = []
    target_layers = range(num_layers) if target_layer is None else [target_layer]

    def dfs(path, root, remaining):
        current = path[-1]
        if remaining == 0:
            has_cross_context = any(node != root for node in path[1:])
            if has_cross_context and _metapath_matches_closure(path, closure):
                paths.append(tuple(path))
            return
        for nxt in range(num_layers):
            if not relation_exists(current, nxt):
                continue
            if remaining > 1 and nxt == root:
                continue
            dfs(path + [nxt], root, remaining - 1)

    for root in target_layers:
        for length in range(1, max_length + 1):
            dfs([root], root, length)
    return sorted(set(paths), key=lambda path: (len(path), path))


def format_metapath(path, layer_names):
    return "-".join(str(layer_names[index]) for index in path)


def load_edge_samples(path, allow_unlabeled_positive=False):
    edges, labels = [], []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            values = [int(float(part)) for part in parts]
            if len(values) == 2 and allow_unlabeled_positive:
                src, dst = values
                label = 1
            elif len(values) == 3:
                src, dst, label = values
            elif len(values) >= 4:
                _type, src, dst, label = values[:4]
            else:
                raise ValueError("Each edge line must be src dst, src dst label, or type src dst label")
            edges.append((src, dst))
            labels.append(label)
    if not edges:
        raise ValueError("No edges found in {}".format(path))
    edge_index = torch.LongTensor(edges).t().contiguous()
    return (edge_index[0], edge_index[1]), torch.FloatTensor(labels)


def load_labeled_edges(path):
    return load_edge_samples(path, allow_unlabeled_positive=False)


def load_positive_edges(path):
    edge_index, edge_label = load_edge_samples(path, allow_unlabeled_positive=True)
    positive = edge_label == 1
    if int(positive.sum()) == 0:
        raise ValueError("No positive edges found in {}".format(path))
    return edge_index[0][positive], edge_index[1][positive]


def _copy_cross_adj(cross_adj):
    return [list(row) for row in cross_adj]


def build_train_message_graph(graph_data, train_path, target_layer, source_layer=None, link_task="intra", undirected=True):
    edge_index = load_positive_edges(train_path)
    rows = edge_index[0].numpy().astype(np.int64)
    cols = edge_index[1].numpy().astype(np.int64)
    source_layer = target_layer if source_layer is None else source_layer
    train_graph = dict(graph_data)
    if link_task == "intra":
        num_nodes = graph_data["intra_adj"][target_layer].shape[0]
        if undirected:
            rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
        adj = sparse.csr_matrix((np.ones(rows.shape[0], dtype=np.float32), (rows, cols)), shape=(num_nodes, num_nodes))
        adj.setdiag(0)
        adj.eliminate_zeros()
        train_graph["intra_adj"] = list(graph_data["intra_adj"])
        train_graph["intra_adj"][target_layer] = adj
        return train_graph
    relation = graph_data["cross_adj"][target_layer][source_layer]
    if relation is None:
        raise ValueError("cross_adj[{}][{}] is missing".format(target_layer, source_layer))
    adj = sparse.csr_matrix((np.ones(rows.shape[0], dtype=np.float32), (cols, rows)), shape=relation.shape)
    train_graph["cross_adj"] = _copy_cross_adj(graph_data["cross_adj"])
    train_graph["cross_adj"][target_layer][source_layer] = adj
    reverse = graph_data["cross_adj"][source_layer][target_layer]
    if reverse is not None and reverse.shape == adj.T.shape:
        train_graph["cross_adj"][source_layer][target_layer] = adj.T.tocsr()
    return train_graph


def build_negative_exclusion_adj(raw_graph_data, train_graph_data, target_layer, mode, source_layer=None, link_task="intra", undirected=True):
    data = raw_graph_data if mode == "full" else train_graph_data
    source_layer = target_layer if source_layer is None else source_layer
    if link_task == "intra":
        matrix = data["intra_adj"][target_layer].tocsr().astype(np.float32)
        if undirected:
            matrix = matrix + matrix.T
        matrix.data[:] = 1.0
        matrix.setdiag(1.0)
        matrix.eliminate_zeros()
        return matrix.tocsr()
    matrix = data["cross_adj"][target_layer][source_layer].tocsr().astype(np.float32)
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix.tocsr()


def sample_negative_edges(exclude_adj, num_samples, undirected=True, seed=None, bipartite=False):
    rng = np.random.default_rng(seed)
    rows, cols, used = [], [], set()
    batch_size = max(4096, num_samples * 2)
    if bipartite:
        n_target, n_source = exclude_adj.shape
        while len(rows) < num_samples:
            src = rng.integers(0, n_source, size=batch_size, dtype=np.int64)
            dst = rng.integers(0, n_target, size=batch_size, dtype=np.int64)
            occupied = np.asarray(exclude_adj[dst, src]).reshape(-1) > 0
            for u, v in zip(src[~occupied].tolist(), dst[~occupied].tolist()):
                if (u, v) not in used:
                    used.add((u, v)); rows.append(u); cols.append(v)
                    if len(rows) >= num_samples: break
        return torch.LongTensor(rows), torch.LongTensor(cols)
    n = exclude_adj.shape[0]
    while len(rows) < num_samples:
        src = rng.integers(0, n, size=batch_size, dtype=np.int64)
        dst = rng.integers(0, n, size=batch_size, dtype=np.int64)
        valid = src != dst
        if undirected:
            valid &= src < dst
        src, dst = src[valid], dst[valid]
        occupied = np.asarray(exclude_adj[src, dst]).reshape(-1) > 0
        for u, v in zip(src[~occupied].tolist(), dst[~occupied].tolist()):
            if (u, v) not in used:
                used.add((u, v)); rows.append(u); cols.append(v)
                if len(rows) >= num_samples: break
    return torch.LongTensor(rows), torch.LongTensor(cols)


def build_dynamic_train_batch(train_pos_edge_index, exclude_adj, negative_ratio, undirected=True, seed=None, link_task="intra"):
    pos_src, pos_dst = train_pos_edge_index
    num_pos = int(pos_src.numel())
    num_neg = max(1, int(round(num_pos * negative_ratio)))
    neg_src, neg_dst = sample_negative_edges(exclude_adj, num_neg, undirected=undirected, seed=seed, bipartite=(link_task == "cross"))
    return (torch.cat([pos_src.cpu(), neg_src]), torch.cat([pos_dst.cpu(), neg_dst])), torch.cat([torch.ones(num_pos), torch.zeros(num_neg)])


def mhgcn_link_loss(scores, labels):
    labels = labels.float()
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    losses = []
    if pos_scores.numel() > 0:
        losses.append(F.logsigmoid(pos_scores))
    if neg_scores.numel() > 0:
        losses.append(F.logsigmoid(-neg_scores))
    return -torch.mean(torch.cat(losses)) if losses else scores.new_tensor(0.0)


def binary_auc(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    neg_sorted = np.sort(neg)
    higher = np.searchsorted(neg_sorted, pos, side="left").sum()
    higher_or_equal = np.searchsorted(neg_sorted, pos, side="right").sum()
    return float((higher + 0.5 * (higher_or_equal - higher)) / (pos.size * neg.size))


def compute_metrics(scores, labels):
    y_scores = scores.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(np.int64)
    y_pred = np.zeros_like(y_true, dtype=np.int64)
    true_num = int(y_true.sum())
    if true_num > 0:
        top_idx = np.argpartition(-y_scores, min(true_num, y_scores.shape[0]) - 1)[:true_num]
        y_pred[top_idx] = 1
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return {"loss": None, "auc": binary_auc(y_true, y_scores), "pr_auc": float(auc(rs, ps)), "f1": float(f1_score(y_true, y_pred))}


def run_sample(model, graph_data, edge_path, device, loss_fn=None, train=False, optimizer=None, target_layer=None, source_layer=None, edge_index=None, edge_label=None):
    if edge_index is None or edge_label is None:
        edge_index, edge_label = load_labeled_edges(edge_path)
    edge_index = tuple(index.to(device) for index in edge_index)
    edge_label = edge_label.to(device)
    model.train(mode=train)
    with torch.set_grad_enabled(train):
        scores = model(graph_data["features_by_layer"], graph_data["intra_adj"], graph_data["cross_adj"], edge_index, target_layer=target_layer, source_layer=source_layer)
        loss = loss_fn(scores, edge_label) if loss_fn is not None else None
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    metrics = compute_metrics(scores, edge_label)
    if loss is not None:
        metrics["loss"] = float(loss.detach().cpu())
    return metrics


def format_metrics(prefix, metrics):
    return "{} loss={:.6f} auc={:.4f} pr_auc={:.4f} f1={:.4f}".format(prefix, metrics.get("loss", float("nan")), metrics.get("auc", float("nan")), metrics.get("pr_auc", float("nan")), metrics.get("f1", float("nan")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", type=str, required=True)
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--valid-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--test-glob", type=str, default=None)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--cross-order", type=int, default=1)
    parser.add_argument("--intra-sample-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--link-task", type=str, default="intra", choices=["intra", "cross"])
    parser.add_argument("--source-layer", type=int, default=None)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--target-message-graph", type=str, default="train", choices=["train", "full"])
    parser.add_argument("--directed-target-graph", action="store_true", default=False)
    parser.add_argument("--train-negative-mode", type=str, default="dynamic", choices=["dynamic", "file"])
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--negative-exclude-graph", type=str, default="full", choices=["full", "train"])
    parser.add_argument("--no-gate", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "pr_auc", "f1"])
    parser.add_argument("--metapaths", type=str, default=None)
    parser.add_argument("--metapath-length", type=int, default=3)
    parser.add_argument("--metapath-closure", type=str, default="closed", choices=["closed", "open", "both"])
    parser.add_argument("--number-layers", type=int, default=1)
    parser.add_argument("--metapath-fusion", type=str, default="conv", choices=["mean", "weight", "conv", "cat"])
    parser.add_argument("--fusion-mode", type=str, default="both", choices=["intra", "context", "both"])
    parser.add_argument("--context-model", type=str, default="mecch", choices=["mecch"])
    parser.add_argument("--context-encoder", type=str, default="mean", choices=["mean", "attention"])
    parser.add_argument("--context-use-v", action="store_true", default=False)
    parser.add_argument("--context-heads", type=int, default=8)
    parser.add_argument("--context-sample-size", type=int, default=None)
    parser.add_argument("--predictor", type=str, default="distmult", choices=["distmult", "dot", "mlp"])
    parser.add_argument("--predictor-hidden-dim", type=int, default=None)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    raw_graph = load_static_mat_data(args.graph_path)
    graph = raw_graph
    target_layer = raw_graph["target_layer"] if args.target_layer is None else args.target_layer
    source_layer = target_layer if args.source_layer is None else args.source_layer
    if args.link_task == "cross" and source_layer == target_layer:
        raise ValueError("--source-layer must differ from --target-layer for cross-link prediction")
    if args.target_message_graph == "train":
        graph = build_train_message_graph(graph, args.train_path, target_layer, source_layer=source_layer, link_task=args.link_task, undirected=not args.directed_target_graph)
        print("{} message graph rebuilt from positive edges in {}".format(args.link_task, args.train_path))
    train_pos = load_positive_edges(args.train_path)
    exclude_adj = build_negative_exclusion_adj(raw_graph, graph, target_layer, args.negative_exclude_graph, source_layer=source_layer, link_task=args.link_task, undirected=not args.directed_target_graph)
    metapaths = parse_metapaths(args.metapaths, graph["layer_names"], args.metapath_closure) if args.metapaths else enumerate_metapaths(graph, None, args.metapath_length, args.metapath_closure)
    if not metapaths:
        raise ValueError("No valid metapaths were found")
    print("Enumerated metapaths: {}".format(", ".join(format_metapath(path, graph["layer_names"]) for path in metapaths)))

    encoder = MCCE_MHGCN(input_dims=graph["input_dims"], hidden_dim=args.hidden_dim, num_layers=graph["num_layers"], gnn_layers=args.gnn_layers, cross_order=args.cross_order, intra_sample_size=args.intra_sample_size, dropout=args.dropout, use_gate=not args.no_gate, metapaths=metapaths, metapath_fusion=args.metapath_fusion, context_encoder=args.context_encoder, context_use_v=args.context_use_v, context_heads=args.context_heads, context_sample_size=args.context_sample_size, metapath_length=args.metapath_length, number_layers=args.number_layers, fusion_mode=args.fusion_mode, context_model=args.context_model)
    model = MCCE_MHGCNLinkPredictor(encoder, target_layer=target_layer, source_layer=source_layer, link_task=args.link_task, predictor=args.predictor, hidden_dim=args.hidden_dim, predictor_hidden_dim=args.predictor_hidden_dim, dropout=args.predictor_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_valid_score, patience_counter = None, -float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_edge_index = train_edge_label = None
        if args.train_negative_mode == "dynamic":
            train_edge_index, train_edge_label = build_dynamic_train_batch(train_pos, exclude_adj, args.negative_ratio, undirected=not args.directed_target_graph, seed=args.seed + epoch, link_task=args.link_task)
        train_metrics = run_sample(model, graph, args.train_path, device, loss_fn=mhgcn_link_loss, train=True, optimizer=optimizer, target_layer=target_layer, source_layer=source_layer, edge_index=train_edge_index, edge_label=train_edge_label)
        if epoch % args.log_every == 0 or epoch == 1:
            print(format_metrics("Epoch {:04d} train".format(epoch), train_metrics))
        if args.valid_path and (epoch % args.log_every == 0 or epoch == args.epochs):
            valid_metrics = run_sample(model, graph, args.valid_path, device, loss_fn=mhgcn_link_loss, target_layer=target_layer, source_layer=source_layer)
            print(format_metrics("Epoch {:04d} valid".format(epoch), valid_metrics))
            valid_score = valid_metrics.get(args.early_stop_metric, float("nan"))
            if np.isfinite(valid_score) and valid_score > best_valid_score:
                best_valid_score = valid_score; patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print("Early stopping at epoch {:04d}".format(epoch)); break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_paths = []
    if args.test_path:
        test_paths.append(args.test_path)
    if args.test_glob:
        test_paths.extend(sorted(glob.glob(args.test_glob)))
    for path in test_paths:
        metrics = run_sample(model, graph, path, device, loss_fn=mhgcn_link_loss, target_layer=target_layer, source_layer=source_layer)
        print(format_metrics("test {}".format(os.path.basename(path)), metrics))


if __name__ == "__main__":
    main()
