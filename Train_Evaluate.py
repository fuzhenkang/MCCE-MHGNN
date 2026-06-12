import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve

from src.Model import MCCE_MHGCN, MCCE_MHGCNLinkPredictor
from src.Utils import get_edge_mask, get_node_features, load_dgl_bin_graph, parse_canonical_etype
from src.baselines import build_baseline_encoder


def canonical_etype_to_text(etype):
    return ":".join(etype)


def parse_metapaths(spec, graph):
    if not spec:
        return None
    relation_to_etype = {etype[1]: etype for etype in graph.canonical_etypes}
    metapaths = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        tokens = [token.strip() for token in raw.replace(";", ">").split(">") if token.strip()]
        path = []
        for token in tokens:
            parts = [part.strip() for part in token.replace(",", ":").split(":") if part.strip()]
            if len(parts) == 3:
                etype = tuple(parts)
            elif len(parts) == 1:
                if parts[0] not in relation_to_etype:
                    raise ValueError("Unknown relation '{}' in metapath '{}'".format(parts[0], raw))
                etype = relation_to_etype[parts[0]]
            else:
                raise ValueError("Metapath token '{}' must be rel or src:rel:dst".format(token))
            if etype not in graph.canonical_etypes:
                raise ValueError("Metapath etype {} is not in graph".format(etype))
            if path and path[-1][2] != etype[0]:
                raise ValueError("Metapath '{}' is not type-continuous at {}".format(raw, token))
            path.append(etype)
        if path:
            metapaths.append(tuple(path))
    return metapaths or None


def closure_ok(path, closure):
    closed = path[0][0] == path[-1][2]
    if closure == "closed":
        return closed
    if closure == "open":
        return not closed
    return True


def enumerate_metapaths(graph, max_length, closure="closed"):
    if max_length < 1:
        raise ValueError("--metapath-length must be >= 1")
    outgoing = {}
    for etype in graph.canonical_etypes:
        outgoing.setdefault(etype[0], []).append(etype)
    paths = []

    def dfs(path, remaining):
        if remaining == 0:
            if closure_ok(path, closure):
                paths.append(tuple(path))
            return
        current = path[-1][2]
        for etype in outgoing.get(current, []):
            dfs(path + [etype], remaining - 1)

    for etype in graph.canonical_etypes:
        for length in range(1, max_length + 1):
            dfs([etype], length - 1)
    return sorted(set(paths), key=lambda path: (len(path), tuple(canonical_etype_to_text(e) for e in path)))


def format_metapath(path):
    return " -> ".join(canonical_etype_to_text(etype) for etype in path)


def masked_positive_edges(graph, etype, mask):
    src, dst = graph.edges(etype=etype)
    return src[mask], dst[mask]


def sample_negative_edges(graph, etype, num_samples, seed=None):
    rng = np.random.default_rng(seed)
    src_type, _, dst_type = etype
    n_src = graph.num_nodes(src_type)
    n_dst = graph.num_nodes(dst_type)
    pos_src, pos_dst = graph.edges(etype=etype)
    occupied = set(zip(pos_src.cpu().tolist(), pos_dst.cpu().tolist()))
    rows, cols, used = [], [], set()
    batch_size = max(4096, num_samples * 2)
    while len(rows) < num_samples:
        src = rng.integers(0, n_src, size=batch_size, dtype=np.int64)
        dst = rng.integers(0, n_dst, size=batch_size, dtype=np.int64)
        for u, v in zip(src.tolist(), dst.tolist()):
            key = (u, v)
            if key in occupied or key in used:
                continue
            used.add(key)
            rows.append(u)
            cols.append(v)
            if len(rows) >= num_samples:
                break
    return torch.LongTensor(rows), torch.LongTensor(cols)


def build_split_batch(graph, etype, mask, negative_ratio=1.0, seed=None):
    pos_src, pos_dst = masked_positive_edges(graph, etype, mask)
    num_pos = int(pos_src.numel())
    num_neg = max(1, int(round(num_pos * negative_ratio)))
    neg_src, neg_dst = sample_negative_edges(graph, etype, num_neg, seed=seed)
    edge_index = (torch.cat([pos_src.cpu(), neg_src]), torch.cat([pos_dst.cpu(), neg_dst]))
    labels = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)])
    return edge_index, labels


def remove_non_train_target_edges(graph, etype, train_mask):
    remove_eids = torch.nonzero(~train_mask.cpu(), as_tuple=False).view(-1)
    if remove_eids.numel() == 0:
        return graph
    import dgl
    return dgl.remove_edges(graph, remove_eids, etype=etype)


def parse_use_etypes(spec, graph):
    if not spec:
        return None
    relation_to_etypes = {}
    for etype in graph.canonical_etypes:
        relation_to_etypes.setdefault(etype[1], []).append(etype)
    selected = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        parts = [part.strip() for part in token.split(":") if part.strip()]
        if len(parts) == 3:
            etype = tuple(parts)
            if etype not in graph.canonical_etypes:
                raise ValueError("--use-etypes contains unknown canonical etype {}".format(etype))
            selected.append(etype)
        elif len(parts) == 1:
            matches = relation_to_etypes.get(parts[0], [])
            if not matches:
                raise ValueError("--use-etypes contains unknown relation '{}'".format(parts[0]))
            selected.extend(matches)
        else:
            raise ValueError("--use-etypes token '{}' must be rel or src:rel:dst".format(token))
    selected = sorted(set(selected), key=graph.canonical_etypes.index)
    if not selected:
        raise ValueError("--use-etypes did not select any edge types")
    return selected


def filter_message_graph_etypes(graph, use_etypes):
    if use_etypes is None:
        return graph
    import dgl
    return dgl.edge_type_subgraph(graph, use_etypes)


def link_loss(scores, labels):
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
    pos = scores[labels == 1]
    neg = scores[labels == 0]
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
        topk = min(true_num, y_scores.shape[0])
        y_pred[np.argpartition(-y_scores, topk - 1)[:topk]] = 1
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return {
        "loss": None,
        "auc": binary_auc(y_true, y_scores),
        "pr_auc": float(auc(rs, ps)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def run_batch(model, graph, features, edge_index, labels, device, train=False, optimizer=None):
    edge_index = tuple(index.to(device) for index in edge_index)
    labels = labels.to(device)
    model.train(mode=train)
    with torch.set_grad_enabled(train):
        scores = model(graph, features, edge_index)
        loss = link_loss(scores, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    metrics = compute_metrics(scores, labels)
    metrics["loss"] = float(loss.detach().cpu())
    return metrics


def format_metrics(prefix, metrics):
    return "{} loss={:.6f} auc={:.4f} pr_auc={:.4f} f1={:.4f}".format(
        prefix, metrics.get("loss", float("nan")), metrics.get("auc", float("nan")),
        metrics.get("pr_auc", float("nan")), metrics.get("f1", float("nan"))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bin", type=str, required=True, help="Path to a DGL .bin file saved by dgl.save_graphs.")
    parser.add_argument("--graph-index", type=int, default=0)
    parser.add_argument("--target-etype", type=str, default=None, help="Canonical etype as src:rel:dst, or a unique relation name.")
    parser.add_argument("--feat-key", type=str, default="feat")
    parser.add_argument("--model", type=str, default="mcce", choices=["mcce", "han", "hgt", "rgcn", "magnn", "hetgnn"])
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads for HAN/HGT baselines.")
    parser.add_argument("--num-bases", type=int, default=-1, help="Reserved for RGCN compatibility.")
    parser.add_argument("--magnn-rnn-type", type=str, default="gru", choices=["gru", "lstm", "linear", "average"], help="Metapath sequence encoder for MAGNN.")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--target-message-graph", type=str, default="train", choices=["train", "full"])
    parser.add_argument("--use-etypes", type=str, default=None, help="Comma-separated message-passing edge types to keep, as rel or src:rel:dst.")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--no-gate", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "pr_auc", "f1"])
    parser.add_argument("--metapaths", type=str, default=None, help="Comma-separated metapaths; each metapath uses rel>rel or src:rel:dst>src:rel:dst.")
    parser.add_argument("--metapath-length", type=int, default=3)
    parser.add_argument("--metapath-closure", type=str, default="closed", choices=["closed", "open", "both"])
    parser.add_argument("--number-layers", type=int, default=1)
    parser.add_argument("--metapath-fusion", type=str, default="conv", choices=["mean", "weight", "conv", "cat"])
    parser.add_argument("--fusion-mode", type=str, default="both", choices=["intra", "context", "both"])
    parser.add_argument("--context-model", type=str, default="mecch", choices=["mecch"])
    parser.add_argument("--context-encoder", type=str, default="mean", choices=["mean", "attention"])
    parser.add_argument("--context-use-v", action="store_true", default=False)
    parser.add_argument("--context-heads", type=int, default=8)
    parser.add_argument("--predictor", type=str, default="distmult", choices=["distmult", "dot", "mlp"])
    parser.add_argument("--predictor-hidden-dim", type=int, default=None)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    raw_graph, _metadata = load_dgl_bin_graph(args.graph_bin, args.graph_index)
    target_etype = parse_canonical_etype(raw_graph, args.target_etype)
    train_mask = get_edge_mask(raw_graph, target_etype, "train")
    valid_mask = get_edge_mask(raw_graph, target_etype, "valid")
    test_mask = get_edge_mask(raw_graph, target_etype, "test")

    message_graph = raw_graph
    if args.target_message_graph == "train":
        message_graph = remove_non_train_target_edges(raw_graph, target_etype, train_mask)
        print("Message graph uses only train_mask positives for {}.".format(canonical_etype_to_text(target_etype)))
    else:
        print("Message graph uses the full target relation.")
    use_etypes = parse_use_etypes(args.use_etypes, message_graph)
    if use_etypes is not None:
        message_graph = filter_message_graph_etypes(message_graph, use_etypes)
        print("Message graph keeps etypes: {}".format(", ".join(canonical_etype_to_text(etype) for etype in use_etypes)))
    message_graph = message_graph.to(device)
    features, input_dims = get_node_features(message_graph, args.feat_key)
    features = {ntype: feat.to(device) for ntype, feat in features.items()}

    metapaths = parse_metapaths(args.metapaths, message_graph) if args.metapaths else enumerate_metapaths(
        message_graph, args.metapath_length, args.metapath_closure
    )
    if not metapaths:
        raise ValueError("No valid metapaths were found in the DGL graph schema.")
    print("Model: {}".format(args.model))
    print("Target etype: {}".format(canonical_etype_to_text(target_etype)))
    print("Enumerated metapaths: {}".format(", ".join(format_metapath(path) for path in metapaths)))

    if args.model == "mcce":
        encoder = MCCE_MHGCN(
            message_graph,
            input_dims=input_dims,
            hidden_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers,
            dropout=args.dropout,
            use_gate=not args.no_gate,
            metapaths=metapaths,
            metapath_fusion=args.metapath_fusion,
            context_encoder=args.context_encoder,
            context_use_v=args.context_use_v,
            context_heads=args.context_heads,
            number_layers=args.number_layers,
            fusion_mode=args.fusion_mode,
            context_model=args.context_model,
        )
    else:
        encoder = build_baseline_encoder(
            args.model,
            message_graph,
            input_dims=input_dims,
            hidden_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers,
            dropout=args.dropout,
            metapaths=metapaths,
            target_etype=target_etype,
            num_heads=args.num_heads,
            num_bases=args.num_bases,
            magnn_rnn_type=args.magnn_rnn_type,
        )
    model = MCCE_MHGCNLinkPredictor(
        encoder,
        target_etype=target_etype,
        predictor=args.predictor,
        hidden_dim=args.hidden_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        dropout=args.predictor_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_valid_score = -float("inf")
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        train_edges, train_labels = build_split_batch(raw_graph, target_etype, train_mask, args.negative_ratio, seed=args.seed + epoch)
        train_metrics = run_batch(model, message_graph, features, train_edges, train_labels, device, train=True, optimizer=optimizer)
        if epoch % args.log_every == 0 or epoch == 1:
            print(format_metrics("Epoch {:04d} train".format(epoch), train_metrics))

        if epoch % args.log_every == 0 or epoch == args.epochs:
            valid_edges, valid_labels = build_split_batch(raw_graph, target_etype, valid_mask, args.negative_ratio, seed=args.seed + 100000 + epoch)
            valid_metrics = run_batch(model, message_graph, features, valid_edges, valid_labels, device)
            print(format_metrics("Epoch {:04d} valid".format(epoch), valid_metrics))
            valid_score = valid_metrics.get(args.early_stop_metric, float("nan"))
            if np.isfinite(valid_score) and valid_score > best_valid_score:
                best_valid_score = valid_score
                patience_counter = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print("Early stopping at epoch {:04d}.".format(epoch))
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_edges, test_labels = build_split_batch(raw_graph, target_etype, test_mask, args.negative_ratio, seed=args.seed + 200000)
    test_metrics = run_batch(model, message_graph, features, test_edges, test_labels, device)
    print(format_metrics("test", test_metrics))


if __name__ == "__main__":
    main()
