import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train_Evaluate import (
    build_split_batch,
    canonical_etype_to_text,
    enumerate_metapaths,
    filter_message_graph_etypes,
    format_metapath,
    format_metrics,
    parse_metapaths,
    parse_use_etypes,
    remove_non_train_target_edges,
    run_batch,
)
from src.Model import MCCE_MHGCNLinkPredictor
from src.Utils import get_edge_mask, get_node_features, load_dgl_bin_graph, parse_canonical_etype


def build_parser(model_name):
    parser = argparse.ArgumentParser(description="Train the {} baseline.".format(model_name))
    parser.add_argument("--graph-bin", type=str, required=True, help="Path to a DGL .bin file saved by dgl.save_graphs.")
    parser.add_argument("--graph-index", type=int, default=0)
    parser.add_argument("--target-etype", type=str, default=None, help="Canonical etype as src:rel:dst, or a unique relation name.")
    parser.add_argument("--feat-key", type=str, default="feat")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-bases", type=int, default=-1, help="Reserved for RGCN compatibility.")
    parser.add_argument("--magnn-rnn-type", type=str, default="gru", choices=["gru", "lstm", "linear", "average"])
    parser.add_argument("--gtn-channels", type=int, default=2)
    parser.add_argument("--hinormer-layers", type=int, default=2)
    parser.add_argument("--hinormer-beta", type=float, default=1.0)
    parser.add_argument("--edge-dim", type=int, default=64)
    parser.add_argument("--slope", type=float, default=0.2)
    parser.add_argument("--simplehgn-beta", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--target-message-graph", type=str, default="train", choices=["train", "full"])
    parser.add_argument("--use-etypes", type=str, default=None, help="Comma-separated message-passing edge types to keep, as rel or src:rel:dst.")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "pr_auc", "f1"])
    parser.add_argument("--metapaths", type=str, default=None, help="Comma-separated metapaths; use rel>rel or src:rel:dst>src:rel:dst.")
    parser.add_argument("--metapath-length", type=int, default=3)
    parser.add_argument("--metapath-closure", type=str, default="closed", choices=["closed", "open", "both"])
    parser.add_argument("--predictor", type=str, default="distmult", choices=["distmult", "dot", "mlp"])
    parser.add_argument("--predictor-hidden-dim", type=int, default=None)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)
    return parser


def _build_metapaths(args, graph, model_name, require_metapaths):
    metapaths = parse_metapaths(args.metapaths, graph) if args.metapaths else enumerate_metapaths(
        graph, args.metapath_length, args.metapath_closure
    )
    if require_metapaths and not metapaths:
        raise ValueError("{} requires at least one closed metapath.".format(model_name))
    return metapaths or []


def train_baseline(model_name, encoder_factory, require_metapaths=False):
    parser = build_parser(model_name)
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
        print("Message graph keeps etypes: {}".format(
            ", ".join(canonical_etype_to_text(etype) for etype in use_etypes)
        ))

    message_graph = message_graph.to(device)
    features, input_dims = get_node_features(message_graph, args.feat_key)
    features = {ntype: feat.to(device) for ntype, feat in features.items()}

    metapaths = _build_metapaths(args, message_graph, model_name, require_metapaths)
    print("Model: {}".format(model_name.lower()))
    print("Target etype: {}".format(canonical_etype_to_text(target_etype)))
    if metapaths:
        print("Metapaths: {}".format(", ".join(format_metapath(path) for path in metapaths)))
    else:
        print("Metapaths: not required by this baseline.")

    encoder = encoder_factory(args, message_graph, input_dims, metapaths, target_etype)
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
        train_edges, train_labels = build_split_batch(
            raw_graph, target_etype, train_mask, args.negative_ratio, seed=args.seed + epoch
        )
        train_metrics = run_batch(
            model, message_graph, features, train_edges, train_labels, device, train=True, optimizer=optimizer
        )
        if epoch % args.log_every == 0 or epoch == 1:
            print(format_metrics("Epoch {:04d} train".format(epoch), train_metrics))

        if epoch % args.log_every == 0 or epoch == args.epochs:
            valid_edges, valid_labels = build_split_batch(
                raw_graph, target_etype, valid_mask, args.negative_ratio, seed=args.seed + 100000 + epoch
            )
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
    test_edges, test_labels = build_split_batch(
        raw_graph, target_etype, test_mask, args.negative_ratio, seed=args.seed + 200000
    )
    test_metrics = run_batch(model, message_graph, features, test_edges, test_labels, device)
    print(format_metrics("test", test_metrics))
