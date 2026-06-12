import argparse
import csv
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from Train_Evaluate import (
    canonical_etype_to_text,
    enumerate_metapaths,
    filter_message_graph_etypes,
    format_metapath,
    parse_metapaths,
    parse_use_etypes,
)
from baselines import build_baseline_encoder
from src.Model import MCCE_MHGCN
from src.Utils import get_node_features, load_dgl_bin_graph


def parse_target_ntype(graph, spec=None, label_key="label"):
    if spec is not None:
        if spec not in graph.ntypes:
            raise ValueError("--target-ntype '{}' is not in graph.ntypes: {}".format(spec, graph.ntypes))
        return spec
    matches = []
    for ntype in graph.ntypes:
        data = graph.nodes[ntype].data
        if label_key in data and "train_mask" in data and (
            "test_mask" in data or "val_mask" in data or "valid_mask" in data
        ):
            matches.append(ntype)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("--target-ntype was not provided and no node type with labels and masks was found.")
    raise ValueError("--target-ntype is ambiguous. Candidates: {}".format(matches))


def get_node_mask(graph, ntype, split):
    data = graph.nodes[ntype].data
    aliases = {
        "train": ["train_mask"],
        "valid": ["valid_mask", "val_mask"],
        "val": ["val_mask", "valid_mask"],
        "test": ["test_mask"],
    }
    for key in aliases.get(split, [split]):
        if key in data:
            return data[key].bool()
    raise KeyError("Node type '{}' does not contain a {} mask".format(ntype, split))


class NodeClassificationModel(nn.Module):
    def __init__(self, encoder, target_ntype, hidden_dim, num_classes, classifier="linear",
                 classifier_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.target_ntype = target_ntype
        if classifier == "linear":
            self.classifier = nn.Linear(hidden_dim, num_classes)
        elif classifier == "mlp":
            inner_dim = classifier_hidden_dim or hidden_dim
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, inner_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, num_classes),
            )
        else:
            raise ValueError("Unknown classifier '{}'".format(classifier))

    def forward(self, graph, features):
        h_dict = self.encoder(graph, features)
        return self.classifier(h_dict[self.target_ntype])


def build_parser():
    parser = argparse.ArgumentParser(description="Node classification training entrypoint.")
    parser.add_argument("--graph-bin", type=str, required=True, help="Path to a DGL .bin file saved by dgl.save_graphs.")
    parser.add_argument("--graph-index", type=int, default=0)
    parser.add_argument("--target-ntype", type=str, default=None, help="Target node type for node classification.")
    parser.add_argument("--feat-key", type=str, default="feat")
    parser.add_argument("--label-key", type=str, default="label")
    parser.add_argument(
        "--model",
        type=str,
        default="mcce",
        choices=["mcce", "han", "hgt", "rgcn", "magnn", "hetgnn", "gtn", "hinormer", "simplehgn"],
    )
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
    parser.add_argument("--use-etypes", type=str, default=None, help="Comma-separated message-passing edge types to keep.")
    parser.add_argument("--no-gate", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", type=str, default="macro_f1",
                        choices=["loss", "accuracy", "macro_f1", "micro_f1"])
    parser.add_argument("--metapaths", type=str, default=None, help="Comma-separated metapaths.")
    parser.add_argument("--metapath-length", type=int, default=3)
    parser.add_argument("--metapath-closure", type=str, default="closed", choices=["closed", "open", "both"])
    parser.add_argument("--number-layers", type=int, default=1)
    parser.add_argument("--metapath-fusion", type=str, default="conv", choices=["mean", "weight", "conv", "cat"])
    parser.add_argument("--fusion-mode", type=str, default="both", choices=["intra", "context", "both"])
    parser.add_argument("--context-model", type=str, default="mecch", choices=["mecch"])
    parser.add_argument("--context-encoder", type=str, default="gcn", choices=["gcn", "conv", "mean", "attention"], help="Cross-layer semantic encoder: gcn/conv for metapath context subgraph convolution, or attention.")
    parser.add_argument("--context-use-v", action="store_true", default=False)
    parser.add_argument("--context-heads", type=int, default=8)
    parser.add_argument("--classifier", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--classifier-hidden-dim", type=int, default=None)
    parser.add_argument("--classifier-dropout", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def build_encoder(args, graph, input_dims, metapaths):
    if args.model == "mcce":
        return MCCE_MHGCN(
            graph,
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
    return build_baseline_encoder(
        args.model,
        graph,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        metapaths=metapaths,
        target_etype=None,
        num_heads=args.num_heads,
        num_bases=args.num_bases,
        magnn_rnn_type=args.magnn_rnn_type,
        gtn_channels=args.gtn_channels,
        hinormer_layers=args.hinormer_layers,
        hinormer_beta=args.hinormer_beta,
        edge_dim=args.edge_dim,
        slope=args.slope,
        simplehgn_beta=args.simplehgn_beta,
    )


def compute_metrics(logits, labels, mask):
    mask = mask.bool()
    logits = logits[mask]
    labels = labels[mask].long()
    loss = F.cross_entropy(logits, labels)
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    true = labels.detach().cpu().numpy()
    return {
        "loss": float(loss.detach().cpu()),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true, pred, average="micro", zero_division=0)),
    }, loss


def run_epoch(model, graph, features, labels, mask, train=False, optimizer=None):
    model.train(mode=train)
    with torch.set_grad_enabled(train):
        logits = model(graph, features)
        metrics, loss = compute_metrics(logits, labels, mask)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return metrics


def format_metrics(prefix, metrics):
    return "{} loss={:.6f} accuracy={:.4f} macro_f1={:.4f} micro_f1={:.4f}".format(
        prefix,
        metrics.get("loss", float("nan")),
        metrics.get("accuracy", float("nan")),
        metrics.get("macro_f1", float("nan")),
        metrics.get("micro_f1", float("nan")),
    )


def metric_record(epoch, split, metrics):
    return {
        "epoch": epoch,
        "split": split,
        "loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "micro_f1": metrics.get("micro_f1"),
    }


def safe_run_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "run"


def save_outputs(args, target_ntype, records, best_valid_score, test_metrics):
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or "{}_nodecls_{}_{}".format(time.strftime("%Y%m%d_%H%M%S"), args.model, target_ntype)
    run_name = safe_run_name(run_name)
    metrics_path = os.path.join(args.output_dir, run_name + "_metrics.csv")
    summary_path = os.path.join(args.output_dir, run_name + "_summary.json")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "accuracy", "macro_f1", "micro_f1"])
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "task": "node_classification",
        "model": args.model,
        "target_ntype": target_ntype,
        "best_valid_{}".format(args.early_stop_metric): best_valid_score,
        "test": {key: test_metrics.get(key) for key in ["loss", "accuracy", "macro_f1", "micro_f1"]},
        "args": vars(args),
        "metrics_csv": metrics_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Saved metrics to {}".format(metrics_path))
    print("Saved summary to {}".format(summary_path))


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    raw_graph, _metadata = load_dgl_bin_graph(args.graph_bin, args.graph_index)
    target_ntype = parse_target_ntype(raw_graph, args.target_ntype, args.label_key)
    labels = raw_graph.nodes[target_ntype].data[args.label_key].long().to(device)
    train_mask = get_node_mask(raw_graph, target_ntype, "train").to(device)
    valid_mask = get_node_mask(raw_graph, target_ntype, "valid").to(device)
    test_mask = get_node_mask(raw_graph, target_ntype, "test").to(device)
    num_classes = int(labels.max().detach().cpu().item()) + 1

    message_graph = raw_graph
    use_etypes = parse_use_etypes(args.use_etypes, message_graph)
    if use_etypes is not None:
        message_graph = filter_message_graph_etypes(message_graph, use_etypes)
        print("Message graph keeps etypes: {}".format(
            ", ".join(canonical_etype_to_text(etype) for etype in use_etypes)
        ))

    message_graph = message_graph.to(device)
    features, input_dims = get_node_features(message_graph, args.feat_key)
    features = {ntype: feat.to(device) for ntype, feat in features.items()}

    metapaths = parse_metapaths(args.metapaths, message_graph) if args.metapaths else enumerate_metapaths(
        message_graph, args.metapath_length, args.metapath_closure
    )
    metapath_models = {"mcce", "han", "magnn"}
    if args.model in metapath_models and not metapaths:
        raise ValueError("{} requires at least one valid metapath.".format(args.model))

    print("Task: node_classification")
    print("Model: {}".format(args.model))
    print("Target ntype: {}".format(target_ntype))
    print("Num classes: {}".format(num_classes))
    print("Masks: train={} valid={} test={}".format(
        int(train_mask.sum().cpu()), int(valid_mask.sum().cpu()), int(test_mask.sum().cpu())
    ))
    if metapaths:
        print("Enumerated metapaths: {}".format(", ".join(format_metapath(path) for path in metapaths)))
    else:
        print("Enumerated metapaths: not required by this model.")

    encoder = build_encoder(args, message_graph, input_dims, metapaths)
    model = NodeClassificationModel(
        encoder,
        target_ntype=target_ntype,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        classifier=args.classifier,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.classifier_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    records = []
    best_state = None
    best_valid_score = float("inf") if args.early_stop_metric == "loss" else -float("inf")
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, message_graph, features, labels, train_mask, train=True, optimizer=optimizer)
        records.append(metric_record(epoch, "train", train_metrics))
        if epoch % args.log_every == 0 or epoch == 1:
            print(format_metrics("Epoch {:04d} train".format(epoch), train_metrics))

        if epoch % args.log_every == 0 or epoch == args.epochs:
            valid_metrics = run_epoch(model, message_graph, features, labels, valid_mask)
            records.append(metric_record(epoch, "valid", valid_metrics))
            print(format_metrics("Epoch {:04d} valid".format(epoch), valid_metrics))
            valid_score = valid_metrics.get(args.early_stop_metric, float("nan"))
            is_better = valid_score < best_valid_score if args.early_stop_metric == "loss" else valid_score > best_valid_score
            if np.isfinite(valid_score) and is_better:
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
    test_metrics = run_epoch(model, message_graph, features, labels, test_mask)
    records.append(metric_record("final", "test", test_metrics))
    print(format_metrics("test", test_metrics))
    save_outputs(args, target_ntype, records, best_valid_score, test_metrics)


if __name__ == "__main__":
    main()
