import argparse

import torch


def get_static_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bin", type=str, required=True)
    parser.add_argument("--graph-index", type=int, default=0)
    parser.add_argument("--target-etype", type=str, default=None)
    parser.add_argument("--feat-key", type=str, default="feat")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--target-message-graph", type=str, default="train", choices=["train", "full"])
    parser.add_argument("--negative-ratio", type=float, default=1.0)
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
    parser.add_argument("--predictor", type=str, default="distmult", choices=["distmult", "dot", "mlp"])
    parser.add_argument("--predictor-hidden-dim", type=int, default=None)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    return args
