from common import train_baseline
from src.baselines import RGCNEncoder


def build_encoder(args, graph, input_dims, metapaths, target_etype):
    del metapaths, target_etype
    return RGCNEncoder(
        graph,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.gnn_layers,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    train_baseline("RGCN", build_encoder, require_metapaths=False)
