from common import train_baseline
from src.baselines import GTNEncoder


def build_encoder(args, graph, input_dims, metapaths, target_etype):
    del metapaths, target_etype
    return GTNEncoder(
        graph,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.gnn_layers,
        num_channels=args.gtn_channels,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    train_baseline("GTN", build_encoder, require_metapaths=False)
