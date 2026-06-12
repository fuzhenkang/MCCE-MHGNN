from common import train_baseline
from src.simplehgn import SimpleHGNEncoder


def build_encoder(args, graph, input_dims, metapaths, target_etype):
    del metapaths, target_etype
    return SimpleHGNEncoder(
        graph,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.gnn_layers,
        num_heads=args.num_heads,
        edge_dim=args.edge_dim,
        dropout=args.dropout,
        negative_slope=args.slope,
        beta=args.simplehgn_beta,
    )


if __name__ == "__main__":
    train_baseline("SimpleHGN", build_encoder, require_metapaths=False)
