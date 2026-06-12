from common import train_baseline
from src.hinormer import HINormerEncoder


def build_encoder(args, graph, input_dims, metapaths, target_etype):
    del metapaths, target_etype
    return HINormerEncoder(
        graph,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_local_layers=args.gnn_layers,
        num_transformer_layers=args.hinormer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        beta=args.hinormer_beta,
    )


if __name__ == "__main__":
    train_baseline("HINormer", build_encoder, require_metapaths=False)
