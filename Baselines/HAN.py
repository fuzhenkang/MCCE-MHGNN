from common import train_baseline
from src.baselines import HANEncoder


def build_encoder(args, graph, input_dims, metapaths, target_etype):
    del target_etype
    return HANEncoder(
        graph,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        metapaths=metapaths,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    train_baseline("HAN", build_encoder, require_metapaths=True)
