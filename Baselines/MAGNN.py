from common import train_baseline
from src.baselines import MAGNNEncoder


def build_encoder(args, graph, input_dims, metapaths, target_etype):
    del target_etype
    return MAGNNEncoder(
        graph,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        metapaths=metapaths,
        num_heads=args.num_heads,
        dropout=args.dropout,
        rnn_type=args.magnn_rnn_type,
    )


if __name__ == "__main__":
    train_baseline("MAGNN", build_encoder, require_metapaths=True)
