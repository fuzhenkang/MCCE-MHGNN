from .gtn import GTNEncoder
from .han import HANEncoder
from .hetgnn import HetGNNEncoder
from .hgt import HGTEncoder
from .hinormer import HINormerEncoder
from .magnn import MAGNNEncoder
from .rgcn import RGCNEncoder
from .simplehgn import SimpleHGNEncoder


def build_baseline_encoder(model_name, graph, input_dims, hidden_dim, gnn_layers, dropout, metapaths, target_etype,
                           num_heads=4, num_bases=-1, magnn_rnn_type="gru", gtn_channels=2,
                           hinormer_layers=2, hinormer_beta=1.0, edge_dim=64, slope=0.2, simplehgn_beta=0.0):
    del target_etype, num_bases
    model_name = model_name.lower()
    if model_name == "han":
        return HANEncoder(graph, input_dims, hidden_dim, metapaths, num_heads=num_heads, dropout=dropout)
    if model_name == "magnn":
        return MAGNNEncoder(graph, input_dims, hidden_dim, metapaths, num_heads=num_heads, dropout=dropout, rnn_type=magnn_rnn_type)
    if model_name == "hgt":
        return HGTEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, num_heads=num_heads, dropout=dropout)
    if model_name == "gtn":
        return GTNEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, num_channels=gtn_channels, dropout=dropout)
    if model_name == "hetgnn":
        return HetGNNEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, dropout=dropout)
    if model_name == "rgcn":
        return RGCNEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, dropout=dropout)
    if model_name == "hinormer":
        return HINormerEncoder(graph, input_dims, hidden_dim, num_local_layers=gnn_layers,
                               num_transformer_layers=hinormer_layers, num_heads=num_heads,
                               dropout=dropout, beta=hinormer_beta)
    if model_name == "simplehgn":
        return SimpleHGNEncoder(graph, input_dims, hidden_dim, num_layers=gnn_layers, num_heads=num_heads,
                                edge_dim=edge_dim, dropout=dropout, negative_slope=slope, beta=simplehgn_beta)
    raise ValueError("Unknown baseline model '{}'".format(model_name))
