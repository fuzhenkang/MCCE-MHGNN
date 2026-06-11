# MCCE-MHGNN

This repository contains `MCCE-MHGCN`, a static multilayer heterogeneous graph model for link prediction. It keeps the MHGCN-style layer-wise structural encoder, but replaces simple cross-layer propagation with MECCH-style metapath context encoding.

## Architecture

```text
Layer features + intra-layer adjacency + cross-layer adjacency grid
  -> DGL heterograph construction for intra-layer and cross-layer relations
  -> layer-wise DGL message-passing GCN structural encoder
  -> automatic metapath enumeration by length and closure type
  -> stacked DGL block-style MECCH metapath context layers
  -> MECCH-style fusion over metapath-context channels
  -> gate fusion with target-layer structural embedding
  -> DistMult target-layer or cross-layer link prediction
  -> positive-negative logsigmoid loss
```

## Main Changes

- Intra-layer GCN now uses DGL message passing instead of direct sparse matrix multiplication.
- Cross-layer MECCH context also uses DGL metapath-reachable edge types and `multi_update_all` style aggregation.
- Intra-layer encoding is controlled by `--gnn-layers`; no `intra_order` parameter is used.
- Link prediction defaults to DistMult: `score = sum(z_source * r * z_target)`.
- Legacy `dot` and `mlp` predictors remain available for ablation.

## Data Format

Training still starts from `.mat` plus split text files. The loader converts graph matrices into DGL heterographs internally.

Required `graph.mat` fields:

```text
features_by_layer: cell/list of layer feature matrices, each [num_nodes_i, feature_dim_i]
intra_adj:         cell/list of intra-layer adjacency matrices, each [num_nodes_i, num_nodes_i]
cross_adj:         num_layers x num_layers cell matrix
```

The cross-layer convention is:

```text
cross_adj[target_layer][source_layer]
shape = [num_target_nodes, num_source_nodes]
```

Optional fields:

```text
target_layer
layer_names
edge_index
edge_label
```

Split files support:

```text
src dst label
type src dst label
```

For dynamic negative sampling, `train.txt` may contain only positive pairs:

```text
src dst
```

For cross-layer prediction, edge files use:

```text
source_id target_id label
```

## Training

Example intra-layer author link prediction:

```bash
python Train_Evaluate.py \
  --graph-path ../data/aminer_mlhgcn_static/graph.mat \
  --train-path ../data/aminer_mlhgcn_static/train.txt \
  --valid-path ../data/aminer_mlhgcn_static/valid.txt \
  --test-path ../data/aminer_mlhgcn_static/test.txt \
  --target-layer 1 \
  --link-task intra \
  --target-message-graph train \
  --train-negative-mode dynamic \
  --negative-ratio 1.0 \
  --negative-exclude-graph full \
  --metapath-length 3 \
  --metapath-closure closed \
  --context-model mecch \
  --context-encoder mean \
  --metapath-fusion conv \
  --fusion-mode both \
  --predictor distmult \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --number-layers 1 \
  --intra-sample-size 16 \
  --epochs 200 \
  --patience 10 \
  --early-stop-metric auc \
  --log-every 10
```

Example cross-layer prediction:

```bash
python Train_Evaluate.py \
  --graph-path ../data/aminer_mlhgcn_static/graph.mat \
  --train-path ../data/aminer_mlhgcn_static/cross_train.txt \
  --valid-path ../data/aminer_mlhgcn_static/cross_valid.txt \
  --test-path ../data/aminer_mlhgcn_static/cross_test.txt \
  --link-task cross \
  --source-layer 1 \
  --target-layer 0 \
  --target-message-graph train \
  --train-negative-mode dynamic \
  --negative-ratio 1.0 \
  --negative-exclude-graph full \
  --metapath-closure both \
  --metapath-length 3 \
  --predictor distmult
```

## Important Parameters

```text
--gnn-layers              Number of ordinary intra-layer GCN layers.
--metapath-length         Maximum edge length for automatic typed-path enumeration.
--metapath-closure        closed, open, or both.
--number-layers           Number of stacked MECCH-style context layers.
--context-model           mecch, han, or magnn.
--context-encoder         mean or attention for MECCH-style channels.
--metapath-fusion         mean, weight, conv, or cat.
--fusion-mode             intra, context, or both.
--predictor               distmult, dot, or mlp.
--target-message-graph    train or full.
--train-negative-mode     dynamic or file.
```

## Dependencies

The code targets PyTorch CUDA builds such as `2.2.3+cu121` or `2.5.0+cu121` with a compatible DGL build. Install the DGL wheel that matches your local PyTorch/CUDA environment.
