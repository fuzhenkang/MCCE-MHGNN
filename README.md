# MCCE-MHGNN

This repository contains `MCCE-MHGCN`, a static multilayer heterogeneous graph model for link prediction. The current implementation reads a DGL `.bin` heterograph directly and uses masks stored in the graph for train/validation/test splits.

## Clone

```bash
git clone https://github.com/fuzhenkang/MCCE-MHGNN.git
cd MCCE-MHGNN
```

## Environment

The intended runtime is:

```text
PyTorch == 2.3.0
DGL >= 2.1.0
```

Install a DGL build compatible with your PyTorch 2.3.0 and CUDA environment.

```bash
pip install -r requirements.txt
```

## Architecture

```text
DGL .bin heterograph
  -> read node features from graph.ndata[feat]
  -> read train/valid/test splits from target edge masks
  -> layer-wise DGL message-passing GCN structural encoder
  -> automatic metapath enumeration from DGL canonical etypes
  -> DGL metapath-reachable MECCH context encoding
  -> MECCH-style fusion over metapath-context channels
  -> gate fusion with structural embedding
  -> DistMult target-edge link prediction
  -> positive-negative logsigmoid loss
```

## Data Format

The model no longer reads `.mat` adjacency matrices or split `.txt` files. Use a DGL graph saved by:

```python
import dgl

dgl.save_graphs("graph.bin", [g])
```

The saved graph should be a DGL heterograph.

### Node Data

Every node type used by the model must contain a feature tensor:

```text
g.nodes[ntype].data["feat"]: shape [num_nodes, feature_dim]
```

Use `--feat-key` if the feature key is not `feat`.

### Edge Data Masks

The target canonical edge type must contain masks:

```text
g.edges[target_etype].data["train_mask"]
g.edges[target_etype].data["val_mask"] or g.edges[target_etype].data["valid_mask"]
g.edges[target_etype].data["test_mask"]
```

Masks should be boolean tensors with length equal to the number of edges in the target edge type. These masks define positive edges for each split. Negative edges are sampled dynamically from node pairs that are not present in the full target relation.

The target edge type can be provided as:

```text
--target-etype source_type:relation_type:target_type
```

or as a unique relation name:

```text
--target-etype relation_type
```

If omitted, the script uses the first edge type that has `train_mask` plus validation/test masks.

## Training

Example:

```bash
python Train_Evaluate.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --feat-key feat \
  --target-message-graph train \
  --negative-ratio 1.0 \
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
  --epochs 200 \
  --patience 10 \
  --early-stop-metric auc \
  --log-every 10
```

`--target-message-graph train` removes non-training target edges from message passing, reducing validation/test leakage. `--target-message-graph full` keeps all target edges for ablation.

## Metapaths

By default, metapaths are automatically enumerated from DGL `canonical_etypes` up to `--metapath-length`.

You can also pass explicit metapaths:

```bash
--metapaths writes>written_by,affiliated_with>has_member
```

or with canonical edge types:

```bash
--metapaths author:writes:paper>paper:written_by:author,author:affiliated_with:org>org:has_member:author
```

Each metapath must be type-continuous: the destination node type of one edge type must equal the source node type of the next edge type.

## Important Parameters

```text
--graph-bin               DGL .bin file saved by dgl.save_graphs.
--graph-index             Graph index inside the .bin file. Default: 0.
--target-etype            Target edge type for link prediction.
--feat-key                Node feature key. Default: feat.
--gnn-layers              Number of ordinary intra-layer GCN layers.
--metapath-length         Maximum length for automatic metapath enumeration.
--metapath-closure        closed, open, or both.
--number-layers           Number of stacked MECCH-style context layers.
--context-encoder         mean or attention.
--metapath-fusion         mean, weight, conv, or cat.
--fusion-mode             intra, context, or both.
--predictor               distmult, dot, or mlp.
--target-message-graph    train or full.
--negative-ratio          Number of sampled negatives per positive edge.
```
