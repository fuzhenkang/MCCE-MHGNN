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
  -> choose encoder: MCCE-MHGCN, HAN, HGT, RGCN, MAGNN, or HetGNN
  -> DistMult / dot / MLP target-edge link prediction
  -> dynamic negative sampling
  -> positive-negative logsigmoid loss
  -> validation/test AUC, PR-AUC, F1
```

`MCCE-MHGCN` keeps the original multilayer heterogeneous metapath-context encoder. The new baseline encoders in `src/baselines.py` reuse the same data loading, target-edge masks, negative sampling, loss, validation, and test logic.

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

MCCE-MHGCN example:

```bash
python Train_Evaluate.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --feat-key feat \
  --model mcce \
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

## Baselines

The training script supports:

```text
--model mcce    MCCE-MHGCN encoder
--model han     HAN metapath-reachable GAT plus semantic attention
--model hgt     HGT heterogeneous attention over all canonical edge types
--model rgcn    RGCN HeteroGraphConv over all canonical edge types
--model magnn   MAGNN-style metapath instance encoding plus attention
--model hetgnn  HetGNN-style heterogeneous neighbor aggregation plus type attention
```

All baselines use the same target link predictor selected by `--predictor`, so you can compare them under the same DistMult, dot-product, or MLP scoring setting.

RGCN example:

```bash
python Train_Evaluate.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model rgcn \
  --predictor distmult \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --epochs 200 \
  --log-every 10
```

HGT example:

```bash
python Train_Evaluate.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model hgt \
  --num-heads 4 \
  --predictor distmult \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --epochs 200
```

HAN example for the `author/paper/venue` graph:

```bash
python Train_Evaluate.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model han \
  --num-heads 4 \
  --predictor distmult \
  --hidden-dim 128 \
  --metapaths author:coauthor:author,author:author_to_paper:paper>paper:paper_to_author:author,author:author_to_venue:venue>venue:venue_to_author:author,author:author_to_paper:paper>paper:paper_to_paper:paper>paper:paper_to_author:author,author:author_to_venue:venue>venue:venue_to_venue:venue>venue:venue_to_author:author \
  --epochs 200
```

For HAN, `--hidden-dim` must be divisible by `--num-heads`, and the metapaths should be closed, such as `author-paper-author` or `author-venue-author`.

MAGNN example for the `author/paper/venue` graph:

```bash
python Train_Evaluate.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model magnn \
  --num-heads 4 \
  --magnn-rnn-type gru \
  --predictor distmult \
  --hidden-dim 128 \
  --metapaths author:coauthor:author,author:author_to_paper:paper>paper:paper_to_author:author,author:author_to_venue:venue>venue:venue_to_author:author \
  --epochs 200
```

HetGNN example:

```bash
python Train_Evaluate.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model hetgnn \
  --predictor distmult \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --epochs 200
```

MAGNN uses closed metapaths and `--hidden-dim` must be divisible by `--num-heads`. HetGNN uses typed neighbors from the full DGL heterograph and does not require explicit metapaths.

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
--model                   mcce, han, hgt, rgcn, magnn, or hetgnn.
--gnn-layers              Number of encoder layers for MCCE/RGCN/HGT.
--num-heads               Number of attention heads for HAN/HGT/MAGNN.
--metapath-length         Maximum length for automatic metapath enumeration.
--metapath-closure        closed, open, or both.
--magnn-rnn-type           MAGNN sequence encoder: gru, lstm, linear, or average.
--number-layers           Number of stacked MECCH-style context layers.
--context-encoder         mean or attention.
--metapath-fusion         mean, weight, conv, or cat.
--fusion-mode             intra, context, or both.
--predictor               distmult, dot, or mlp.
--target-message-graph    train or full.
--negative-ratio          Number of sampled negatives per positive edge.
```
