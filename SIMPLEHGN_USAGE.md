# SimpleHGN Baseline

This repository includes an OpenHGNN-inspired SimpleHGN baseline in:

```text
src/simplehgn.py
Train_Evaluate_SimpleHGN.py
```

SimpleHGN extends GAT-style attention by adding edge-type embeddings into the attention coefficient. This adaptation converts the DGL heterograph to a homogeneous graph internally, uses DGL node/edge type IDs, then returns heterogeneous node embeddings for the shared link-prediction head.

## Command

```bash
python -u Train_Evaluate_SimpleHGN.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --num-heads 4 \
  --edge-dim 64 \
  --simplehgn-beta 0.0 \
  --predictor mlp \
  --predictor-hidden-dim 128 \
  --predictor-dropout 0.1 \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --epochs 500 \
  --patience 50 \
  --early-stop-metric auc \
  --log-every 10
```

## Heterogeneous-only Variant

```bash
python -u Train_Evaluate_SimpleHGN.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --use-etypes author_to_paper,paper_to_author,author_to_venue,venue_to_author,paper_to_venue,venue_to_paper \
  --num-heads 4 \
  --edge-dim 64 \
  --predictor distmult \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --epochs 200
```

## Parameters

```text
--gnn-layers        Number of SimpleHGN attention layers.
--num-heads         Number of attention heads. hidden_dim must be divisible by this value.
--edge-dim          Edge-type embedding dimension.
--slope             LeakyReLU negative slope in attention.
--simplehgn-beta    Edge-attention residual weight.
--use-etypes        Optional message graph edge-type filter.
```

This is adapted to the repository link-prediction flow. It is not a node-classification wrapper from OpenHGNN, but the attention equation uses node features plus edge-type embeddings as in SimpleHGN.
