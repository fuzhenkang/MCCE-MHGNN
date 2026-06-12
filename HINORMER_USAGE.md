# HINormer Baseline

This repository includes a DGL `.bin` compatible HINormer-style baseline in:

```text
src/hinormer.py
Train_Evaluate_HINormer.py
```

The original HINormer is a node-classification model with local structure encoding and relation-aware global attention over sampled heterogeneous node sequences. This adaptation keeps those two ideas but returns node embeddings for the shared link-prediction head.

## Command

```bash
python -u Train_Evaluate_HINormer.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --num-heads 4 \
  --hinormer-layers 2 \
  --hinormer-beta 1.0 \
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
python -u Train_Evaluate_HINormer.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --use-etypes author_to_paper,paper_to_author,author_to_venue,venue_to_author,paper_to_venue,venue_to_paper \
  --num-heads 4 \
  --hinormer-layers 2 \
  --predictor distmult \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --epochs 200
```

## Parameters

```text
--gnn-layers        Number of local heterogeneous structure layers.
--hinormer-layers   Number of relation-aware global attention layers.
--hinormer-beta     Weight of relation/type bias in attention.
--num-heads         Number of attention heads. hidden_dim must be divisible by this value.
--use-etypes        Optional message graph edge-type filter.
```
