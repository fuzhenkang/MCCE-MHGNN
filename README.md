# MCCE-MHGNN

This repository contains `MCCE-MHGCN`, a static multilayer heterogeneous graph model for link prediction. The current implementation reads a DGL `.bin` heterograph directly and uses masks stored in the graph for train/validation/test splits.

## Clone

```bash
git clone https://github.com/fuzhenkang/MCCE-MHGNN.git
cd MCCE-MHGNN
```

## Environment

```text
PyTorch == 2.3.0
DGL >= 2.1.0
```

Install a DGL build compatible with your PyTorch 2.3.0 and CUDA environment.

```bash
pip install -r requirements.txt
```

## Training Entrypoints

Use `Link_Prediction.py` as the unified link prediction entry for MCCE-MHGCN and all baselines:

```bash
python Link_Prediction.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model mcce
```

`Train_Evaluate.py` contains shared helper functions and is kept for compatibility. New training commands should use `Link_Prediction.py`.

Use `Node_Classification.py` for node classification:

```bash
python Node_Classification.py \
  --graph-bin data/graph.bin \
  --target-ntype author \
  --model hgt
```

## Code Layout

Baseline model implementations are organized as independent files under `baselines/`:

```text
baselines/han.py
baselines/magnn.py
baselines/hetgnn.py
baselines/hgt.py
baselines/rgcn.py
baselines/gtn.py
baselines/hinormer.py
baselines/simplehgn.py
```

`baselines/__init__.py` only exports the encoders and the baseline factory used by `Link_Prediction.py`.

## Data Format

The model reads a DGL heterograph saved by:

```python
import dgl

dgl.save_graphs("graph.bin", [g])
```

Every node type used by the encoder must have node features:

```text
g.nodes[ntype].data["feat"]
```

Use `--feat-key` if your feature key is not `feat`.

The target canonical edge type must contain split masks:

```text
g.edges[target_etype].data["train_mask"]
g.edges[target_etype].data["val_mask"] or g.edges[target_etype].data["valid_mask"]
g.edges[target_etype].data["test_mask"]
```

The masks define positive edges. Negative edges are sampled dynamically from node pairs that do not appear in the full target relation.

For node classification, the target node type must contain:

```text
g.nodes[target_ntype].data["label"]
g.nodes[target_ntype].data["train_mask"]
g.nodes[target_ntype].data["val_mask"] or g.nodes[target_ntype].data["valid_mask"]
g.nodes[target_ntype].data["test_mask"]
```

## Supported Models

```text
--model mcce       MCCE-MHGCN encoder
--model han        HAN metapath-reachable GAT plus semantic attention
--model magnn      MAGNN-style metapath instance encoding plus attention
--model hetgnn     HetGNN-style heterogeneous neighbor aggregation
--model hgt        HGT heterogeneous attention
--model rgcn       RGCN HeteroGraphConv
--model gtn        GTN-style soft relation selection
--model hinormer   HINormer-style local encoder plus relation-aware global attention
--model simplehgn  SimpleHGN-style edge-type-aware heterogeneous attention
```

All models use the same target-edge predictor selected by `--predictor`: `distmult`, `dot`, or `mlp`.

## Outputs

Every training run saves metrics under `outputs/` by default:

```text
outputs/<run_name>_metrics.csv
outputs/<run_name>_summary.json
```

For link prediction, the CSV records `epoch`, `split`, `loss`, `auc`, `pr_auc`, and `f1`.

For node classification, the CSV records `epoch`, `split`, `loss`, `accuracy`, `macro_f1`, and `micro_f1`.

Use `--output-dir` to choose another directory and `--run-name` to set the file prefix.

## Examples

MCCE-MHGCN:

```bash
python Link_Prediction.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --feat-key feat \
  --model mcce \
  --target-message-graph train \
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
  --negative-ratio 1.0 \
  --epochs 200 \
  --patience 10 \
  --early-stop-metric auc \
  --log-every 10
```

HAN:

```bash
python Link_Prediction.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model han \
  --metapaths author:coauthor:author,author:author_to_paper:paper>paper:paper_to_author:author,author:author_to_venue:venue>venue:venue_to_author:author \
  --num-heads 4 \
  --hidden-dim 128 \
  --predictor distmult \
  --epochs 200 \
  --log-every 10
```

MAGNN:

```bash
python Link_Prediction.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model magnn \
  --metapaths author:coauthor:author,author:author_to_paper:paper>paper:paper_to_author:author,author:author_to_venue:venue>venue:venue_to_author:author \
  --num-heads 4 \
  --magnn-rnn-type gru \
  --hidden-dim 128 \
  --predictor distmult \
  --epochs 200
```

HGT/RGCN/GTN/HetGNN do not require explicit metapaths:

```bash
python Link_Prediction.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model hgt \
  --num-heads 4 \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --predictor distmult \
  --epochs 200
```

HINormer:

```bash
python Link_Prediction.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model hinormer \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --hinormer-layers 2 \
  --num-heads 4 \
  --hinormer-beta 1.0 \
  --predictor distmult \
  --epochs 200 \
  --log-every 10
```

SimpleHGN:

```bash
python Link_Prediction.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model simplehgn \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --num-heads 4 \
  --edge-dim 64 \
  --simplehgn-beta 0.0 \
  --predictor distmult \
  --epochs 200 \
  --log-every 10
```

Node classification:

```bash
python Node_Classification.py \
  --graph-bin data/graph.bin \
  --target-ntype author \
  --model hgt \
  --num-heads 4 \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --classifier mlp \
  --classifier-hidden-dim 128 \
  --classifier-dropout 0.1 \
  --epochs 200 \
  --patience 20 \
  --early-stop-metric macro_f1 \
  --log-every 10
```

## Relation Selection

`--target-message-graph train` removes validation/test target edges from message passing. Use `--target-message-graph full` only for ablation.

Use `--use-etypes` to keep only selected message-passing edge types. For example, to remove same-type relations from message passing:

```bash
python Link_Prediction.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --model hgt \
  --use-etypes author_to_paper,paper_to_author,author_to_venue,venue_to_author,paper_to_venue,venue_to_paper \
  --hidden-dim 128 \
  --gnn-layers 2
```

## Important Parameters

```text
--graph-bin               DGL .bin file saved by dgl.save_graphs.
--graph-index             Graph index inside the .bin file. Default: 0.
--target-etype            Target edge type for link prediction.
--feat-key                Node feature key. Default: feat.
--model                   mcce, han, hgt, rgcn, magnn, hetgnn, gtn, hinormer, or simplehgn.
--gnn-layers              Encoder layer count.
--num-heads               Attention heads for attention-based models.
--metapaths               Explicit metapaths for MCCE/HAN/MAGNN.
--metapath-length         Maximum automatic metapath length.
--metapath-closure        closed, open, or both.
--magnn-rnn-type          MAGNN sequence encoder: gru, lstm, linear, or average.
--gtn-channels            Number of soft relation-selection channels for GTN.
--hinormer-layers         HINormer global attention layer count.
--hinormer-beta           Relation-bias weight for HINormer attention.
--edge-dim                Edge-type embedding dimension for SimpleHGN.
--simplehgn-beta          Edge-attention residual weight for SimpleHGN.
--predictor               distmult, dot, or mlp.
--target-message-graph    train or full.
--use-etypes              Edge types kept in the message graph, as rel or src:rel:dst.
--negative-ratio          Number of sampled negatives per positive edge.
--output-dir              Directory for saved metric CSV and summary JSON files. Default: outputs.
--run-name                Optional file name prefix for saved outputs.
```

Node classification also supports:

```text
--target-ntype            Target node type for classification.
--label-key               Node label key. Default: label.
--classifier              linear or mlp.
--classifier-hidden-dim   Hidden dimension for the MLP classifier.
--classifier-dropout      Dropout for the MLP classifier.
--early-stop-metric       loss, accuracy, macro_f1, or micro_f1.
```
