# Independent Baseline Entrypoints

This folder contains one runnable entrypoint per baseline model. All entrypoints share the same DGL `.bin` loading, edge masks, dynamic negative sampling, predictor choices, training loop, validation loop, and final test evaluation from `Baselines/common.py`.

Run commands from the repository root:

```bash
python -u Baselines/HAN.py --graph-bin data/graph.bin --target-etype author:coauthor:author
python -u Baselines/MAGNN.py --graph-bin data/graph.bin --target-etype author:coauthor:author
python -u Baselines/HetGNN.py --graph-bin data/graph.bin --target-etype author:coauthor:author
python -u Baselines/HGT.py --graph-bin data/graph.bin --target-etype author:coauthor:author
python -u Baselines/RGCN.py --graph-bin data/graph.bin --target-etype author:coauthor:author
python -u Baselines/GTN.py --graph-bin data/graph.bin --target-etype author:coauthor:author
python -u Baselines/HINormer.py --graph-bin data/graph.bin --target-etype author:coauthor:author
python -u Baselines/SimpleHGN.py --graph-bin data/graph.bin --target-etype author:coauthor:author
```

## Metapath-based Models

`HAN.py` and `MAGNN.py` require closed metapaths. For your author/paper/venue graph, a typical command is:

```bash
python -u Baselines/HAN.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --metapaths author:coauthor:author,author:author_to_paper:paper>paper:paper_to_author:author,author:author_to_venue:venue>venue:venue_to_author:author \
  --num-heads 4 \
  --hidden-dim 128 \
  --predictor distmult \
  --epochs 200 \
  --log-every 10
```

MAGNN uses the same metapath format and additionally supports `--magnn-rnn-type gru|lstm|linear|average`.

## Graph-wide Message-passing Models

`HGT.py`, `RGCN.py`, `GTN.py`, `HetGNN.py`, `HINormer.py`, and `SimpleHGN.py` do not require explicit metapaths. They use DGL canonical edge types from the message graph. Use `--use-etypes` when you want to remove same-type relations or keep only selected heterogeneous relations.

Example without same-type message-passing relations:

```bash
python -u Baselines/HGT.py \
  --graph-bin data/graph.bin \
  --target-etype author:coauthor:author \
  --use-etypes author_to_paper,paper_to_author,author_to_venue,venue_to_author,paper_to_venue,venue_to_paper \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --num-heads 4 \
  --predictor distmult \
  --epochs 200 \
  --log-every 10
```

## Model-specific Parameters

```text
HAN.py        --num-heads, --metapaths
MAGNN.py      --num-heads, --metapaths, --magnn-rnn-type
HGT.py        --num-heads, --gnn-layers
RGCN.py       --gnn-layers
GTN.py        --gnn-layers, --gtn-channels
HetGNN.py     --gnn-layers
HINormer.py   --gnn-layers, --hinormer-layers, --hinormer-beta, --num-heads
SimpleHGN.py  --gnn-layers, --num-heads, --edge-dim, --slope, --simplehgn-beta
```

The old `Train_Evaluate.py --model ...` route is kept for compatibility, but the recommended baseline route is now the independent file under this `Baselines/` directory.
