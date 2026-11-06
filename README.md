# TensorGAT

TensorGAT represents a function with four aligned graph views: abstract syntax (AST), control flow (CFG), data dependence (DDG), and natural code sequence (NCS). The revised implementation follows the paper's model definition and evaluates Python--Java function-pair matching.

## Revision status

The original release used language-specific AST-type one-hot vectors even though the paper described CodeBERT node features. It also fixed the feature width below the combined vocabulary size, substituted AST edges for missing views, and could decompose disconnected graphs jointly in a mini-batch. The revised path corrects these mismatches:

- `codebert_features.py` produces frozen 768-dimensional contextual CodeBERT features aligned to graph nodes.
- `reproducibility/data.py` preserves AST/CFG/DDG/NCS semantics and accepts both archived `ddgEdges` and `dfgEdges` field names. NCS is rebuilt from consecutive Tree-sitter leaf tokens in source order; archived preorder chains are ignored.
- `reproducibility/model.py` applies t-SVD and CP independently to every graph.
- `reproducibility/make_class_splits.py` creates semantic-class-disjoint splits and deterministic within-split negatives.
- `reproducibility/train.py` runs the full model, TensorGCN, and the `no_cp`, `no_svd`, and `no_cp_no_svd` ablations under one protocol.

The root `TrainGAT.py`, `TrainGCN.py`, and `TrainBaseLine.py` files are retained only as historical scripts for old processed `.pt` files. They use the legacy one-hot format and are not the executable path for the revised paper.

## Installation

Create a Python 3.9 environment and install dependencies appropriate for the local CUDA version:

```bash
pip install -r requirements.txt
```

The reported environment used PyTorch 2.5.1 with CUDA 12.4, PyTorch Geometric 2.6.1, Transformers 4.46.3, and an NVIDIA RTX 4090. The CodeBERT checkpoint is `microsoft/codebert-base`.

## Reproducing the revised evaluation

See [`reproducibility/README.md`](reproducibility/README.md) for the complete sequence:

1. create class-disjoint pair manifests;
2. precompute frozen contextual CodeBERT node features;
3. cache per-graph t-SVD/CP preprocessing for each variant;
4. train seeds 1234, 2025, and 3407;
5. aggregate results and profile synchronized CUDA inference.

All generated features and graph caches are content-addressed or tied to a recorded configuration. Functions with more than 600 graph nodes are excluded rather than truncated.

## Task boundary

The revision evaluates cross-language function-pair matching. It does not implement single-function vulnerability detection or repository-level code understanding. The paper reports audited fixed-prompt LLM comparisons as a separate evaluation; this repository does not invoke remote model APIs or redistribute provider-side call logs.

## Data

The archived dataset package used for the revised experiments is publicly available at <https://drive.google.com/drive/folders/1aMq1cyCNnibq9v9DWRASwKtQN7u5LgUr?usp=drive_link>. The folder contains `dataset.rar` (6.21 GB) and `README.en.md`. Extract the archive, pass its root with `--datasets`, and use the released split-generation utilities or frozen JSONL manifests for the corresponding protocol.

The scripts record filtering counts, split assignments, configuration values, and alignment coverage so that a changed corpus cannot silently reproduce a different experiment under the same name. Original source links remain documented in the paper, and the graph-extraction utilities are available at <https://github.com/smallsuccful/Treetools.git>.
