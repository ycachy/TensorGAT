# Reproducible TensorGAT rerun

This directory implements the model and protocol described in the revised paper. It keeps the GitHub architecture's four graph views, cyclic cross-view propagation, t-SVD preprocessing, CP structural aggregation, three GRU rounds, and Set2Set pooling. It changes the released fixed-width one-hot node input to frozen contextual CodeBERT features so the executable model matches the paper. The default pair classifier remains concatenation plus an MLP; the optional small-data mode adds shared graph-encoder weights and explicit absolute-difference/product interactions.

## Corrections

- `matrixFormation.py` no longer assumes a 173-dimensional vocabulary and maps unknown legacy node types to an explicit UNK column.
- `codebert_features.py` encodes complete source files in overlapping windows, aligns Tree-sitter nodes to source offsets, pools contextual subwords, and uses a deterministic type-only fallback.
- `data.py` reads DDG edges from `ddgEdges` (or the legacy `dfgEdges` name) and rebuilds NCS from consecutive Tree-sitter leaf tokens in source order instead of chaining AST preorder ids.
- `model.py` computes t-SVD and CP data per graph. `prepare_graphs.py` caches those graph-only results before mini-batch training, which prevents batch composition from changing a graph's decomposition.
- `make_class_splits.py` creates semantic-class-disjoint train/validation/test partitions and auditable within-split negative pairs.
- `train.py` selects checkpoints by validation ROC-AUC and freezes a validation-F1 threshold before internal and external testing. It also records results at threshold 0.5.
- For balanced data with few semantic families, `train.py` can resample a leakage-safe difficulty-matched negative for every training anchor and epoch, keep each positive/negative pair in one batch, and combine BCE with a within-anchor ranking loss. n1 samples within an operation family, n2 within a CWE across operation families, and n3 across CWEs. `--shared-encoder --pair-interactions` compares graph embeddings in a shared space without changing the four-view TensorGAT encoder.

## Data package

Download the public archive from <https://drive.google.com/drive/folders/1aMq1cyCNnibq9v9DWRASwKtQN7u5LgUr?usp=drive_link>. The folder contains `dataset.rar` (6.21 GB) and `README.en.md`. Extract the archive and set `$datasets` in the commands below to the extracted dataset root. Split manifests and generated caches remain separate outputs so the archive can be checked independently of model training.
## Environment

The reported rerun used Python 3.9, CUDA 12.4, an NVIDIA RTX 4090, and the versions in `requirements.txt`. CodeBERT is frozen; cached features must be regenerated when source bytes, parser output, or the model checkpoint changes.

## Pipeline

Run commands from the repository root. The examples use PowerShell variables only to keep paths readable.

```powershell
$datasets = '<dataset-root>'
$splits = '<output>\splits'
$features = '<output>\feature_cache'
$graphs = '<output>\graph_cache'
$runs = '<output>\runs'

python reproducibility/make_class_splits.py `
  --corpus TransLib --source '<pair-manifest-directory>' --output $splits

python reproducibility/precompute_codebert.py `
  --datasets $datasets --corpus TransLib `
  --manifest "$splits\TransLib_python-java_train.jsonl" `
  --manifest "$splits\TransLib_python-java_val.jsonl" `
  --manifest "$splits\TransLib_python-java_test.jsonl" `
  --cache $features --index "$features\translib_python-java.json"

python reproducibility/prepare_graphs.py `
  --datasets $datasets --corpus TransLib `
  --manifest "$splits\TransLib_python-java_train.jsonl" `
  --manifest "$splits\TransLib_python-java_val.jsonl" `
  --manifest "$splits\TransLib_python-java_test.jsonl" `
  --variant full --output $graphs
```

Repeat `prepare_graphs.py` with `--variant no_cp`, `no_svd`, `no_cp_no_svd`, and `gcn`. The `no_cp_no_svd` variant retains GAT propagation while removing both low-rank components. Prepare the external corpus with its frozen manifest in the same way. Then run each variant for seeds 1234, 2025, and 3407:

```powershell
python reproducibility/train.py `
  --datasets $datasets --corpus TransLib `
  --train "$splits\TransLib_python-java_train.jsonl" `
  --val "$splits\TransLib_python-java_val.jsonl" `
  --test "$splits\TransLib_python-java_test.jsonl" `
  --features "$features\translib_python-java.json" `
  --prepared '<variant-graph-cache>\index.json' `
  --variant full --seed 1234 --output $runs `
  --external-corpus AWS --external-test '<aws-manifest>.jsonl' `
  --external-features "$features\aws_python-java.json" `
  --external-prepared '<aws-variant-graph-cache>\index.json'

python reproducibility/aggregate_results.py --runs $runs --output '<output>\results_summary.json'
```

For a small-family protocol, add the following training options. Dynamic candidates are drawn only from positive C fragments already assigned to the training split, so validation and test families remain isolated.

```powershell
  --dynamic-negatives --rank-weight 0.5 `
  --shared-encoder --pair-interactions `
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 `
  --epochs 60 --patience 8 --min-auc-improvement 1e-4
```

Use `profile_inference.py` with a saved checkpoint to measure synchronized CUDA inference time and peak allocated memory. Every JSON run record contains the full configuration, selected threshold, split sizes, excluded pair counts, epoch history, logits, labels, and internal/external metrics.

For a frozen query--positive--negative manifest, `evaluate_ranking_diagnostic.py` loads an existing checkpoint and its validation-selected threshold without fitting or tuning on the diagnostic set. It reports pair metrics, within-query ranking, low-lexical positive accuracy, high-lexical negative false-positive rate, and the subset in which identifier/number Jaccard favors the negative candidate. Run it once for each variant and seed, then aggregate the JSON outputs with `summarize_ranking_diagnostic.py`:

```powershell
python reproducibility/evaluate_ranking_diagnostic.py `
  --datasets $datasets --corpus AWS `
  --manifest '<frozen-diagnostic>.jsonl' `
  --features '<aws-feature-index>.json' `
  --prepared '<aws-variant-graph-cache>\index.json' `
  --checkpoint '<runs>\full_seed1234.pt' `
  --training-result '<runs>\full_seed1234.json' `
  --variant full --seed 1234 `
  --output '<diagnostic-results>\full_seed1234.json'

python reproducibility/summarize_ranking_diagnostic.py `
  --results '<diagnostic-results>' --output-dir '<diagnostic-summary>'
```

## Scope

This pipeline evaluates cross-language function-pair matching and does not implement single-function vulnerability detection. The paper reports fixed-prompt LLM comparisons separately; these local scripts do not invoke remote model APIs or reproduce provider-side calls.
