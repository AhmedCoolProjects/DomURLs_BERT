# CONTEXT — Metadata-Augmented Domain Classification

## Goal

The current framework (`main_plm.py` / `main_charnn.py`) classifies a domain from its **string only** (`input` column). For each domain in `data/domain_datasets/ThreatFox_MalDomains/{train,dev,test}.csv` we now also have a **scraped metadata vector** (~185 numeric features: RDAP, DNS, TLS, IP/hosting, CT logs, cross-signals) stored in `/Users/bargadym1max/Desktop/agents/domains/metadata/{train,dev,test}_meta.csv`.

The question we want to answer:

> **Does this metadata actually move the needle on classification, or does the domain string alone already carry most of the signal?**

Plan: answer it cheaply with an SVM **before** spending time wiring metadata into the BERT/CharNN training loop.

## Why SVM first

- Fast to train on tabular features, no GPU needed.
- Gives a clean read on the metadata's raw discriminative power, independent of the PLM.
- Three baselines to compare:
  1. **SVM on metadata only** — pure signal from scraped features.
  2. **SVM on simple string features** (e.g. char n-gram TF-IDF on `input`) — string-only baseline.
  3. **SVM on metadata + string features** — does concatenation help?
- If (1)/(3) clearly beat (2), it justifies modifying the framework to fuse metadata into the PLM. If not, we save the integration work.

## Data layout & alignment

- The meta CSVs have identical row count and the same `input,class,label` columns as the dataset CSVs, plus ~185 numeric `rdap_* / dns_* / tls_* / ip_* / ct_* / x_*` columns and bookkeeping columns `_error`, `_elapsed_s`.
- Row count matches exactly (train 105 640, dev 35 214, test 35 214). We'll still **join on `input`** (not row index) to be safe against any reordering, and assert row counts after the merge.
- Feature columns to drop: `_error`, `_elapsed_s`, `input_valid` (constant 1 in practice — verify).
- For the binary task use `label` (legit/malicious); for multiclass use `class`.

## Working plan

### Phase 1 — SVM probe (this branch, no framework changes)

A standalone script `experiments/svm_baseline.py` that:

1. Loads `train/dev/test.csv` (labels) and joins each with `train/dev/test_meta.csv` on `input`.
2. Builds three feature sets per split: `meta_only`, `string_only`, `meta_plus_string`.
3. Scales metadata (`StandardScaler`), TF-IDFs the domain string at the char level (e.g. `analyzer='char_wb', ngram_range=(3,5)`).
4. Trains `LinearSVC` for each feature set (linear scales to 100k rows; RBF would not). Optionally `SGDClassifier(loss='hinge')` if LinearSVC is slow.
5. Reports accuracy / macro-F1 / per-class F1 on **dev** for tuning and **test** for the final number. Saves a results table.

Run:
```bash
python experiments/svm_baseline.py \
    --dataset ThreatFox_MalDomains \
    --meta_dir /Users/bargadym1max/Desktop/agents/domains/metadata \
    --label_column label
```

**Decision gate:** look at the test macro-F1 of `meta_plus_string` vs `string_only`. A material gap (e.g. ≥1–2 macro-F1 points) ⇒ proceed to Phase 2.

### Phase 2 — Fuse metadata into the framework (only if Phase 1 looks promising)

Sketch (not implemented yet):

- Extend `data_utils/load_data.py` to optionally load the matching `*_meta.csv` and align by `input`.
- Extend `data_utils/bertdataset.py` (and the CharNN dataset) to return `(tokens, meta_vector, label)`.
- Wrap the classifier head: PLM `[CLS]` embedding ⊕ a small MLP over the (scaled) metadata vector → fused representation → linear classifier. Persist the metadata `StandardScaler` alongside the checkpoint.
- Add CLI flags: `--use_metadata`, `--metadata_dir`, `--metadata_dim`.

## Testing checklist (per phase)

Phase 1:
- [ ] Row counts match dataset and meta CSVs for each split.
- [ ] Inner join on `input` does not drop rows (assert).
- [ ] No NaNs in the final feature matrix (the meta files use 0 for missing flags, but verify).
- [ ] Label encoder fit only on train.
- [ ] Print test results for all three feature sets in one table.

Phase 2 (later):
- [ ] Dataloader yields a batch with the expected shapes including `meta_vector`.
- [ ] One-epoch overfit on a tiny subset still works.
- [ ] `--use_metadata` off → numerically equivalent to current `main_plm.py`.
- [ ] Metadata scaler is saved with the checkpoint and reused at eval time.

## Out of scope for now

- Other datasets in `data/domain_datasets/` (we only have meta for ThreatFox).
- Tuning the PLM. We want a clean A/B on metadata, not a new SOTA run.
- Heavy hyperparameter search on the SVM — the goal is signal, not the best model.
