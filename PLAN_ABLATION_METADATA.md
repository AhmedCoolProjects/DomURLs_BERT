# PLAN — Metadata Category Ablation

## Why

Phase-1 SVM showed `meta+string` beats `string_only` by ~+9 macro-F1 — the metadata clearly carries signal. Before we wire metadata into the PLM, we want to know **which categories of metadata** are pulling the weight. That tells us:

- Which probes are worth keeping when scraping new domains (some are expensive — RDAP/CT can be slow).
- Whether one category dominates (cheap fix: just use that) or signal is distributed (need everything).
- Whether some categories are redundant given the others.

## Categories (verified against `train_meta.csv` header)

| Prefix    | # cols | What it covers                                    |
|-----------|-------:|---------------------------------------------------|
| `rdap_*`  |     44 | WHOIS / registration: age, expiration, NS, status |
| `dns_*`   |     44 | A/AAAA/MX/NS/CNAME records, TTLs, DNSSEC, DMARC   |
| `ip_*`    |     37 | Resolved IPs, ASN/country, hosting provider       |
| `tls_*`   |     33 | Cert issuer, age, SANs, validity                  |
| `ct_*`    |     16 | Certificate Transparency log activity             |
| `x_*`     |     16 | Cross-category derived features                   |

Plus 2 always-on flags (`input_valid`, `input_is_ip_address`).

## Ablation suite

One script invocation runs **all** of these and emits one wandb run per row, grouped together. Total: **13 SVM trains** per invocation (small — meta-only feature counts are ≤44, LinearSVC fits in seconds).

| Run name          | Features used                          | What it answers                          |
|-------------------|----------------------------------------|------------------------------------------|
| `full_meta`       | all 6 categories                       | Upper bound (reference)                  |
| `only_rdap`       | only rdap                              | What can RDAP do alone?                  |
| `only_dns`        | only dns                               | What can DNS alone?                      |
| `only_tls`        | only tls                               | …                                        |
| `only_ip`         | only ip                                | …                                        |
| `only_ct`         | only ct                                | …                                        |
| `only_x`          | only cross-signals                     | …                                        |
| `drop_rdap`       | all except rdap                        | What does losing RDAP cost?              |
| `drop_dns`        | all except dns                         | …                                        |
| `drop_tls`        | all except tls                         | …                                        |
| `drop_ip`         | all except ip                          | …                                        |
| `drop_ct`         | all except ct                          | …                                        |
| `drop_x`          | all except x                           | …                                        |

The `only_*` rows tell us **standalone power**; the `drop_*` rows tell us **unique marginal value** (high drop_X = X is hard to replace; low drop_X = redundant given the others). Together they're enough to make a keep/drop decision per category.

## Why pure metadata (no string features) for this pass

The string baseline is constant noise across rows — including it adds compute, hides smaller per-category effects, and isn't what we're trying to measure here. If a category looks promising, we can re-run `meta_X + string` separately to confirm it still helps on top of the string baseline.

## Wandb layout

- One **wandb group** per script invocation (timestamp-based name).
- One **wandb run** per row in the table above. `config.feature_set = full_meta / only_rdap / drop_rdap / …` so the runs table sorts/groups cleanly.
- Summary columns logged: `test/accuracy`, `test/precision_macro`, `test/recall_macro`, `test/f1_macro`, `test/auc`, `fit_seconds`, `n_features`.
- One trailing `summary` run holds a `wandb.Table` with all rows side-by-side + the `summary.csv`/`results.json` artifact.

In the wandb UI: group by `group`, sort by `test/f1_macro` descending, and you can read off the answer in 10 seconds.

## One command vs separate commands — decision

**One command.** Rationale:
- Data load + label encoding + scaler fit is identical for every row; running them once is much faster than 13 invocations.
- 13 LinearSVC fits on ≤44 metadata features is well under a minute total.
- All 13 wandb runs land in the same group automatically, no manual `--wandb_group` juggling.
- Failure isolation: each ablation is wrapped in try/except so one blowup doesn't kill the others.

If you ever want a single ablation in isolation (e.g. re-running just one), the script also accepts `--ablations only_dns,drop_dns` to subset.

## Implementation

New file: `experiments/svm_ablation.py`. Mirrors `svm_baseline.py` (same load/scale/LinearSVC/metric stack) but iterates over a list of `(name, category_mask)` specs instead of `meta/string/meta+string`.

## Command

```bash
python experiments/svm_ablation.py \
    --dataset ThreatFox_MalDomains \
    --experiment_type domain \
    --meta_dir /home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/domains/DomainsMetadata \
    --label_column label \
    --wandb_project DomURLs_BERT_metadata
```

Add `--ablations only_rdap,drop_rdap` to run a subset; add `--no_wandb` for a local-only smoke test.

## Decision gate (after running)

Look at the **sorted runs table** in wandb (sorted by `test/f1_macro` desc):

- If one category alone gets close to `full_meta` ⇒ that's the keep, others can be deprioritized.
- If `drop_X` for some X is far below `full_meta` ⇒ X is critical, keep.
- If `drop_X` ≈ `full_meta` for every X ⇒ heavy redundancy; we can drop the cheapest-to-collect one for free.

Outcome feeds directly into Phase 2: the metadata vector going into the PLM head should contain only the categories that survive this filter.
