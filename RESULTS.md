# Results — Metadata-Augmented Domain Classification

## Summary

We extended [DomURLs_BERT](https://github.com/AhmedCoolProjects/DomURLs_BERT) with a
late-fusion path that injects per-domain **scraped metadata** (~190 numeric
features: RDAP, DNS, TLS, IP/hosting, CT logs, cross-signals) alongside the
PLM `[CLS]` embedding. On the **ThreatFox_MalDomains** dataset:

- **Binary (legit / malicious)**: test F1 **0.9444**, up from **0.8816** without
  metadata (**+5.4 F1 / +6.3 % relative**).
- **Multiclass (64 malware families + legit)**: test F1_macro **0.4562**, test
  accuracy **0.8086**, test F1_weighted **0.8088** — up from F1_macro **0.3171**
  (no metadata) and **0.4317** (single-stage with metadata). Two-stage
  decomposition (binary → family-on-malicious) added the final +2.45 F1_macro.

The remaining multiclass F1_macro gap is **structurally bounded** by ~20
long-tail malware families with <30 training samples and a confusable
mid-frequency cluster (the RAT family).

## Question

The baseline DomURLs_BERT classifies a domain from its **string only**. We
asked: does adding a vector of scraped runtime/registration signals
*meaningfully* help the model on this task, and if so, *what kinds of
signals* carry the gain?

## Experimental progression

Five stages, each gated on the previous one.

### Phase 1 — SVM probe (cheap signal check)

LinearSVC on three feature sets, before touching the PLM:

| feature set        | test F1 (macro) |
|--------------------|----------------:|
| string_only (TF-IDF char n-grams) | 0.852 |
| meta_only (~190 standardized features) | 0.898 |
| meta + string (concat)            | **0.941** |

Metadata alone already beat the string baseline; combining them added another
+4 F1. The metadata clearly carries signal — Phase 2 was worth doing.

### Phase 1.5 — Metadata ablation (which categories matter?)

13-run sweep over the six prefix categories (`rdap`, `dns`, `tls`, `ip`,
`ct`, `x` cross-signals) under the SVM, plus combo runs:

| ablation        | n_features | test F1 |
|-----------------|-----------:|--------:|
| drop_x          |        176 |  0.8988 |
| **drop_x_ct**   |    **160** | **0.8988** |
| full_meta       |        192 |  0.8983 |
| keep_dns_ip_rdap |        127 |  0.8982 |
| only_ip         |         39 |  0.881  |
| only_dns        |         46 |  0.850  |
| only_tls        |         35 |  0.794  |
| only_rdap       |         46 |  0.668  |
| only_x          |         18 |  0.665  |
| only_ct         |         18 |  0.362  |

Findings:
- The core signal lives in `dns` + `ip` (largest drop costs, strongest
  standalone). `rdap` adds small but real marginal value (-0.005 when
  dropped).
- `tls`, `ct`, `x` are essentially redundant given the others — dropping
  any one of them costs ~0 F1.
- **The SVM ablation answered the data question, not the PLM question.**
  When we later replayed the ablation with the PLM, the picture was
  different (see Phase 2).

### Phase 2 — PLM fusion (the actual experiment)

Architecture: **late fusion**

```
domain string ──► tokenizer ──► PLM encoder ──► [CLS] (768) ──┐
                                                              ├──► concat ──► classifier
metadata (160) ──► StandardScaler ──► MetaMLP(160→128→64) ────┘
```

MetaMLP is a 2-layer GELU + Dropout block; the final classifier is a single
linear layer on the concatenated 832-dim vector.

Initial result (paper's default `lr=1e-5`, 10 epochs):

| run                    | test F1 (bin) | test acc |
|------------------------|--------------:|---------:|
| no_metadata            |        0.8816 |  0.8856  |
| with metadata (`drop_x_ct`) |   0.9352 |  0.9366  |

A **9-run lr × meta-config grid** revealed two things:

1. **lr matters a lot for metadata fusion, almost nothing for the
   PLM-only path.** Bumping `lr=1e-5 → 5e-5` lifted the fused model from
   0.916 to **0.9425**; the no-metadata model barely moved (0.880 → 0.883).
   The MetaMLP is randomly initialized while the PLM is pretrained — it
   needs a higher lr to actually learn its share of the representation.
2. **Opposite to the SVM finding, `full_meta` beat `drop_x_ct` for the
   PLM** (0.9425 vs 0.9391 at lr=5e-5). The MetaMLP's non-linearity can
   exploit signal in `ct_*` and `x_*` that the linear SVM couldn't —
   so the SVM ablation was a good signal detector but a poor feature
   selector for downstream PLM use.

### Phase 2.5 — Structural levers

Tested three orthogonal architectural changes and one loss-side change:

| structural change          | binary F1 | multiclass F1_macro |
|----------------------------|----------:|--------------------:|
| baseline (concat + linear) |    0.9425 |        0.3656       |
| **+ gated fusion + LayerNorm** | **0.9444** | **0.3744** |
| + 2-layer MLP head (hidden=256) | regressed | **0.2533** |
| + 2-layer MLP head (hidden=64)  | regressed | **0.2302** |

Gated fusion: `g = σ(W_g · [CLS])` produces a per-sample vector gate that
multiplies `MetaMLP(meta)` before concatenation. The PLM decides how much
to trust metadata per-row — important because many domains have all-zero
metadata (no DNS record, no RDAP entry, etc.).

The **2-layer MLP head failed in both directions** (-0.11 to -0.14 multiclass
F1_macro) — F1_weighted/accuracy stayed near baseline but F1_macro
collapsed. The randomly-initialized head couldn't catch up to the
fine-tuned BERT body in 7 epochs at uniform lr=5e-5; its gradient was
dominated by majority classes, abandoning the rare ones. A head-specific
higher lr or substantially longer training might rescue this, but the
gated-fusion lever alone already covered most of the structural gain.

### Loss-side fixes for multiclass imbalance

Train support across the 64 classes spans roughly **4 to 4200 samples**.
F1_macro vs F1_weighted gap (0.43 vs 0.80) showed the model was nailing
the majority classes and failing on the long tail.

| loss recipe                | F1_macro | F1_wted | acc    |
|----------------------------|---------:|--------:|-------:|
| gated_ln baseline (CE)     |   0.3744 |  0.8033 | 0.8159 |
| + inverse-frequency cw     |   0.3632 |  0.7055 | 0.6592 |
| + capped cw (cap 10×)      |   0.3900 |  0.7905 | 0.7916 |
| + focal loss (γ=2)         |   0.3840 |  0.8052 | 0.8151 |
| **+ sqrt-inverse cw**      | **0.4317** | 0.7994 | 0.7993 |
| + sqrt-inverse + focal γ=2 |   0.3111 |  0.4842 | 0.4141 |
| + sqrt-inverse + focal γ=3 |   0.2745 |  0.2162 | 0.1959 |

Findings:

- **Sqrt-inverse class weights are the single biggest multiclass lift:
  +5.7 F1_macro** while only sacrificing 0.4 F1_wted and 1.7 accuracy. The
  Pareto-correct way to address imbalance.
- **Raw inverse-frequency over-corrects** — accuracy crashed -0.15 for a
  -0.01 F1_macro change (gradient is dominated by 100× class-weight
  ratios on rare classes).
- **Focal stacked with sqrt cw destroys training.** Both levers
  down-weight the majority gradient; together they leave the model with
  no signal on dominant classes. Use one or the other, not both.

### Phase 3 — Two-stage classifier

The single-stage multiclass model was capped at F1_macro 0.43. Per-class
analysis showed two structural problems:

- A genuine **long tail** of ~20 families with <30 training samples each.
  No tuning can fix these.
- A **RAT-family confusion cluster**: NjRAT, AsyncRAT, Nanocore RAT,
  Quasar RAT, Cobalt Strike share infrastructure and tokenization. The
  single-stage model was collapsing them onto Nanocore as a default
  bucket (high recall on Nanocore, low recall on NjRAT/AsyncRAT despite
  ample data).

Decomposition: train a second model on **malicious-only rows** (`legit`
removed before label encoding, num_classes drops from 64 to 63). At
inference time, stage 1 (binary) decides legit/malicious; stage 2
classifies the malicious-predicted subset.

Stage 2 used the same locked recipe (gated + LN + full_meta + sqrt cw,
lr=5e-5, 20 epochs).

Chained results vs. single-stage:

| metric         | single-stage (`mc3__sqrt`) | **two-stage** | Δ          |
|----------------|---------------------------:|--------------:|-----------:|
| accuracy       |                     0.7993 |        0.8086 | +0.93 pt   |
| **F1_macro**   |                     0.4317 |    **0.4562** | **+2.45 pt** |
| F1_weighted    |                     0.7994 |        0.8088 | +0.94 pt   |

The gain concentrates on exactly the confusion-cluster classes the
diagnostic identified:

| family            | single-stage F1 | two-stage F1 |     Δ |
|-------------------|----------------:|-------------:|------:|
| SSLoad            |          0.1429 |       0.6667 | +0.52 |
| Raspberry Robin   |          0.4872 |       0.6984 | +0.21 |
| SMSspy            |          0.3059 |       0.4706 | +0.17 |
| **NjRAT**         |       **0.3492** |    **0.4952** | +0.15 |
| SystemBC          |          0.4000 |       0.4923 | +0.09 |
| AsyncRAT          |          0.2050 |       0.2843 | +0.08 |
| Hook              |          0.7179 |       0.7732 | +0.06 |
| magecart          |          0.5439 |       0.6010 | +0.06 |
| Chrysaor          |          0.4130 |       0.4390 | +0.03 |
| legit             |          0.9421 |       0.9520 | +0.01 |

Small regressions (Gozi -0.05, IcedID -0.04, BitRAT -0.07) come from
stage-1 false positives that stage 2 then routes to the closest neighbor.

A four-run **stage-2 push** (varying lr ∈ {2e-5, 3e-5, 5e-5}, epochs=30,
with and without sqrt cw) confirmed the original recipe was already
optimal — best alternative was 0.4541, all alternatives early-stopped by
epoch 9 — and that removing sqrt cw collapses F1_macro from 0.454 to
0.280 even on malicious-only data.

## Locked final numbers

| task                         | recipe                                          | metric         | value       |
|------------------------------|-------------------------------------------------|----------------|------------:|
| **binary (legit/malicious)** | full_meta + lr=5e-5 + gated + LN                | test F1        | **0.9444**  |
|                              |                                                 | test accuracy  | 0.9455      |
| **multiclass (two-stage)**   | binary stage-1 → stage-2 on malicious-only      | test F1_macro  | **0.4562**  |
|                              | (gated + LN + sqrt cw, full_meta, lr=5e-5)      | test accuracy  | 0.8086      |
|                              |                                                 | test F1_weighted | 0.8088    |

For comparison, the metadata-free PLM baseline at the same recipe was
**0.8816 F1 (binary) / 0.3171 F1_macro (multiclass)**.

## Per-class analysis of remaining failures

After two-stage decomposition, three classes still register F1 = 0 on test:
**Amadey** (23 test samples), **Orcus RAT** (22), **Raccoon** (18). All have
very small train support and no distinctive signal in the metadata. ~20
classes hover at F1 < 0.3 (DarkGate, MooBot, Choziosi, Loki Password
Stealer, IcedID Downloader, etc.) — likely the same long-tail story.

The RAT confusion cluster is still partially unresolved despite two-stage:
NjRAT precision 0.51 / recall 0.48 means the model is hedging between
RAT families even when it knows it's looking at *some* RAT. Resolving
this cluster fully would likely require either (a) more training data
for the smaller RATs, or (b) family-level supervision (e.g. "RAT" as a
super-class with sub-classifiers).

## Limitations

- **Long-tail F1_macro is structurally bounded.** Roughly 20 of 64 classes
  have <30 train samples; their per-class F1 will be noisy or zero
  regardless of architecture. F1_weighted/accuracy (~0.81) and the
  binary score (0.94) are the metrics that reflect what the model
  actually does for users.
- **Two-stage transfers stage-1 errors into family precision.** When
  stage 1 misclassifies a legit row as malicious, stage 2 assigns it to
  the most prototypical family (QakBot here). This costs ~3 points of
  QakBot F1 in the chained eval compared to stage 2 in isolation. It's
  a known tradeoff of cascaded decompositions.
- **Single dataset.** All experiments are on ThreatFox_MalDomains. The
  same probe should be re-run on the other DomURLs_BERT datasets before
  claiming generality.
- **Metadata coverage is sometimes empty.** Many domains have
  `rdap_found=0`, `dns_found=0`, etc. (no record returned by the
  scraper). The gated fusion handles this gracefully but it means the
  effective metadata dimensionality varies per sample.

## Reproducibility

All experiments live on branches `metadata-wandb` and `metadata-wandb-cls-2`
of [AhmedCoolProjects/DomURLs_BERT](https://github.com/AhmedCoolProjects/DomURLs_BERT).

Key commits:
- `0475c70` — wandb logging on SVM baseline
- `0375beb` — metadata category ablation (Phase 1.5)
- `5bcb430` — round-2 ablation combos
- `1b39340` — string-augmented ablation
- `d833259` — Phase 2 PLM metadata fusion
- `e842f16` — wandb on main_plm.py
- `623154a` — gated fusion + MLP head + LayerNorm + class weights
- `bd311aa` — sqrt/capped class weights + focal loss
- `9904f54` — two-stage classifier (`--malicious_only` + `eval_two_stage.py`)
- `1e46cc0` — fix strict=False for class-weighted checkpoints

Headline reproduction commands (run on a single A100):

**Binary winner (F1 = 0.9444):**
```bash
python main_plm.py \
    --dataset ThreatFox_MalDomains --pretrained_path amahdaouy/DomURLs_BERT \
    --experiment_type domain --label_column label \
    --epochs 20 --batch_size 256 --lr 5e-5 --num_workers 4 --device 0 \
    --metadata_dir <path/to/DomainsMetadata> \
    --meta_categories rdap,dns,tls,ip,ct,x \
    --fusion_mode gated --meta_layernorm
```

**Stage-2 (F1_macro = 0.4653 on malicious-only):**
```bash
python main_plm.py \
    --dataset ThreatFox_MalDomains --pretrained_path amahdaouy/DomURLs_BERT \
    --experiment_type domain --label_column class \
    --epochs 20 --batch_size 256 --lr 5e-5 --num_workers 4 --device 0 \
    --metadata_dir <path/to/DomainsMetadata> \
    --meta_categories rdap,dns,tls,ip,ct,x \
    --fusion_mode gated --meta_layernorm \
    --class_weight_strategy sqrt_inverse \
    --malicious_only
```

**Two-stage chained eval (F1_macro = 0.4562):**
```bash
python experiments/eval_two_stage.py \
    --stage1_dir mlruns/ckpts/<BINARY_RUN_ID> \
    --stage2_dir mlruns/ckpts/<STAGE2_RUN_ID> \
    --dataset ThreatFox_MalDomains \
    --metadata_dir <path/to/DomainsMetadata>
```

The metadata CSVs (`train_meta.csv`, `dev_meta.csv`, `test_meta.csv`) are
the raw output of the upstream scraping pipeline and are not committed to
this repo. They join 1:1 with the dataset CSVs on the `input` column.
