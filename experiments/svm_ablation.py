"""Metadata category ablation: which prefixes (rdap/dns/tls/ip/ct/x) carry the signal?

For each ablation row (full_meta + 6 only_X + 6 drop_X), trains a LinearSVC on
just the matching metadata columns and logs metrics to wandb as its own run
(all sharing one group). See PLAN_ABLATION_METADATA.md.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent

# Columns in meta CSVs that are never features.
NON_FEATURE_COLS = {"input", "class", "label", "_error", "_elapsed_s"}

# Category prefix -> matcher. Order here defines the default ablation order.
CATEGORIES = ["rdap", "dns", "tls", "ip", "ct", "x"]
# Always-on baseline flags (not tied to a category).
ALWAYS_ON = ["input_valid", "input_is_ip_address"]

# Hand-picked combos for the second-round ablation (see PLAN_ABLATION_METADATA.md).
# Maps run_name -> set of category prefixes to KEEP.
COMBO_SPECS = {
    "drop_x_ct":         {"rdap", "dns", "tls", "ip"},        # drop the two zero-marginal cats
    "drop_x_ct_tls":     {"rdap", "dns", "ip"},               # drop the three weakest cats
    "keep_dns_ip":       {"dns", "ip"},                       # core duo
    "keep_dns_ip_rdap":  {"dns", "ip", "rdap"},               # core + rdap
    "keep_dns_ip_tls":   {"dns", "ip", "tls"},                # core + tls
}


def categorize(feature_cols):
    """Return {category: [columns]} plus a 'general' bucket for always-on flags."""
    buckets = {c: [] for c in CATEGORIES}
    general = []
    for col in feature_cols:
        if col in ALWAYS_ON:
            general.append(col)
            continue
        prefix = col.split("_", 1)[0]
        if prefix in buckets:
            buckets[prefix].append(col)
        # silently skip unrecognized prefixes
    return buckets, general


def build_ablation_specs():
    """List of (run_name, set-of-categories-to-include)."""
    specs = [("full_meta", set(CATEGORIES))]
    for cat in CATEGORIES:
        specs.append((f"only_{cat}", {cat}))
    for cat in CATEGORIES:
        specs.append((f"drop_{cat}", set(CATEGORIES) - {cat}))
    for name, cats in COMBO_SPECS.items():
        specs.append((name, set(cats)))
    return specs


def load_split(dataset_dir, meta_dir, split, label_column):
    df_data = pd.read_csv(dataset_dir / f"{split}.csv")
    df_meta = pd.read_csv(meta_dir / f"{split}_meta.csv")
    meta_drop = [c for c in ("class", "label") if c in df_meta.columns]
    df_meta_features = df_meta.drop(columns=meta_drop)
    merged = df_data.merge(df_meta_features, on="input", how="inner")
    if len(merged) != len(df_data):
        raise RuntimeError(f"[{split}] join lost rows: data={len(df_data)} joined={len(merged)}")

    feature_cols = [c for c in df_meta_features.columns if c not in NON_FEATURE_COLS]
    X = merged[feature_cols].to_numpy(dtype=np.float64)
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).sum())
        print(f"[{split}] WARNING: {n_nan} NaNs in metadata — filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    return merged, X, merged[label_column].values, feature_cols


def auc_or_nan(y_true, y_scores, n_classes):
    try:
        if n_classes == 2:
            scores = y_scores if y_scores.ndim == 1 else y_scores[:, 1]
            return float(roc_auc_score(y_true, scores))
        return float(roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro"))
    except Exception as e:
        print(f"  [auc] failed: {e}")
        return float("nan")


def compute_metrics(clf, X, y, n_classes):
    y_pred = clf.predict(X)
    y_scores = clf.decision_function(X)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    p_wt,  r_wt,  f1_wt,  _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy":            float(accuracy_score(y, y_pred)),
        "precision_macro":     float(p_mac),
        "recall_macro":        float(r_mac),
        "f1_macro":            float(f1_mac),
        "precision_weighted":  float(p_wt),
        "recall_weighted":     float(r_wt),
        "f1_weighted":         float(f1_wt),
        "auc":                 auc_or_nan(y, y_scores, n_classes),
        "y_pred":              y_pred,
    }


def main(args):
    dataset_dir = REPO_ROOT / "data" / f"{args.experiment_type}_datasets" / args.dataset
    meta_dir = Path(args.meta_dir)
    out_dir = REPO_ROOT / "experiments" / "results" / f"ablation_{args.dataset}_{args.label_column}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset_dir = {dataset_dir}")
    print(f"meta_dir    = {meta_dir}")
    print(f"label_col   = {args.label_column}")

    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    if args.no_wandb:
        print("wandb disabled by --no_wandb")
    elif not _WANDB_AVAILABLE:
        print("wandb not installed; skipping (pip install wandb to enable)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_name = args.wandb_group or f"ablation_{args.dataset}_{args.label_column}_{timestamp}"

    # --- Load + label encode once ---
    train_df, X_tr_all, y_tr_raw, feat_cols = load_split(dataset_dir, meta_dir, "train", args.label_column)
    dev_df,   X_dv_all, y_dv_raw, _         = load_split(dataset_dir, meta_dir, "dev",   args.label_column)
    test_df,  X_te_all, y_te_raw, _         = load_split(dataset_dir, meta_dir, "test",  args.label_column)
    print(f"sizes: train={len(train_df)} dev={len(dev_df)} test={len(test_df)} | total meta cols={len(feat_cols)}")

    buckets, general = categorize(feat_cols)
    for cat in CATEGORIES:
        print(f"  {cat:5s}: {len(buckets[cat]):3d} cols")
    print(f"  general (always on): {len(general)} cols -> {general}")

    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)
    y_dv = le.transform(y_dv_raw)
    y_te = le.transform(y_te_raw)
    label_names = [str(c) for c in le.classes_]
    n_classes = len(label_names)
    print(f"classes ({n_classes}): {label_names}\n")

    # Column-name -> index lookup for fast slicing.
    col_index = {col: i for i, col in enumerate(feat_cols)}

    # Pre-decide which ablation specs to actually run.
    all_specs = build_ablation_specs()
    if args.ablations:
        wanted = {name.strip() for name in args.ablations.split(",") if name.strip()}
        specs = [s for s in all_specs if s[0] in wanted]
        unknown = wanted - {s[0] for s in all_specs}
        if unknown:
            print(f"WARNING: unknown ablation names ignored: {unknown}")
    else:
        specs = all_specs

    base_config = {
        "dataset":         args.dataset,
        "experiment_type": args.experiment_type,
        "label_column":    args.label_column,
        "model":           "LinearSVC",
        "C":               args.C,
        "max_iter":        args.max_iter,
        "seed":            args.seed,
        "num_classes":     n_classes,
        "classes":         label_names,
        "n_train":         int(len(train_df)),
        "n_dev":           int(len(dev_df)),
        "n_test":          int(len(test_df)),
    }

    summary_rows = []
    for name, cats in specs:
        # Build the column subset for this ablation.
        cols = list(general)
        for cat in CATEGORIES:
            if cat in cats:
                cols.extend(buckets[cat])
        if not cols:
            print(f"[{name}] skipped: no columns selected")
            continue

        idx = np.array([col_index[c] for c in cols], dtype=np.int64)
        Xtr, Xdv, Xte = X_tr_all[:, idx], X_dv_all[:, idx], X_te_all[:, idx]

        # Scale on train.
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xdv_s = scaler.transform(Xdv)
        Xte_s = scaler.transform(Xte)

        cats_sorted = sorted(cats)
        print(f"--- [{name}] cats={cats_sorted} | n_features={Xtr_s.shape[1]} ---")
        try:
            t0 = time.time()
            clf = LinearSVC(C=args.C, max_iter=args.max_iter, dual="auto", random_state=args.seed)
            clf.fit(Xtr_s, y_tr)
            fit_seconds = time.time() - t0
            dev_m = compute_metrics(clf, Xdv_s, y_dv, n_classes)
            test_m = compute_metrics(clf, Xte_s, y_te, n_classes)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            continue

        print(
            f"  fit={fit_seconds:.1f}s  "
            f"test: acc={test_m['accuracy']:.4f}  prec={test_m['precision_macro']:.4f}  "
            f"rec={test_m['recall_macro']:.4f}  f1={test_m['f1_macro']:.4f}  auc={test_m['auc']:.4f}"
        )

        summary_rows.append({
            "ablation":             name,
            "categories":           ",".join(cats_sorted),
            "n_features":           int(Xtr_s.shape[1]),
            "fit_seconds":          round(fit_seconds, 2),
            "dev_accuracy":         dev_m["accuracy"],
            "dev_precision_macro":  dev_m["precision_macro"],
            "dev_recall_macro":     dev_m["recall_macro"],
            "dev_f1_macro":         dev_m["f1_macro"],
            "dev_auc":              dev_m["auc"],
            "test_accuracy":        test_m["accuracy"],
            "test_precision_macro": test_m["precision_macro"],
            "test_recall_macro":    test_m["recall_macro"],
            "test_f1_macro":        test_m["f1_macro"],
            "test_f1_weighted":     test_m["f1_weighted"],
            "test_auc":             test_m["auc"],
        })

        if use_wandb:
            run_config = dict(base_config)
            run_config.update({
                "ablation":      name,
                "categories":    cats_sorted,
                "n_features":    int(Xtr_s.shape[1]),
                "fit_seconds":   fit_seconds,
            })
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=group_name,
                job_type=name,
                name=f"{name}__{timestamp}",
                config=run_config,
                tags=["svm", "ablation", args.dataset, args.label_column, name],
            )
            for split_name, m in (("dev", dev_m), ("test", test_m)):
                for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                          "precision_weighted", "recall_weighted", "f1_weighted", "auc"):
                    run.summary[f"{split_name}/{k}"] = m[k]
            run.summary["fit_seconds"] = fit_seconds
            run.summary["n_features"] = int(Xtr_s.shape[1])

            # Per-class on test.
            per_class = precision_recall_fscore_support(
                y_te, test_m["y_pred"], labels=list(range(n_classes)), zero_division=0
            )
            for cls_idx, cls_name in enumerate(label_names):
                run.summary[f"test/per_class/{cls_name}/precision"] = float(per_class[0][cls_idx])
                run.summary[f"test/per_class/{cls_name}/recall"]    = float(per_class[1][cls_idx])
                run.summary[f"test/per_class/{cls_name}/f1"]        = float(per_class[2][cls_idx])
                run.summary[f"test/per_class/{cls_name}/support"]   = int(per_class[3][cls_idx])
            run.finish()

    summary = pd.DataFrame(summary_rows).sort_values("test_f1_macro", ascending=False)
    print("\n=== ABLATION SUMMARY (sorted by test_f1_macro desc) ===")
    print(summary.to_string(index=False))

    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nwrote: {out_dir / 'summary.csv'}")
    print(f"wrote: {out_dir / 'summary.json'}")

    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=group_name,
            job_type="summary",
            name=f"summary__{timestamp}",
            config=base_config,
            tags=["svm", "ablation", args.dataset, args.label_column, "summary"],
        )
        run.log({"ablation_table": wandb.Table(dataframe=summary)})
        artifact = wandb.Artifact(
            name=f"svm-ablation-{args.dataset}-{args.label_column}",
            type="results",
            metadata={"dataset": args.dataset, "label_column": args.label_column,
                      "group": group_name},
        )
        artifact.add_file(str(out_dir / "summary.csv"))
        artifact.add_file(str(out_dir / "summary.json"))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metadata category ablation for the SVM probe.")
    parser.add_argument("--dataset", type=str, default="ThreatFox_MalDomains")
    parser.add_argument("--meta_dir", type=str, required=True)
    parser.add_argument("--experiment_type", type=str, default="domain", choices=["domain", "url"])
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ablations", type=str, default=None,
                        help="Comma-separated subset of ablation names "
                             "(default: run all 13). E.g. 'full_meta,only_rdap,drop_rdap'")
    parser.add_argument("--wandb_project", type=str, default="DomURLs_BERT_metadata")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    main(args)
