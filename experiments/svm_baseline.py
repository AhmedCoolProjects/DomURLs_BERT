"""SVM probe: does the scraped metadata add signal over the domain string alone?

Trains LinearSVC on three feature sets — metadata-only, string-only (char n-gram
TF-IDF), and the concatenation — and reports dev/test metrics for each. See
CONTEXT.md for the full plan.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
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

# Columns in the meta CSVs that are not features.
NON_FEATURE_COLS = {"input", "class", "label", "_error", "_elapsed_s"}


def load_split(dataset_dir: Path, meta_dir: Path, split: str, label_column: str):
    """Load a split, inner-join data <-> meta on `input`, return (df, X_meta, y_raw)."""
    df_data = pd.read_csv(dataset_dir / f"{split}.csv")
    df_meta = pd.read_csv(meta_dir / f"{split}_meta.csv")

    n_data, n_meta = len(df_data), len(df_meta)
    # Drop label/class from meta side to avoid suffix collisions; keep them from data.
    meta_drop = [c for c in ("class", "label") if c in df_meta.columns]
    df_meta_features = df_meta.drop(columns=meta_drop)

    merged = df_data.merge(df_meta_features, on="input", how="inner")
    if len(merged) != n_data:
        raise RuntimeError(
            f"[{split}] join lost rows: data={n_data} meta={n_meta} joined={len(merged)}"
        )

    feature_cols = [c for c in df_meta_features.columns if c not in NON_FEATURE_COLS]
    X_meta = merged[feature_cols].to_numpy(dtype=np.float64)
    if np.isnan(X_meta).any():
        n_nan = int(np.isnan(X_meta).sum())
        print(f"[{split}] WARNING: {n_nan} NaNs in metadata — filling with 0")
        X_meta = np.nan_to_num(X_meta, nan=0.0)

    return merged, X_meta, merged[label_column].values, feature_cols


def _compute_auc(y_true, y_scores, n_classes):
    """ROC-AUC. Binary uses 1-D decision scores; multiclass uses macro OvR."""
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
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy":        float(accuracy_score(y, y_pred)),
        "precision_macro": float(p_mac),
        "recall_macro":    float(r_mac),
        "f1_macro":        float(f1_mac),
        "precision_weighted": float(p_wt),
        "recall_weighted":    float(r_wt),
        "f1_weighted":        float(f1_wt),
        "auc":             _compute_auc(y, y_scores, n_classes),
        "y_pred":          y_pred,
        "y_scores":        y_scores,
    }


def evaluate(name, clf, X_dev, y_dev, X_test, y_test, label_names, n_classes):
    out = {}
    for split_name, X, y in (("dev", X_dev, y_dev), ("test", X_test, y_test)):
        m = compute_metrics(clf, X, y, n_classes)
        out[split_name] = m
        out[split_name]["report"] = classification_report(
            y, m["y_pred"], target_names=label_names, digits=4, zero_division=0
        )
    print(f"\n=== {name} ===")
    for split_name in ("dev", "test"):
        r = out[split_name]
        print(
            f"[{split_name}] acc={r['accuracy']:.4f}  prec={r['precision_macro']:.4f}  "
            f"rec={r['recall_macro']:.4f}  f1={r['f1_macro']:.4f}  auc={r['auc']:.4f}"
        )
    print(f"\n[{name}] test classification report:\n{out['test']['report']}")
    return out


def main(args):
    dataset_dir = REPO_ROOT / "data" / f"{args.experiment_type}_datasets" / args.dataset
    meta_dir = Path(args.meta_dir)
    out_dir = REPO_ROOT / "experiments" / "results" / f"svm_{args.dataset}_{args.label_column}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset_dir = {dataset_dir}")
    print(f"meta_dir    = {meta_dir}")
    print(f"label_col   = {args.label_column}")

    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    if args.no_wandb:
        print("wandb disabled by --no_wandb")
    elif not _WANDB_AVAILABLE:
        print("wandb not installed; skipping wandb logging (pip install wandb to enable)")

    # All feature-set runs share a group so they line up as rows of one experiment.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_name = args.wandb_group or f"svm_{args.dataset}_{args.label_column}_{timestamp}"

    train_df, X_meta_tr, y_tr_raw, feat_cols = load_split(dataset_dir, meta_dir, "train", args.label_column)
    dev_df,   X_meta_dv, y_dv_raw, _         = load_split(dataset_dir, meta_dir, "dev",   args.label_column)
    test_df,  X_meta_te, y_te_raw, _         = load_split(dataset_dir, meta_dir, "test",  args.label_column)
    print(f"sizes: train={len(train_df)} dev={len(dev_df)} test={len(test_df)} | meta_features={len(feat_cols)}")

    # Encode labels using train only.
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)
    y_dv = le.transform(y_dv_raw)
    y_te = le.transform(y_te_raw)
    label_names = [str(c) for c in le.classes_]
    n_classes = len(label_names)
    print(f"classes ({n_classes}): {label_names}")

    # --- Feature builders ---
    # Metadata: standardize using train stats.
    scaler = StandardScaler()
    M_tr = scaler.fit_transform(X_meta_tr)
    M_dv = scaler.transform(X_meta_dv)
    M_te = scaler.transform(X_meta_te)

    # String: char n-gram TF-IDF on the domain. char_wb respects word boundaries.
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
    )
    S_tr = vec.fit_transform(train_df["input"].astype(str).values)
    S_dv = vec.transform(dev_df["input"].astype(str).values)
    S_te = vec.transform(test_df["input"].astype(str).values)
    print(f"string tfidf vocab size: {S_tr.shape[1]}")

    # Concatenation (sparse) for meta+string.
    MS_tr = hstack([csr_matrix(M_tr), S_tr]).tocsr()
    MS_dv = hstack([csr_matrix(M_dv), S_dv]).tocsr()
    MS_te = hstack([csr_matrix(M_te), S_te]).tocsr()

    feature_sets = {
        "meta_only":        (M_tr,  M_dv,  M_te),
        "string_only":      (S_tr,  S_dv,  S_te),
        "meta_plus_string": (MS_tr, MS_dv, MS_te),
    }

    # Config shared by every per-feature-set run.
    base_config = {
        "dataset":         args.dataset,
        "experiment_type": args.experiment_type,
        "label_column":    args.label_column,
        "model":           "LinearSVC",
        "C":               args.C,
        "max_iter":        args.max_iter,
        "seed":            args.seed,
        "tfidf_analyzer":  "char_wb",
        "tfidf_ngram":     "3-5",
        "tfidf_min_df":    2,
        "num_classes":     n_classes,
        "classes":         label_names,
        "n_train":         int(len(train_df)),
        "n_dev":           int(len(dev_df)),
        "n_test":          int(len(test_df)),
        "n_meta_features": len(feat_cols),
    }

    all_results = {}
    for name, (Xtr, Xdv, Xte) in feature_sets.items():
        print(f"\n--- training LinearSVC on '{name}' (X_train shape={Xtr.shape}) ---")
        t0 = time.time()
        clf = LinearSVC(C=args.C, max_iter=args.max_iter, dual="auto", random_state=args.seed)
        clf.fit(Xtr, y_tr)
        fit_seconds = time.time() - t0
        print(f"fit took {fit_seconds:.1f}s")

        results = evaluate(name, clf, Xdv, y_dv, Xte, y_te, label_names, n_classes)
        all_results[name] = results

        # One wandb run per feature set, all grouped together → clean row-per-row
        # comparison in the runs table with columns accuracy/precision/recall/f1/auc.
        if use_wandb:
            run_config = dict(base_config)
            run_config["feature_set"] = name
            run_config["n_features"] = int(Xtr.shape[1])
            run_config["fit_seconds"] = fit_seconds

            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=group_name,
                job_type=name,
                name=f"{name}__{timestamp}",
                config=run_config,
                tags=["svm", "baseline", args.dataset, args.label_column, name],
            )
            # Use summary.* for run-table columns (one value per run, no time axis).
            for split_name in ("dev", "test"):
                m = results[split_name]
                for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                          "precision_weighted", "recall_weighted", "f1_weighted", "auc"):
                    run.summary[f"{split_name}/{k}"] = m[k]
            run.summary["fit_seconds"] = fit_seconds
            run.summary["n_features"] = int(Xtr.shape[1])

            # Per-class F1 from the classification report on test.
            per_class = precision_recall_fscore_support(
                y_te, results["test"]["y_pred"], labels=list(range(n_classes)), zero_division=0
            )
            for cls_idx, cls_name in enumerate(label_names):
                run.summary[f"test/per_class/{cls_name}/precision"] = float(per_class[0][cls_idx])
                run.summary[f"test/per_class/{cls_name}/recall"]    = float(per_class[1][cls_idx])
                run.summary[f"test/per_class/{cls_name}/f1"]        = float(per_class[2][cls_idx])
                run.summary[f"test/per_class/{cls_name}/support"]   = int(per_class[3][cls_idx])

            run.finish()

    # --- summary table (across feature sets) ---
    summary_rows = []
    for name, r in all_results.items():
        summary_rows.append({
            "feature_set":         name,
            "dev_accuracy":        r["dev"]["accuracy"],
            "dev_precision_macro": r["dev"]["precision_macro"],
            "dev_recall_macro":    r["dev"]["recall_macro"],
            "dev_f1_macro":        r["dev"]["f1_macro"],
            "dev_auc":             r["dev"]["auc"],
            "test_accuracy":       r["test"]["accuracy"],
            "test_precision_macro":r["test"]["precision_macro"],
            "test_recall_macro":   r["test"]["recall_macro"],
            "test_f1_macro":       r["test"]["f1_macro"],
            "test_f1_weighted":    r["test"]["f1_weighted"],
            "test_auc":            r["test"]["auc"],
        })
    summary = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    summary.to_csv(out_dir / "summary.csv", index=False)
    # Strip non-JSON-serializable arrays before dumping.
    serializable = {}
    for name, splits in all_results.items():
        serializable[name] = {}
        for split, r in splits.items():
            serializable[name][split] = {
                k: v for k, v in r.items() if k not in ("y_pred", "y_scores")
            }
    with open(out_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nwrote: {out_dir / 'summary.csv'}")
    print(f"wrote: {out_dir / 'results.json'}")

    # One extra wandb run that holds the comparison table + the artifact, so you have
    # a single place to look at the full picture for this experiment.
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=group_name,
            job_type="summary",
            name=f"summary__{timestamp}",
            config=base_config,
            tags=["svm", "baseline", args.dataset, args.label_column, "summary"],
        )
        run.log({"comparison": wandb.Table(dataframe=summary)})
        artifact = wandb.Artifact(
            name=f"svm-results-{args.dataset}-{args.label_column}",
            type="results",
            metadata={"dataset": args.dataset, "label_column": args.label_column,
                      "group": group_name},
        )
        artifact.add_file(str(out_dir / "summary.csv"))
        artifact.add_file(str(out_dir / "results.json"))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM baseline: metadata vs string vs both.")
    parser.add_argument("--dataset", type=str, default="ThreatFox_MalDomains")
    parser.add_argument("--meta_dir", type=str, required=True,
                        help="Directory with {train,dev,test}_meta.csv")
    parser.add_argument("--experiment_type", type=str, default="domain", choices=["domain", "url"])
    parser.add_argument("--label_column", type=str, default="label",
                        help="'label' for binary, 'class' for multiclass")
    parser.add_argument("--C", type=float, default=1.0, help="LinearSVC C")
    parser.add_argument("--max_iter", type=int, default=10000,
                        help="LinearSVC max_iter (5k under-converged on meta+string)")
    parser.add_argument("--seed", type=int, default=3407)
    # wandb
    parser.add_argument("--wandb_project", type=str, default="DomURLs_BERT_metadata",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity (team/user); defaults to your wandb login")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="wandb group name; default svm_<dataset>_<label>_<timestamp>")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    args = parser.parse_args()
    main(args)
