"""SVM probe: does the scraped metadata add signal over the domain string alone?

Trains LinearSVC on three feature sets — metadata-only, string-only (char n-gram
TF-IDF), and the concatenation — and reports dev/test metrics for each. See
CONTEXT.md for the full plan.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
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


def evaluate(name: str, clf, X_dev, y_dev, X_test, y_test, label_names, wandb_run=None):
    out = {}
    for split_name, X, y in (("dev", X_dev, y_dev), ("test", X_test, y_test)):
        y_pred = clf.predict(X)
        out[split_name] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "macro_f1": float(f1_score(y, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y, y_pred, average="weighted")),
            "report": classification_report(
                y, y_pred, target_names=label_names, digits=4, zero_division=0
            ),
        }
    print(f"\n=== {name} ===")
    for split_name in ("dev", "test"):
        r = out[split_name]
        print(
            f"[{split_name}] acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}  "
            f"weighted_f1={r['weighted_f1']:.4f}"
        )
    print(f"\n[{name}] test classification report:\n{out['test']['report']}")

    if wandb_run is not None:
        wandb_run.log({
            f"{name}/dev/accuracy":     out["dev"]["accuracy"],
            f"{name}/dev/macro_f1":     out["dev"]["macro_f1"],
            f"{name}/dev/weighted_f1":  out["dev"]["weighted_f1"],
            f"{name}/test/accuracy":    out["test"]["accuracy"],
            f"{name}/test/macro_f1":    out["test"]["macro_f1"],
            f"{name}/test/weighted_f1": out["test"]["weighted_f1"],
        })
    return out


def main(args):
    dataset_dir = REPO_ROOT / "data" / f"{args.experiment_type}_datasets" / args.dataset
    meta_dir = Path(args.meta_dir)
    out_dir = REPO_ROOT / "experiments" / "results" / f"svm_{args.dataset}_{args.label_column}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset_dir = {dataset_dir}")
    print(f"meta_dir    = {meta_dir}")
    print(f"label_col   = {args.label_column}")

    # --- wandb setup (optional) ---
    wandb_run = None
    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    if args.no_wandb:
        print("wandb disabled by --no_wandb")
    elif not _WANDB_AVAILABLE:
        print("wandb not installed; skipping wandb logging (pip install wandb to enable)")
    if use_wandb:
        run_name = args.wandb_run_name or f"svm_{args.dataset}_{args.label_column}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
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
            },
            tags=["svm", "baseline", args.dataset, args.label_column],
            reinit=True,
        )

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
    print(f"classes ({len(label_names)}): {label_names}")

    if wandb_run is not None:
        wandb_run.config.update({
            "num_classes":     len(label_names),
            "classes":         label_names,
            "n_train":         int(len(train_df)),
            "n_dev":           int(len(dev_df)),
            "n_test":          int(len(test_df)),
            "n_meta_features": len(feat_cols),
        })

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

    all_results = {}
    for name, (Xtr, Xdv, Xte) in feature_sets.items():
        print(f"\n--- training LinearSVC on '{name}' (X_train shape={Xtr.shape}) ---")
        t0 = time.time()
        clf = LinearSVC(C=args.C, max_iter=args.max_iter, dual="auto", random_state=args.seed)
        clf.fit(Xtr, y_tr)
        fit_seconds = time.time() - t0
        print(f"fit took {fit_seconds:.1f}s")
        if wandb_run is not None:
            wandb_run.log({
                f"{name}/fit_seconds":     fit_seconds,
                f"{name}/n_features":      Xtr.shape[1],
            })
        all_results[name] = evaluate(name, clf, Xdv, y_dv, Xte, y_te, label_names, wandb_run=wandb_run)

    # Summary table.
    summary_rows = []
    for name, r in all_results.items():
        summary_rows.append({
            "feature_set":      name,
            "dev_accuracy":     r["dev"]["accuracy"],
            "dev_macro_f1":     r["dev"]["macro_f1"],
            "test_accuracy":    r["test"]["accuracy"],
            "test_macro_f1":    r["test"]["macro_f1"],
            "test_weighted_f1": r["test"]["weighted_f1"],
        })
    summary = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    summary.to_csv(out_dir / "summary.csv", index=False)
    serializable = {
        name: {split: {k: v for k, v in r.items() if k != "report"} | {"report": r["report"]}
               for split, r in splits.items()}
        for name, splits in all_results.items()
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nwrote: {out_dir / 'summary.csv'}")
    print(f"wrote: {out_dir / 'results.json'}")

    if wandb_run is not None:
        wandb_run.log({"summary": wandb.Table(dataframe=summary)})
        artifact = wandb.Artifact(
            name=f"svm-results-{args.dataset}-{args.label_column}",
            type="results",
            metadata={"dataset": args.dataset, "label_column": args.label_column},
        )
        artifact.add_file(str(out_dir / "summary.csv"))
        artifact.add_file(str(out_dir / "results.json"))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM baseline: metadata vs string vs both.")
    parser.add_argument("--dataset", type=str, default="ThreatFox_MalDomains")
    parser.add_argument("--meta_dir", type=str, required=True,
                        help="Directory with {train,dev,test}_meta.csv")
    parser.add_argument("--experiment_type", type=str, default="domain", choices=["domain", "url"])
    parser.add_argument("--label_column", type=str, default="label",
                        help="'label' for binary, 'class' for multiclass")
    parser.add_argument("--C", type=float, default=1.0, help="LinearSVC C")
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    # wandb
    parser.add_argument("--wandb_project", type=str, default="DomURLs_BERT_metadata",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity (team/user); defaults to your wandb login")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name; default svm_<dataset>_<label_column>")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    args = parser.parse_args()
    main(args)
