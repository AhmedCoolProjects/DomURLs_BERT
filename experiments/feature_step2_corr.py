"""Step 2 of feature-selection pipeline: Spearman correlation + hierarchical
clustering for redundancy pruning.

After step 1 (dead features) and the step-3 MI audit (restored rare-but-useful),
we still have features that say *the same thing* as each other (e.g.
`tls_certificate_expired` vs `tls_days_to_expiration < 0`). This script
finds those clusters and keeps only one feature per cluster — the one with
the highest mutual information vs the label.

Procedure
---------
1. Load the train CSV and the current keep list (typically the v2 from step 3).
2. Compute Spearman rank correlation on those features (robust to scale + skew).
3. Convert to distance: d = 1 - |corr|.
4. Hierarchical cluster (average linkage by default).
5. Cut at distance < (1 - --corr_threshold) to form groups of mutually-correlated
   features.
6. For each cluster of size > 1, keep the feature with the highest MI vs the
   chosen --label_column; drop the rest.

Run on TRAIN ONLY. Outputs (next to the CSV unless --output_dir given):
  - <stem>_step2_clusters.csv      every feature with its cluster_id, MI, kept-bool
  - <stem>_keep_columns_v3.txt     pruned keep list (use this for re-training)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def read_list(p):
    return [ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip()]


def spearman_corr_matrix(X):
    """Spearman = Pearson on rank-transformed columns. Faster than pandas .corr."""
    n, d = X.shape
    # average ranks so ties get the average; matches scipy.stats.spearmanr
    R = np.apply_along_axis(rankdata, axis=0, arr=X)
    R = R - R.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(R, axis=0, keepdims=True)
    # Avoid div-by-zero for constant columns
    norms[norms == 0] = 1.0
    R = R / norms
    return R.T @ R  # (d, d)


def main(args):
    csv_path = Path(args.csv).resolve()
    df = pd.read_csv(csv_path)
    keep_cols = read_list(args.keep_file)
    print(f"loaded {len(df)} rows, clustering {len(keep_cols)} features "
          f"(target label='{args.label_column}', corr_threshold={args.corr_threshold}, "
          f"linkage={args.linkage})")

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"keep features not in CSV: {missing[:5]}…")

    # Feature matrix + labels
    X = df[keep_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    y = LabelEncoder().fit_transform(df[args.label_column].values)

    # 1. Mutual information vs label (used to pick the cluster representative)
    print("computing mutual information vs label …")
    mi = mutual_info_classif(X, y, discrete_features="auto", random_state=args.seed)

    # 2. Spearman correlation matrix
    print("computing Spearman correlation matrix …")
    C = spearman_corr_matrix(X)
    C = np.nan_to_num(C, nan=0.0)

    # 3. Distance matrix; force symmetry + zero diagonal for squareform
    D = 1.0 - np.abs(C)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    condensed = squareform(D, checks=False)

    # 4 + 5. Hierarchical clustering, cut at d = 1 - corr_threshold
    print(f"linkage + fcluster …")
    Z = linkage(condensed, method=args.linkage)
    distance_threshold = 1.0 - args.corr_threshold
    cluster_ids = fcluster(Z, t=distance_threshold, criterion="distance")

    table = pd.DataFrame({
        "feature": keep_cols,
        "cluster": cluster_ids,
        "mi": mi,
    }).sort_values(["cluster", "mi"], ascending=[True, False])

    # 6. Within each cluster, keep the feature with the highest MI
    kept = (
        table.groupby("cluster", sort=False)
             .head(1)["feature"]
             .tolist()
    )
    kept_set = set(kept)
    table["kept"] = table["feature"].isin(kept_set)

    # --- reporting ---
    sizes = table.groupby("cluster").size()
    redundant = sizes[sizes > 1].index.tolist()
    print(f"\n=== {len(redundant)} redundant clusters "
          f"(|spearman| >= {args.corr_threshold}, {args.linkage} linkage) ===")
    for cid in redundant:
        members = table[table["cluster"] == cid]
        print(f"\n  cluster {cid}  ({len(members)} features):")
        for _, row in members.iterrows():
            tag = "KEEP" if row["kept"] else "drop"
            print(f"    [{tag}]  {row['feature']:48s}  mi={row['mi']:.4f}")

    # Singletons (no near-duplicate) just for the row count
    n_singleton = int((sizes == 1).sum())
    print(f"\n  + {n_singleton} singleton features kept (no near-duplicates).")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    keep_v3 = out_dir / f"{stem}_keep_columns_v3.txt"
    clusters_csv = out_dir / f"{stem}_step2_clusters.csv"
    keep_v3.write_text("\n".join(kept))
    table.to_csv(clusters_csv, index=False)

    print(f"\nwrote: {keep_v3}")
    print(f"wrote: {clusters_csv}")
    print(f"\nsummary: {len(keep_cols)} in -> {len(kept)} kept after step 2 "
          f"({100*(len(keep_cols)-len(kept))/max(len(keep_cols),1):.1f}% dropped)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Step 2: Spearman + hierarchical clustering for redundancy pruning.")
    p.add_argument("--csv", required=True, type=str,
                   help="Train *_meta.csv (must include the --label_column).")
    p.add_argument("--keep_file", required=True, type=str,
                   help="Path to the current keep list (e.g. keep_columns_v2.txt).")
    p.add_argument("--label_column", default="class", choices=["label", "class"],
                   help="Used to pick the cluster representative (highest MI).")
    p.add_argument("--corr_threshold", type=float, default=0.9,
                   help="Features in a cluster all have |spearman| >= this. Default 0.9.")
    p.add_argument("--linkage", default="average",
                   choices=["average", "complete", "single", "ward"],
                   help="Hierarchical linkage method. 'complete' is strictest.")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--output_dir", default=None, type=str)
    args = p.parse_args()
    main(args)
