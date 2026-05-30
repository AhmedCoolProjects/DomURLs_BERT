"""Step 3 (audit pass): mutual information of step-1's dropped features vs the label.

Step 1 dropped 72 features for low variance / near-constancy. A rare boolean
that fires in <0.1% of rows still has low variance, but it can be perfectly
predictive (e.g. only 1 for one specific class). Variance can't see that —
mutual information can.

This script computes MI between each STEP-1-DROPPED feature and the label,
ranks them, and writes a new keep list = (original keep) ∪ (top restored by MI).

Run on TRAIN ONLY. Outputs:
  - <stem>_step3_mi_audit.csv      ranked MI table for the dropped features
  - <stem>_keep_columns_v2.txt     updated keep list with restored features
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def read_list(p):
    return [ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip()]


def main(args):
    csv_path = Path(args.csv).resolve()
    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} rows from {csv_path}")

    dropped = read_list(args.dropped_file)
    kept    = read_list(args.keep_file)
    print(f"step 1: {len(kept)} kept, {len(dropped)} dropped")

    # Sanity: features must actually be in the dataframe.
    missing = [c for c in dropped if c not in df.columns]
    if missing:
        raise RuntimeError(f"dropped features not in CSV: {missing[:5]}…")

    if args.label_column not in df.columns:
        raise RuntimeError(f"label column '{args.label_column}' not in CSV columns "
                           f"(have: {sorted(df.columns)[:10]}…)")

    # Build feature matrix from the DROPPED features and the labels.
    X = df[dropped].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    y_raw = df[args.label_column].values
    y = LabelEncoder().fit_transform(y_raw)
    print(f"audit target = '{args.label_column}'   "
          f"({len(np.unique(y))} classes, X shape={X.shape})")

    print("computing mutual information ...")
    mi = mutual_info_classif(X, y, discrete_features="auto", random_state=args.seed)

    audit = pd.DataFrame({"feature": dropped, "mi": mi}).sort_values("mi", ascending=False)

    # Pick which to restore. Either by MI threshold or by top-K (default top-K).
    if args.min_mi is not None:
        restore_mask = audit["mi"] >= args.min_mi
        rule = f"mi >= {args.min_mi}"
    else:
        restore_mask = audit.index.isin(audit.head(args.top_k).index)
        rule = f"top-{args.top_k} by MI"
    restored = audit[restore_mask]["feature"].tolist()
    still_dead = audit[~restore_mask]["feature"].tolist()

    print(f"\n=== full MI ranking of step-1 dropped features ===")
    print(audit.to_string(index=False))

    print(f"\n=== restoring {len(restored)} features ({rule}) ===")
    print(audit[restore_mask].to_string(index=False))

    # Combine with the step-1 keep list. Preserve order: original kept first,
    # restored after, in MI-desc order.
    final_keep = list(kept) + [c for c in restored if c not in kept]
    print(f"\nfinal keep list size: {len(final_keep)}  "
          f"(was {len(kept)} after step 1; +{len(final_keep) - len(kept)} restored)")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    audit_path     = out_dir / f"{stem}_step3_mi_audit.csv"
    keep_v2_path   = out_dir / f"{stem}_keep_columns_v2.txt"
    audit.to_csv(audit_path, index=False)
    keep_v2_path.write_text("\n".join(final_keep))

    print(f"\nwrote: {audit_path}")
    print(f"wrote: {keep_v2_path}")
    if still_dead:
        # Save the survivors of step-3 dead list too, for transparency.
        dead_v2_path = out_dir / f"{stem}_dead_columns_v2.txt"
        dead_v2_path.write_text("\n".join(still_dead))
        print(f"wrote: {dead_v2_path}  ({len(still_dead)} features still dropped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3 audit: mutual information of step-1-dropped features vs label.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to train_meta.csv (has label/class columns).")
    parser.add_argument("--dropped_file", type=str, required=True,
                        help="Path to *_dead_columns.txt produced by step 1.")
    parser.add_argument("--keep_file", type=str, required=True,
                        help="Path to *_keep_columns.txt produced by step 1.")
    parser.add_argument("--label_column", type=str, default="label",
                        choices=["label", "class"],
                        help="Which label to audit against. Run with --label_column class "
                             "as a second pass if you also want family-level signal.")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top-MI features to restore.")
    parser.add_argument("--min_mi", type=float, default=None,
                        help="Alternative to --top_k: restore everything with MI >= this.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write outputs (default: next to --csv).")
    args = parser.parse_args()
    main(args)
