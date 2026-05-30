"""Step 1 of feature-selection pipeline: find dead / near-constant metadata features.

What "dead" means here:
  - exactly constant         (nunique == 1)
  - effectively constant     (mode frequency > --mode_frac_threshold, e.g. 0.99)
  - tiny variance            (variance < --variance_threshold)
  - high NaN rate            (na rate > --nan_threshold)

Run this on your TRAIN split only — selection must not peek at dev/test.

Inputs:  one *_meta.csv (or any CSV that has the metadata columns).
Outputs: a printed report + two text files written next to the CSV:
  - <stem>_dead_columns.txt   one column name per line that you should drop
  - <stem>_keep_columns.txt   one column name per line that survives
  - <stem>_feature_stats.csv  full per-feature stats table (for inspection)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Columns in the meta CSVs that are never features (labels / bookkeeping).
NON_FEATURE_COLS = {"input", "class", "label", "_error", "_elapsed_s"}


def analyze(df, feature_cols):
    """Return a DataFrame with one row per feature and stats columns."""
    rows = []
    n = len(df)
    for col in feature_cols:
        s = df[col]
        # nunique excludes NaN by default; we also report the raw na count.
        nunique = int(s.nunique(dropna=True))
        na_count = int(s.isna().sum())
        na_rate = na_count / max(n, 1)

        # Mode and mode frequency (skip NaN). If all-NaN, mark as constant 0 stats.
        s_clean = s.dropna()
        if len(s_clean) == 0:
            mode_val, mode_frac = float("nan"), 1.0
            var = 0.0
            mn = mx = mean = std = float("nan")
        else:
            mode_val = s_clean.mode().iloc[0]
            mode_frac = float((s_clean == mode_val).mean())
            # Cast to float for variance/etc.; non-numeric will raise -> skip.
            try:
                v = pd.to_numeric(s_clean, errors="raise").astype("float64")
            except Exception:
                v = None
            if v is None:
                var = float("nan")
                mn = mx = mean = std = float("nan")
            else:
                var = float(v.var(ddof=0))   # population variance, matches VarianceThreshold
                mn = float(v.min())
                mx = float(v.max())
                mean = float(v.mean())
                std = float(v.std(ddof=0))

        rows.append({
            "feature": col,
            "nunique": nunique,
            "na_rate": na_rate,
            "mode_value": mode_val,
            "mode_frac": mode_frac,
            "variance": var,
            "min": mn,
            "max": mx,
            "mean": mean,
            "std": std,
        })
    return pd.DataFrame(rows)


def flag_dead(stats, variance_threshold, mode_frac_threshold, nan_threshold):
    """Add a 'reason' column. Empty string ⇒ keep."""
    def _reason(row):
        reasons = []
        if row["nunique"] <= 1:
            reasons.append("constant")
        if pd.notna(row["variance"]) and row["variance"] < variance_threshold:
            reasons.append(f"var<{variance_threshold}")
        if row["mode_frac"] > mode_frac_threshold:
            reasons.append(f"mode_frac>{mode_frac_threshold}")
        if row["na_rate"] > nan_threshold:
            reasons.append(f"na_rate>{nan_threshold}")
        return ",".join(reasons)
    stats = stats.copy()
    stats["drop_reason"] = stats.apply(_reason, axis=1)
    stats["keep"] = stats["drop_reason"] == ""
    return stats


def main(args):
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} rows, {df.shape[1]} columns from {csv_path}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"non-feature columns dropped: {sorted(set(df.columns) & NON_FEATURE_COLS)}")
    print(f"analyzing {len(feature_cols)} candidate feature columns")

    stats = analyze(df, feature_cols)
    stats = flag_dead(stats, args.variance_threshold, args.mode_frac_threshold, args.nan_threshold)

    # Sort for readability: dead ones first (by mode_frac desc, then variance asc).
    sort_keys = stats.sort_values(
        by=["keep", "mode_frac", "variance"],
        ascending=[True, False, True],
    )

    dead = sort_keys[~sort_keys["keep"]]
    keep = sort_keys[ sort_keys["keep"]]
    print(f"\n=== {len(dead)} features flagged as dead ===")
    if len(dead) > 0:
        print(dead[["feature", "nunique", "mode_value", "mode_frac",
                    "variance", "na_rate", "drop_reason"]]
              .to_string(index=False))

    print(f"\n=== {len(keep)} features kept ===  (showing 10 with smallest variance for sanity)")
    print(keep.nsmallest(10, "variance")[
        ["feature", "nunique", "mode_value", "mode_frac", "variance", "na_rate"]
    ].to_string(index=False))

    # Group dropped features by category prefix to see where the dead weight lives.
    if len(dead) > 0:
        dead = dead.copy()
        dead["prefix"] = dead["feature"].str.split("_", n=1).str[0]
        by_prefix = dead.groupby("prefix").size().sort_values(ascending=False)
        print(f"\n=== dead features by category prefix ===")
        print(by_prefix.to_string())

    # Write outputs next to the CSV (or to --output_dir if given).
    out_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    dead_path  = out_dir / f"{stem}_dead_columns.txt"
    keep_path  = out_dir / f"{stem}_keep_columns.txt"
    stats_path = out_dir / f"{stem}_feature_stats.csv"

    dead_path.write_text("\n".join(dead["feature"].tolist()))
    keep_path.write_text("\n".join(keep["feature"].tolist()))
    sort_keys.to_csv(stats_path, index=False)

    print(f"\nwrote: {dead_path}")
    print(f"wrote: {keep_path}")
    print(f"wrote: {stats_path}")
    print(f"\nsummary: {len(feature_cols)} candidate -> {len(keep)} kept, {len(dead)} dropped "
          f"({100*len(dead)/max(len(feature_cols),1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: drop dead / near-constant metadata features.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to a *_meta.csv (use the TRAIN split — do not peek at dev/test).")
    parser.add_argument("--variance_threshold", type=float, default=1e-4,
                        help="Drop features with variance < this. Default 1e-4.")
    parser.add_argument("--mode_frac_threshold", type=float, default=0.99,
                        help="Drop features whose mode value covers > this fraction of rows. Default 0.99.")
    parser.add_argument("--nan_threshold", type=float, default=0.5,
                        help="Drop features whose NaN rate exceeds this. Default 0.5.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write outputs (default: next to --csv).")
    args = parser.parse_args()
    main(args)
