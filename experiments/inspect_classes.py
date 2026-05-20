"""Per-class diagnostic: how is multiclass F1 distributed, and how much of the
F1_macro headroom is tunable vs structurally capped by tiny class supports?

Reads:
  1. A saved classification report (e.g. mlruns/ckpts/<run_id>/test_classification_report.txt)
  2. The train CSV (to count train-support per class)

Prints a per-class table (sorted by test F1) and a summary breaking down
F1_macro by train-support tier. Tells you at a glance which classes the
model can't learn because they have too few training samples vs. which
classes are genuinely hard.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match a class row: "<name>  <prec> <rec> <f1> <support>"
# We split each non-empty line and require the last 4 tokens be numeric.
# Stop tokens (accuracy / macro avg / weighted avg) are filtered out.
STOP_NAMES = {"accuracy", "macro", "weighted"}


def parse_report(path):
    rows = []
    text = Path(path).read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 5:
            continue
        try:
            support = int(toks[-1])
            f1      = float(toks[-2])
            recall  = float(toks[-3])
            precis  = float(toks[-4])
        except ValueError:
            continue
        name_toks = toks[:-4]
        # Skip the three summary rows (their first token is in STOP_NAMES).
        if name_toks and name_toks[0].lower() in STOP_NAMES:
            continue
        name = " ".join(name_toks)
        rows.append({
            "class":         name,
            "precision":     precis,
            "recall":        recall,
            "test_f1":       f1,
            "test_support":  support,
        })
    if not rows:
        raise RuntimeError(f"no class rows parsed from {path}")
    return pd.DataFrame(rows)


def support_tier(n):
    if n < 5:    return "tier1_<5"
    if n < 20:   return "tier2_5-19"
    if n < 100:  return "tier3_20-99"
    if n < 1000: return "tier4_100-999"
    return "tier5_>=1000"


def main(args):
    df_report = parse_report(args.report)

    train_path = REPO_ROOT / "data" / "domain_datasets" / args.dataset / "train.csv"
    df_train = pd.read_csv(train_path)
    train_counts = (
        df_train[args.label_column].value_counts()
        .rename_axis("class").reset_index(name="train_support")
    )

    df = df_report.merge(train_counts, on="class", how="left")
    df["train_support"] = df["train_support"].fillna(0).astype(int)
    df["tier"] = df["train_support"].apply(support_tier)

    df = df.sort_values(["test_f1", "train_support"], ascending=[True, True])

    # Output: per-class table
    print(f"\n=== Per-class results (sorted by test_f1 asc) ===\n")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df[["class", "train_support", "test_support",
                  "precision", "recall", "test_f1", "tier"]].to_string(index=False))

    # Summary
    print(f"\n=== Summary ({len(df)} classes) ===")
    print(f"  classes with test_f1 == 0:        {(df['test_f1'] == 0).sum()}")
    print(f"  classes with 0 < test_f1 < 0.3:   {((df['test_f1'] > 0) & (df['test_f1'] < 0.3)).sum()}")
    print(f"  classes with 0.3 <= test_f1 < 0.7:{((df['test_f1'] >= 0.3) & (df['test_f1'] < 0.7)).sum()}")
    print(f"  classes with test_f1 >= 0.7:      {(df['test_f1'] >= 0.7).sum()}")
    print(f"  classes with train_support < 5:   {(df['train_support'] < 5).sum()}")
    print(f"  classes with train_support < 20:  {(df['train_support'] < 20).sum()}")
    print(f"  classes with train_support == 0:  {(df['train_support'] == 0).sum()}")

    # Macro F1 by tier — answers: which support tier is dragging F1_macro down?
    print(f"\n=== F1_macro per train-support tier ===")
    tier_summary = (
        df.groupby("tier", sort=True)
          .agg(num_classes=("class", "size"),
               mean_test_f1=("test_f1", "mean"),
               median_test_f1=("test_f1", "median"),
               max_train_support=("train_support", "max"),
               min_train_support=("train_support", "min"))
          .reset_index()
    )
    print(tier_summary.to_string(index=False))

    # F1_macro contribution: each class contributes test_f1/N to the macro avg.
    # Show how much of the F1_macro shortfall lives in the small-support tiers.
    overall_macro = df["test_f1"].mean()
    print(f"\n  overall_F1_macro = {overall_macro:.4f}  ({len(df)} classes)")
    for tier in sorted(df["tier"].unique()):
        sub = df[df["tier"] == tier]
        contrib = sub["test_f1"].sum() / len(df)
        max_contrib = len(sub) / len(df)
        print(f"  {tier:18s}  n={len(sub):3d}  "
              f"contributes {contrib:.4f} of macro (ceiling {max_contrib:.4f})")

    # Filtered F1_macro: if we drop classes with tiny train support, what does
    # the "learnable" F1_macro look like?
    print(f"\n=== F1_macro filtered to classes with train_support >= N ===")
    for n in (1, 5, 10, 20, 50, 100):
        sub = df[df["train_support"] >= n]
        if len(sub) == 0:
            continue
        macro = sub["test_f1"].mean()
        print(f"  N>={n:4d}: {len(sub):3d} classes, F1_macro={macro:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-class diagnostic for the multiclass run.")
    parser.add_argument("--report", type=str, required=True,
                        help="Path to test_classification_report.txt "
                             "(e.g. mlruns/ckpts/<run_id>/test_classification_report.txt)")
    parser.add_argument("--dataset", type=str, default="ThreatFox_MalDomains")
    parser.add_argument("--label_column", type=str, default="class")
    args = parser.parse_args()
    main(args)
