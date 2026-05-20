"""End-to-end evaluation of the two-stage classifier.

Loads stage-1 (binary legit/malicious) and stage-2 (family classifier trained
on malicious-only) checkpoints, runs both on the test set, and combines:

    stage-1 says legit          -> final label = 'legit'
    stage-1 says malicious      -> final label = stage-2 prediction

Reports overall accuracy, macro-F1, weighted-F1, plus per-class breakdown,
so we can compare directly against the single-stage multiclass headline
(F1_macro = 0.4317).

Each checkpoint dir must contain `best_checkpoint.ckpt` (Lightning's
default) and `model_metadata.p` (saved by main_plm.py).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils.bertdataset import BERTDataset
from models.plm import PLMEncoder
from models.urlbert import URLBertForSequenceClassification
from module.pl_module import BaseModel
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_recall_fscore_support,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_stage(ckpt_dir, batch_size=256, num_workers=4):
    """Returns (lit_model, meta_pkl)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "best_checkpoint.ckpt"
    meta_path = ckpt_dir / "model_metadata.p"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # Re-instantiate the classifier exactly as in main_plm.py.
    model_params = {
        "output_size":     meta["num_classes"],
        "drop_prob":       meta["drop_prob"],
        "pretrained_path": meta["pretrained_path"],
        "meta_dim":        meta["meta_dim"],
        "meta_hidden_dim": meta.get("meta_hidden_dim", 128),
        "meta_out_dim":    meta.get("meta_out_dim", 64),
        "fusion_mode":     meta.get("fusion_mode", "concat"),
        "head_hidden_dim": meta.get("head_hidden_dim", 0),
        "meta_layernorm":  meta.get("meta_layernorm", False),
    }
    if "urlbert" in meta["pretrained_path"].lower():
        classifier = URLBertForSequenceClassification(**model_params)
    else:
        classifier = PLMEncoder(**model_params)

    lit = BaseModel.load_from_checkpoint(
        ckpt_path, classifier=classifier,
        num_classes=meta["num_classes"],
        criterion=torch.nn.CrossEntropyLoss(),
        config=type("Cfg", (), {"lr": meta["lr"], "weight_decay": meta["weight_decay"]})(),
        names=meta["label_encoder"].classes_,
        # strict=False ignores criterion.weight buffer when training used class weights —
        # we only need the classifier params for inference.
        strict=False,
    )
    lit.eval()
    return lit, meta


def predict(lit, meta_pkl, dataset_dir, args):
    """Run the model on test.csv and return (y_pred_int, y_true_int, raw_input_strings)."""
    test_csv = REPO_ROOT / "data" / "domain_datasets" / args.dataset / "test.csv"
    df = pd.read_csv(test_csv)

    le = meta_pkl["label_encoder"]
    pretrained_path = meta_pkl["pretrained_path"]
    max_length = meta_pkl["max_length"]
    label_column = meta_pkl["label_column"]
    meta_dim = meta_pkl["meta_dim"]

    # Build the same meta vectors as in training (using the saved scaler).
    meta_vectors = None
    if meta_dim > 0:
        meta_csv = Path(args.metadata_dir) / "test_meta.csv"
        df_meta = pd.read_csv(meta_csv).drop(columns=[c for c in ("class", "label") if c in pd.read_csv(meta_csv).columns])
        merged = df.merge(df_meta, on="input", how="inner")
        if len(merged) != len(df):
            raise RuntimeError(f"meta join lost rows: {len(df)} -> {len(merged)}")
        df = merged
        feature_cols = meta_pkl["meta_feature_cols"]
        X = df[feature_cols].to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0)
        meta_vectors = meta_pkl["meta_scaler"].transform(X).astype(np.float32)

    # For stage 2 we only ran inference on rows where stage 1 said malicious;
    # the caller will hand us a filtered df. Here we use whatever `df` is.
    y_true_str = df[label_column] if label_column in df.columns else None
    ds = BERTDataset(
        texts=df["input"].values,
        labels=np.zeros(len(df), dtype=np.int64),  # placeholder; we only care about preds
        max_length=max_length,
        pretrained_path=pretrained_path,
        meta_vectors=meta_vectors,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    preds = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit.to(device)
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            x = {k: v.to(device) for k, v in x.items()}
            logits = lit.classifier(**x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred_int = np.concatenate(preds)
    y_pred_str = le.inverse_transform(y_pred_int)

    return df["input"].values, y_pred_str


def main(args):
    print(f"loading stage 1 from {args.stage1_dir}")
    s1_lit, s1_meta = load_stage(args.stage1_dir)
    print(f"loading stage 2 from {args.stage2_dir}")
    s2_lit, s2_meta = load_stage(args.stage2_dir)

    # Stage 1: classify everything as legit/malicious.
    print("\nrunning stage 1 (legit vs malicious) on full test set...")
    inputs_s1, y_pred_s1 = predict(s1_lit, s1_meta, None, args)

    # Stage 2: classify the malicious-predicted rows into families.
    test_csv = REPO_ROOT / "data" / "domain_datasets" / args.dataset / "test.csv"
    df_test = pd.read_csv(test_csv)
    y_true = df_test["class"].values  # the multiclass ground truth (incl. 'legit')

    df_mal = df_test[pd.Series(y_pred_s1) == "malicious"].reset_index(drop=True)
    print(f"stage 1 predicted {len(df_mal)} / {len(df_test)} as malicious; "
          f"running stage 2 on those rows...")

    # Tweak: predict() reads the test.csv from disk and uses its rows. We need to
    # restrict to df_mal — easiest is to write a temp CSV or rebuild the dataset
    # directly here.
    le2 = s2_meta["label_encoder"]
    pretrained_path = s2_meta["pretrained_path"]
    max_length = s2_meta["max_length"]
    meta_dim = s2_meta["meta_dim"]

    meta_vectors2 = None
    if meta_dim > 0:
        meta_csv = Path(args.metadata_dir) / "test_meta.csv"
        df_meta = pd.read_csv(meta_csv)
        df_meta = df_meta.drop(columns=[c for c in ("class", "label") if c in df_meta.columns])
        merged = df_mal.merge(df_meta, on="input", how="inner")
        if len(merged) != len(df_mal):
            raise RuntimeError(f"stage2 meta join lost rows: {len(df_mal)} -> {len(merged)}")
        df_mal = merged
        feature_cols = s2_meta["meta_feature_cols"]
        X = df_mal[feature_cols].to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0)
        meta_vectors2 = s2_meta["meta_scaler"].transform(X).astype(np.float32)

    ds2 = BERTDataset(
        texts=df_mal["input"].values,
        labels=np.zeros(len(df_mal), dtype=np.int64),
        max_length=max_length,
        pretrained_path=pretrained_path,
        meta_vectors=meta_vectors2,
    )
    loader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s2_lit.to(device)
    preds2 = []
    with torch.no_grad():
        for batch in loader2:
            x, _ = batch
            x = {k: v.to(device) for k, v in x.items()}
            logits = s2_lit.classifier(**x)
            preds2.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred_s2_int = np.concatenate(preds2)
    y_pred_s2_str = le2.inverse_transform(y_pred_s2_int)

    # Combine: where stage1 said legit -> 'legit'; else stage2 prediction.
    s2_lookup = dict(zip(df_mal["input"].values, y_pred_s2_str))
    final = []
    for inp, s1 in zip(df_test["input"].values, y_pred_s1):
        if s1 == "legit":
            final.append("legit")
        else:
            final.append(s2_lookup.get(inp, "legit"))  # shouldn't happen
    final = np.array(final)

    # Metrics
    print("\n=== End-to-end two-stage results ===")
    print(f"accuracy:    {accuracy_score(y_true, final):.4f}")
    print(f"F1_macro:    {f1_score(y_true, final, average='macro', zero_division=0):.4f}")
    print(f"F1_weighted: {f1_score(y_true, final, average='weighted', zero_division=0):.4f}")

    print("\n=== Per-class report ===")
    print(classification_report(y_true, final, digits=4, zero_division=0))

    # Save predictions for later analysis.
    out = REPO_ROOT / "experiments" / "results" / "two_stage_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "input": df_test["input"].values,
        "y_true": y_true,
        "stage1_pred": y_pred_s1,
        "final_pred": final,
    }).to_csv(out, index=False)
    print(f"\nwrote predictions to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end two-stage evaluation.")
    parser.add_argument("--stage1_dir", type=str, required=True,
                        help="Checkpoint dir for stage-1 (binary). Contains best_checkpoint.ckpt + model_metadata.p")
    parser.add_argument("--stage2_dir", type=str, required=True,
                        help="Checkpoint dir for stage-2 (family on malicious-only).")
    parser.add_argument("--dataset", type=str, default="ThreatFox_MalDomains")
    parser.add_argument("--metadata_dir", type=str,
                        default="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/domains/DomainsMetadata")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
