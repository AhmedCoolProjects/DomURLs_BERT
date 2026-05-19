import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Non-feature columns in *_meta.csv that must never end up in the metadata vector.
_META_NON_FEATURE_COLS = {"input", "class", "label", "_error", "_elapsed_s"}


def _select_meta_columns(all_feature_cols, categories):
    """Return the metadata column subset matching the given category prefixes.

    Always keeps the always-on flags (input_valid, input_is_ip_address)."""
    always_on = [c for c in ("input_valid", "input_is_ip_address") if c in all_feature_cols]
    if categories is None:
        return always_on + [c for c in all_feature_cols if c not in always_on]
    cat_set = set(categories)
    picked = list(always_on)
    for c in all_feature_cols:
        if c in always_on:
            continue
        prefix = c.split("_", 1)[0]
        if prefix in cat_set:
            picked.append(c)
    return picked


def _load_meta(meta_dir, split, categories):
    """Read one meta split CSV. Returns (df_meta_id, X_meta, feature_cols)."""
    df = pd.read_csv(os.path.join(meta_dir, f"{split}_meta.csv"))
    # The meta file has class/label too; drop them so we only join on `input`.
    df = df.drop(columns=[c for c in ("class", "label") if c in df.columns])
    all_feature_cols = [c for c in df.columns if c not in _META_NON_FEATURE_COLS]
    feature_cols = _select_meta_columns(all_feature_cols, categories)
    return df[["input"] + feature_cols], feature_cols


def load_dataset(path, label_column, meta_dir=None, meta_categories=None):
    """Load train/dev/test plus (optionally) the metadata vectors aligned by `input`.

    When `meta_dir` is given, returns extra keys:
        'meta_train', 'meta_dev', 'meta_test'  -> np.ndarray (n, d)
        'meta_scaler'                          -> fitted StandardScaler
        'meta_feature_cols'                    -> list[str], in column order
    """
    le = LabelEncoder()

    df_train = pd.read_csv(os.path.join(path, "train.csv"))
    df_dev   = pd.read_csv(os.path.join(path, "dev.csv"))
    df_test  = pd.read_csv(os.path.join(path, "test.csv"))

    df_train[label_column] = le.fit_transform(df_train[label_column])
    df_dev[label_column]   = le.transform(df_dev[label_column])
    df_test[label_column]  = le.transform(df_test[label_column])

    out = {"train": df_train, "dev": df_dev, "test": df_test, "label_encoder": le}

    if meta_dir is None:
        return out

    meta_train, feature_cols = _load_meta(meta_dir, "train", meta_categories)
    meta_dev,   _            = _load_meta(meta_dir, "dev",   meta_categories)
    meta_test,  _            = _load_meta(meta_dir, "test",  meta_categories)

    def _align(df_data, df_meta, split_name):
        before = len(df_data)
        merged = df_data.merge(df_meta, on="input", how="inner")
        if len(merged) != before:
            raise RuntimeError(
                f"[{split_name}] join lost rows: data={before} joined={len(merged)}"
            )
        return merged

    df_train_m = _align(df_train, meta_train, "train")
    df_dev_m   = _align(df_dev,   meta_dev,   "dev")
    df_test_m  = _align(df_test,  meta_test,  "test")

    X_tr = df_train_m[feature_cols].to_numpy(dtype=np.float32)
    X_dv = df_dev_m[feature_cols].to_numpy(dtype=np.float32)
    X_te = df_test_m[feature_cols].to_numpy(dtype=np.float32)
    for name, X in (("train", X_tr), ("dev", X_dv), ("test", X_te)):
        if np.isnan(X).any():
            print(f"[meta:{name}] WARNING: NaNs in metadata; filling with 0")
            np.nan_to_num(X, copy=False, nan=0.0)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_dv_s = scaler.transform(X_dv).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    # Replace dfs with merged versions so input order matches the meta order.
    out["train"], out["dev"], out["test"] = df_train_m, df_dev_m, df_test_m
    out["meta_train"]        = X_tr_s
    out["meta_dev"]          = X_dv_s
    out["meta_test"]         = X_te_s
    out["meta_scaler"]       = scaler
    out["meta_feature_cols"] = feature_cols
    return out
