"""
pipeline/features.py
─────────────────────
Feature engineering for Windows Event Log data.

Exports:
    build_features(df)  →  (X_scaled, df_enriched)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


CATEGORICAL_COLS = ["MachineName", "Category", "EntryType", "Source"]


def build_features(df: pd.DataFrame):
    """
    Build the full feature matrix from a raw event log DataFrame.

    Returns
    -------
    X_scaled : np.ndarray  shape (n_samples, n_features), float32
    df       : pd.DataFrame with temporal helper columns attached
    """
    df = df.copy()

    # ── Temporal ──────────────────────────────────────────────────────────────
    if "TimeGenerated" not in df.columns:
        df["TimeGenerated"] = pd.Timestamp("2000-01-01")
    df["TimeGenerated"] = pd.to_datetime(df["TimeGenerated"], errors="coerce")
    df["TimeGenerated"].fillna(pd.Timestamp("2000-01-01"), inplace=True)

    ts   = df["TimeGenerated"]
    hour = ts.dt.hour
    dow  = ts.dt.dayofweek

    temporal = pd.DataFrame({
        "hour":        hour,
        "dayofweek":   dow,
        "day":         ts.dt.day,
        "month":       ts.dt.month,
        "quarter":     ts.dt.quarter,
        "after_hours": ((hour < 7) | (hour >= 19)).astype(int),
        "is_weekend":  (dow >= 5).astype(int),
        "hour_sin":    np.sin(2 * np.pi * hour / 24),
        "hour_cos":    np.cos(2 * np.pi * hour / 24),
        "dow_sin":     np.sin(2 * np.pi * dow  / 7),
        "dow_cos":     np.cos(2 * np.pi * dow  / 7),
    })

    # Attach helper cols to df for downstream use
    for col in ["hour", "dayofweek", "after_hours", "is_weekend"]:
        df[col] = temporal[col].values

    # ── Categorical label encoding ─────────────────────────────────────────────
    cat_features = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "unknown"
        le = LabelEncoder()
        cat_features[f"le_{col}"] = le.fit_transform(
            df[col].fillna("unknown").astype(str)
        )
    cat_df = pd.DataFrame(cat_features)

    # ── TF-IDF on Message ─────────────────────────────────────────────────────
    if "Message" not in df.columns:
        df["Message"] = ""
    messages = df["Message"].fillna("").astype(str)

    tfidf = TfidfVectorizer(
        max_features=50, stop_words="english",
        ngram_range=(1, 2), min_df=max(1, len(df) // 200),
    )
    try:
        tfidf_matrix = tfidf.fit_transform(messages).toarray()
    except Exception:
        tfidf_matrix = np.zeros((len(df), 50))
    tfidf_df = pd.DataFrame(
        tfidf_matrix,
        columns=[f"tfidf_{w}" for w in (tfidf.get_feature_names_out()
                 if hasattr(tfidf, "get_feature_names_out") else [])],
    )

    # ── Message statistics ────────────────────────────────────────────────────
    msg_stats = pd.DataFrame({
        "msg_len":         messages.str.len(),
        "msg_word_count":  messages.str.split().str.len().fillna(0),
        "msg_upper_ratio": messages.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)),
        "msg_digit_ratio": messages.apply(lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1)),
        "msg_has_ip":      messages.str.contains(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", regex=True, na=False).astype(int),
        "msg_has_error":   messages.str.contains(r"error|fail|critical|denied|invalid", case=False, na=False).astype(int),
        "msg_has_login":   messages.str.contains(r"login|logon|logoff|authentication", case=False, na=False).astype(int),
    })

    # ── Combine & scale ───────────────────────────────────────────────────────
    X = np.hstack([temporal.values, cat_df.values,
                   tfidf_df.values, msg_stats.values]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    X_scaled = StandardScaler().fit_transform(X).astype(np.float32)
    return X_scaled, df
