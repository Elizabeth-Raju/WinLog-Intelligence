"""
pipeline/detector.py
─────────────────────
Anomaly detection pipeline — trains the Autoencoder, scores every event,
then calls build_summary() + write_report() from report.py.

Exports:
    run_pipeline(job_id, df_raw, jobs_store, report_dir)
    severity_label(score)
    DEVICE
"""

import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pipeline.features import build_features
from pipeline.model    import Autoencoder
from pipeline.report   import build_summary, write_report


ANOMALY_PERCENTILE = 95.0
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS             = 30


def severity_label(score: float) -> str:
    if score >= 0.9: return "CRITICAL"
    if score >= 0.7: return "HIGH"
    if score >= 0.5: return "MEDIUM"
    return "LOW"


def run_pipeline(job_id: str, df_raw: pd.DataFrame,
                 jobs_store: dict, report_dir: Path) -> None:
    """
    Full pipeline:
        1. Feature engineering
        2. Autoencoder training
        3. Reconstruction error -> threshold -> severity labels
        4. build_summary()  (notebook-compatible field names)
        5. write_report()   (LLM narrative + full report)

    Updates jobs_store[job_id] in-place at every stage.
    """
    try:
        # ── 1. Features ───────────────────────────────────────────────────────
        jobs_store[job_id]["status"] = "feature_engineering"
        X_scaled, df = build_features(df_raw)

        # ── 2. Train ──────────────────────────────────────────────────────────
        jobs_store[job_id]["status"] = "training_autoencoder"

        model     = Autoencoder(X_scaled.shape[1]).to(DEVICE)
        dataset   = torch.utils.data.TensorDataset(torch.FloatTensor(X_scaled))
        loader    = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        history = []
        for _ in range(EPOCHS):
            model.train()
            ep_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                loss  = criterion(model(batch), batch)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            history.append(ep_loss / len(loader))

        # ── 3. Detect ─────────────────────────────────────────────────────────
        jobs_store[job_id]["status"] = "detecting_anomalies"

        model.eval()
        with torch.no_grad():
            recon = model(torch.FloatTensor(X_scaled).to(DEVICE)).cpu().numpy()

        errors        = np.mean((X_scaled - recon) ** 2, axis=1)
        threshold     = float(np.percentile(errors, ANOMALY_PERCENTILE))
        is_anomaly    = errors > threshold
        anomaly_score = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        df["reconstruction_error"] = errors
        df["anomaly_score"]        = anomaly_score
        df["is_anomaly"]           = is_anomaly
        df["severity"]             = [severity_label(s) if a else "NORMAL"
                                      for s, a in zip(anomaly_score, is_anomaly)]
        anomaly_df = df[is_anomaly].copy()

        # ── 4. Build summary (notebook-compatible field names) ─────────────────
        jobs_store[job_id]["status"] = "generating_report"

        summary = build_summary(df, anomaly_df, errors, threshold)

        # Attach extra fields needed by the frontend (loss curve, hourly, etc.)
        summary["loss_curve"]    = [round(v, 6) for v in history[-10:]]
        summary["total_events"]  = len(df)   # alias for frontend compat

        hourly = {}
        if "hour" in df.columns:
            for h, grp in df.groupby("hour"):
                hourly[int(h)] = int(grp["is_anomaly"].sum())
        summary["hourly_anomalies"] = hourly

        top_cols      = [c for c in ["MachineName", "Source", "EntryType", "Category",
                                      "severity", "reconstruction_error", "anomaly_score"]
                         if c in anomaly_df.columns]
        summary["top_anomalies"] = (anomaly_df[top_cols]
                                    .sort_values("reconstruction_error", ascending=False)
                                    .head(10).to_dict(orient="records"))

        # ── 5. Write report ───────────────────────────────────────────────────
        report_path = report_dir / f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        write_report(summary, report_path, anomaly_df)

        jobs_store[job_id].update({
            "status":      "done",
            "result":      summary,
            "report_path": str(report_path),
        })

    except Exception:
        jobs_store[job_id].update({
            "status": "error",
            "error":  traceback.format_exc(),
        })
