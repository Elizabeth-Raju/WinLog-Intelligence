"""
pipeline/report.py
───────────────────
Incident report generation — mirrors notebook 04_incident_generation.ipynb.

Model  : google/flan-t5-base (cached after first load)
Prompts: prompt_executive_summary, prompt_impact, prompt_root_cause, prompt_remediation
Report : build_summary() -> write_report()

Exports:
    build_summary(df_all, df_anomaly, errors, threshold)  -> dict
    write_report(summary, path, anomaly_df)
"""

import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL_NAME      = "google/flan-t5-base"
MAX_NEW_TOKENS     = 512
ANOMALY_PERCENTILE = 95.0

# ── LLM — loaded once, cached forever ────────────────────────────────────────
_llm_pipeline = None
_llm_lock     = threading.Lock()


def _get_llm():
    """Load Flan-T5 once and cache — instant on all subsequent calls."""
    global _llm_pipeline
    with _llm_lock:
        if _llm_pipeline is None:
            from transformers import (pipeline as hf_pipeline,
                                      AutoTokenizer, AutoModelForSeq2SeqLM)
            print(f"[LLM] Loading {HF_MODEL_NAME} (first time — will be cached)...")
            tok           = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            mdl           = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
            _llm_pipeline = hf_pipeline(
                "text2text-generation", model=mdl,
                tokenizer=tok, max_new_tokens=MAX_NEW_TOKENS,
            )
            print("[LLM] Model loaded and cached.")
    return _llm_pipeline


def llm_generate(prompt: str) -> str:
    """Call the LLM and return generated text. Safe fallback on error."""
    try:
        result = _get_llm()(prompt, max_new_tokens=MAX_NEW_TOKENS)
        return result[0]["generated_text"].strip()
    except Exception as e:
        return f"[LLM generation failed: {e}]"


# ── Summary builder — mirrors notebook Cell 5 ─────────────────────────────────

def build_summary(df_all: pd.DataFrame, df_anomaly: pd.DataFrame,
                  errors: np.ndarray, threshold: float) -> dict:
    """
    Build the summary dict exactly as in notebook 04_incident_generation.ipynb.
    Field names match the notebook (anomaly_rate_pct, total_records, etc.)
    """

    def safe_value_counts(df: pd.DataFrame, col: str, n: int = 5) -> dict:
        if col not in df.columns:
            print(f"  Warning: column '{col}' not found — skipping.")
            return {}
        return df[col].value_counts().head(n).to_dict()

    def safe_mean(df: pd.DataFrame, col: str):
        if col not in df.columns:
            return None
        return float(df[col].mean())

    return {
        # Core stats
        "total_records":         len(df_all),
        "total_anomalies":       len(df_anomaly),
        "anomaly_rate_pct":      round(len(df_anomaly) / len(df_all) * 100, 2)
                                 if len(df_all) > 0 else 0.0,
        "threshold":             round(float(threshold), 6),

        # Error stats
        "max_error":             round(float(errors.max()), 6),
        "min_error":             round(float(errors.min()), 6),
        "mean_error_all":        round(float(errors.mean()), 6),
        "std_error_all":         round(float(errors.std()), 6),
        "mean_error_anomalies":  round(safe_mean(df_anomaly, "reconstruction_error") or 0, 6),

        # Severity
        "severity_breakdown":    safe_value_counts(df_anomaly, "severity", n=10),

        # Top offenders
        "top_anomaly_sources":   safe_value_counts(df_anomaly, "Source"),
        "top_anomaly_machines":  safe_value_counts(df_anomaly, "MachineName"),
        "top_anomaly_countries": safe_value_counts(df_anomaly, "country"),
        "top_entry_types":       safe_value_counts(df_anomaly, "EntryType"),
        "top_categories":        safe_value_counts(df_anomaly, "Category"),
        "top_isps":              safe_value_counts(df_anomaly, "isp"),

        # Time range
        "date_range_start":      str(df_all["TimeGenerated"].min())
                                 if "TimeGenerated" in df_all.columns else "N/A",
        "date_range_end":        str(df_all["TimeGenerated"].max())
                                 if "TimeGenerated" in df_all.columns else "N/A",
    }


# ── Prompt builders — mirrors notebook Cell 9 ─────────────────────────────────

def prompt_executive_summary(s: dict) -> str:
    top_sev = list(s["severity_breakdown"].keys())[0] if s["severity_breakdown"] else "Unknown"
    return (
        f"Write a 2-sentence executive summary for a CISO about a Windows security "
        f"anomaly detection report: "
        f"{s['total_anomalies']:,} anomalies detected ({s['anomaly_rate_pct']}% rate), "
        f"highest severity: {top_sev}, "
        f"from {len(s['top_anomaly_countries'])} countries. "
        f"Keep it professional and concise."
    )


def prompt_impact(s: dict) -> str:
    return (
        f"You are a cybersecurity analyst. Assess the business impact of this "
        f"Windows security incident: "
        f"{s['anomaly_rate_pct']}% anomaly rate "
        f"({s['total_anomalies']} events out of {s['total_records']}), "
        f"severity breakdown: {s['severity_breakdown']}, "
        f"top affected machines: {list(s['top_anomaly_machines'].keys())[:3]}, "
        f"top countries involved: {list(s['top_anomaly_countries'].keys())[:3]}. "
        f"Provide a 3-sentence impact assessment covering operational, data, "
        f"and reputational risk."
    )


def prompt_root_cause(s: dict, df_anomaly: pd.DataFrame) -> str:
    top_msgs = []
    if "Message" in df_anomaly.columns:
        top_msgs = (df_anomaly.nlargest(3, "reconstruction_error")["Message"]
                    .fillna("").str[:120].tolist())
    msg_sample = " | ".join(top_msgs) if top_msgs else "N/A"
    return (
        f"As a Windows security expert, identify possible root causes for these anomalies: "
        f"Top event sources: {list(s['top_anomaly_sources'].keys())[:3]}, "
        f"categories: {list(s['top_categories'].keys())[:3]}, "
        f"entry types with highest anomaly rate: {s['top_entry_types']}, "
        f"sample high-error log messages: {msg_sample}. "
        f"List the top 3 most likely root causes concisely with one sentence each."
    )


def prompt_remediation(s: dict) -> str:
    return (
        f"As a senior Windows system administrator, provide remediation steps "
        f"for a security incident: "
        f"anomaly rate {s['anomaly_rate_pct']}%, "
        f"top anomalous sources: {list(s['top_anomaly_sources'].keys())[:3]}, "
        f"severity distribution: {s['severity_breakdown']}, "
        f"countries involved: {list(s['top_anomaly_countries'].keys())[:3]}. "
        f"Give 4 numbered, actionable remediation steps with priority level (P1/P2/P3)."
    )


# ── Formatting helpers — mirrors notebook Cell 13 ─────────────────────────────

def _fmt_dict(d: dict, indent: int = 2) -> str:
    pad = " " * indent
    if not d:
        return f"{pad}(none)"
    return "\n".join(f"{pad}{str(k):<35} {v:>6}" for k, v in d.items())


def _fmt_top_anomalies(df: pd.DataFrame, n: int = 5) -> str:
    if df is None or len(df) == 0:
        return "  (none)"
    cols  = ["MachineName", "Source", "EntryType", "severity", "reconstruction_error"]
    avail = [c for c in cols if c in df.columns]
    top   = df.nlargest(n, "reconstruction_error")[avail]
    lines = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        lines.append(f"  [{i}] " + " | ".join(f"{c}: {row[c]}" for c in avail))
    return "\n".join(lines) if lines else "  (none)"


def _interpret_error(max_err: float, thresh: float) -> str:
    ratio = max_err / (thresh + 1e-8)
    if ratio > 10: return "EXTREME"
    if ratio > 5:  return "VERY HIGH"
    if ratio > 2:  return "HIGH"
    return "MODERATE"


# ── Main export ───────────────────────────────────────────────────────────────

def write_report(summary: dict, path: Path,
                 anomaly_df: "pd.DataFrame | None" = None) -> None:
    """
    Generate LLM narrative sections and write the full incident report to `path`.

    Parameters
    ----------
    summary   : dict from build_summary()
    path      : output .txt file Path
    anomaly_df: anomalous rows DataFrame (for TOP CRITICAL RECORDS section)
    """
    s = summary
    if anomaly_df is None:
        anomaly_df = pd.DataFrame()

    # ── LLM sections — mirrors notebook Cell 11 ───────────────────────────────
    print("[Report] Generating LLM sections...")

    print("  Executive Summary...")
    exec_summary = llm_generate(prompt_executive_summary(s))
    print(f"    -> {exec_summary[:100]}...")

    print("  Impact Assessment...")
    impact_text  = llm_generate(prompt_impact(s))
    print(f"    -> {impact_text[:100]}...")

    print("  Root Cause Analysis...")
    root_cause   = llm_generate(prompt_root_cause(s, anomaly_df))
    print(f"    -> {root_cause[:100]}...")

    print("  Remediation...")
    remediation  = llm_generate(prompt_remediation(s))
    print(f"    -> {remediation[:100]}...")

    print("[Report] All LLM sections generated!")

    # ── Assemble report — mirrors notebook Cell 13 ────────────────────────────
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    incident_id = datetime.now().strftime("INC-%Y%m%d-%H%M%S")

    report = f"""
{"=" * 80}
                    WINDOWS SECURITY INCIDENT REPORT
{"=" * 80}
Incident ID    : {incident_id}
Generated At   : {timestamp}
Classification : {"CRITICAL" if "CRITICAL" in s["severity_breakdown"] else "HIGH"}
Detection      : Autoencoder Anomaly Detection (PyTorch)
Status         : OPEN — Pending Investigation
{"=" * 80}

EXECUTIVE SUMMARY
{"─" * 69}
{exec_summary}

QUANTITATIVE OVERVIEW
{"─" * 69}
  Total Events Analyzed       : {s['total_records']:,}
  Anomalies Detected          : {s['total_anomalies']:,}
  Anomaly Rate                : {s['anomaly_rate_pct']}%
  Detection Threshold (MSE)   : {s['threshold']:.6f}
  Max Reconstruction Error    : {s['max_error']:.6f}  [{_interpret_error(s['max_error'], s['threshold'])} deviation]
  Min Reconstruction Error    : {s['min_error']:.6f}
  Mean Error (All)            : {s['mean_error_all']:.6f}
  Std Error (All)             : {s['std_error_all']:.6f}
  Mean Error (Anomalies)      : {s['mean_error_anomalies']:.6f}
  Date Range                  : {s['date_range_start']}  ->  {s['date_range_end']}

SEVERITY BREAKDOWN
{"─" * 69}
{_fmt_dict(s['severity_breakdown'])}

TOP AFFECTED MACHINES
{"─" * 69}
{_fmt_dict(s['top_anomaly_machines'])}

TOP ANOMALOUS EVENT SOURCES
{"─" * 69}
{_fmt_dict(s['top_anomaly_sources'])}

ENTRY TYPE DISTRIBUTION (Anomalies)
{"─" * 69}
{_fmt_dict(s['top_entry_types'])}

CATEGORY BREAKDOWN (Anomalies)
{"─" * 69}
{_fmt_dict(s['top_categories'])}

GEOGRAPHIC DISTRIBUTION
{"─" * 69}
{_fmt_dict(s['top_anomaly_countries'])}

ISP / NETWORK ORIGIN
{"─" * 69}
{_fmt_dict(s['top_isps'])}

IMPACT ASSESSMENT  [AI-Generated — Flan-T5]
{"─" * 69}
{impact_text}

ROOT CAUSE ANALYSIS  [AI-Generated — Flan-T5]
{"─" * 69}
{root_cause}

REMEDIATION RECOMMENDATIONS  [AI-Generated — Flan-T5]
{"─" * 69}
{remediation}

TOP CRITICAL ANOMALY RECORDS
{"─" * 69}
{_fmt_top_anomalies(anomaly_df, n=5)}

DETECTION METHODOLOGY
{"─" * 69}
  Model            : Symmetric Autoencoder (PyTorch)
  Architecture     : Input -> 128 -> 64 -> 32 -> 16 <- 32 <- 64 <- 128 <- Input
  Training Mode    : Unsupervised (learns normal behaviour)
  Anomaly Metric   : Per-sample MSE reconstruction error
  Threshold Policy : {ANOMALY_PERCENTILE}th percentile of training errors
  Feature Groups   : Temporal (11) + Categorical LE (4) + TF-IDF (50) + Msg Stats (7)
  LLM Narrative    : {HF_MODEL_NAME} (HuggingFace Transformers)

{"=" * 80}
                          END OF INCIDENT REPORT
{"=" * 80}
""".strip()

    path.write_text(report, encoding="utf-8")
    print(f"[Report] Saved: {path}")
