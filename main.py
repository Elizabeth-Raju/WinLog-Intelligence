"""
main.py — Windows Log Intelligence: FastAPI Backend
====================================================
Routes only. All ML logic lives in pipeline/.

  POST /upload          → upload CSV, start background pipeline job
  GET  /job/{job_id}    → poll status + results
  GET  /report/{job_id} → download incident report (.txt)
  GET  /health          → health check

Run:
    pip install -r requirements.txt
    python main.py
    # or:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── All ML logic imported from the pipeline package ───────────────────────────
from pipeline.detector import run_pipeline, DEVICE


# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
REPORT_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"

REPORT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ── In-memory job store  {job_id: {status, result, report_path, error}} ───────
JOBS: dict[str, dict] = {}


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Windows Log Intelligence", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/upload")
async def upload_log(background_tasks: BackgroundTasks,
                     file: UploadFile = File(...)):
    """Upload a CSV → triggers full pipeline in the background."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(400, "Uploaded CSV is empty.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "filename": file.filename, "rows": len(df)}

    background_tasks.add_task(run_pipeline, job_id, df, JOBS, REPORT_DIR)

    return {"job_id": job_id, "rows": len(df), "columns": list(df.columns)}


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    """Poll a background job for its current status and results."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    return {
        "job_id":   job_id,
        "status":   job["status"],
        "filename": job.get("filename"),
        "rows":     job.get("rows"),
        "result":   job.get("result"),
        "error":    job.get("error"),
    }


@app.get("/report/{job_id}")
async def download_report(job_id: str):
    """Download the generated incident report as a plain-text file."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    path = JOBS[job_id].get("report_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Report not ready yet.")
    return FileResponse(
        path,
        media_type="text/plain",
        filename=f"incident_{job_id[:8]}.txt",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Windows Log Intelligence — FastAPI Backend")
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
