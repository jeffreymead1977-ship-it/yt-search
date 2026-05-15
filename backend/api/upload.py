"""E57 upload endpoint with background segmentation job management."""
import uuid
import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api")

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store (replace with DB for production)
_jobs: dict[str, dict] = {}


def _run_segmentation(job_id: str, e57_path: Path) -> None:
    try:
        # Imports here so startup is fast even if open3d is slow to load
        from processing.e57_loader import load_e57
        from processing.segmenter import segment
        from processing.exporter import export_labeled_ply, export_summary

        _jobs[job_id]["status"] = "processing"

        xyz, metadata = load_e57(e57_path)
        result = segment(xyz)
        result.scan_metadata = metadata

        ply_path = OUTPUT_DIR / f"{job_id}.ply"
        json_path = OUTPUT_DIR / f"{job_id}.json"

        export_labeled_ply(result, ply_path)
        summary = export_summary(result, json_path)

        _jobs[job_id].update({
            "status": "done",
            "ply_path": str(ply_path),
            "summary": summary,
        })
    except Exception as exc:
        _jobs[job_id].update({"status": "error", "error": str(exc)})


@router.post("/upload/e57")
async def upload_e57(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".e57"):
        raise HTTPException(status_code=400, detail="Only .e57 files are accepted")

    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}.e57"

    content = await file.read()
    save_path.write_bytes(content)

    _jobs[job_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(_run_segmentation, job_id, save_path)

    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/download")
async def download_ply(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")
    return FileResponse(job["ply_path"], filename=f"tank_{job_id}.ply", media_type="application/octet-stream")
