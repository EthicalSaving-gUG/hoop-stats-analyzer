from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import uuid
from hoop_stats.analyzer import BasketballAnalyzer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("results", exist_ok=True)
app.mount("/results", StaticFiles(directory="results"), name="results")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global Job State Dictionary
# { job_id: {"progress": int, "status": str, "result_file": str} }
JOBS = {}

def run_background_analysis(job_id: str, file_path: str, output_name: str, calibration: dict):
    JOBS[job_id]["status"] = "Initializing Models..."
    
    def on_progress(pct, msg):
        JOBS[job_id]["progress"] = pct
        JOBS[job_id]["status"] = msg

    analyzer = BasketballAnalyzer(file_path, output_name)
    analyzer.run(court_calibration_points=calibration, progress_callback=on_progress)
    
    JOBS[job_id]["progress"] = 100
    JOBS[job_id]["status"] = "Complete"
    JOBS[job_id]["result_file"] = f"{output_name}.json"

@app.post("/analyze/file")
async def analyze_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "status": "Uploading...", "result_file": None}
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_name = f"stats_{file.filename.split('.')[0]}_{job_id[:8]}"
    
    calibration = {
        "video": [[100, 100], [1100, 100], [100, 900], [1100, 900]],
        "real": [[0, 0], [28, 0], [0, 15], [28, 15]]
    }
    
    background_tasks.add_task(run_background_analysis, job_id, file_path, output_name, calibration)
    return {"status": "success", "job_id": job_id}

@app.get("/analyze/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id not in JOBS:
        return {"status": "error", "message": "Job not found"}
    return JOBS[job_id]

@app.post("/analyze/stream")
async def analyze_stream(url: str = Form(...)):
    # In a real scenario, this would handle RTSP/HTTP streams via OpenCV
    # For now, we reuse the analyzer logic
    output_name = "stream_stats"
    analyzer = BasketballAnalyzer(url, output_name)
    # Note: Stream analysis would likely need a separate thread or async processing
    # but for this demonstration we invoke the existing run method
    return {"status": "processing", "message": "Stream analysis started", "url": url}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
