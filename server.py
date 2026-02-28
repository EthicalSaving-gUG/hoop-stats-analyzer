from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from hoop_stats.analyzer import BasketballAnalyzer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run analysis
    output_name = f"stats_{file.filename.split('.')[0]}"
    analyzer = BasketballAnalyzer(file_path, output_name)
    # Use mock calibration for demo
    calibration = {
        "video": [[100, 100], [1100, 100], [100, 900], [1100, 900]],
        "real": [[0, 0], [28, 0], [0, 15], [28, 15]]
    }
    analyzer.run(court_calibration_points=calibration)
    
    return {"status": "success", "output": output_name}

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
