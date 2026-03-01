from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import uuid
from hoop_stats.analyzer import BasketballAnalyzer
from hoop_stats.game_state import GameState
from hoop_stats.stream_manager import StreamManager
from hoop_stats.basketball_rules import get_rules
from hoop_stats.agents.shot_clock_agent import ShotClockAgent
from hoop_stats.agents.game_clock_agent import GameClockAgent
from hoop_stats.agents.scoresheet_agent import ScoresheetAgent
from hoop_stats.agents.referee_agent import RefereeAgent

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

# ── Analysis Jobs (file upload mode) ────────────────────────────────────────
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
    output_name = f"results/stats_{file.filename.split('.')[0]}_{job_id[:8]}"
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

# ── Live Kampfgericht Mode ──────────────────────────────────────────────────
GAME_STATE: GameState = None
STREAM_MGR: StreamManager = None
AGENTS: dict = {}

AGENT_CLASSES = {
    "shot_clock": ShotClockAgent,
    "game_clock": GameClockAgent,
    "scoresheet": ScoresheetAgent,
    "referee":    RefereeAgent,
}

@app.post("/game/start")
async def start_game(config: dict):
    """
    Start a live Kampfgericht session.
    config = {
      "cameras": [{"id": "cam1", "source": "uploads/game.mp4"}, ...],
      "agents": ["shot_clock", "game_clock", "scoresheet", "referee"],
      "team_home": "Grizzlies",
      "team_away": "Uffenheim"
    }
    """
    global GAME_STATE, STREAM_MGR, AGENTS

    # Stop any existing session
    if STREAM_MGR:
        for a in AGENTS.values():
            a.stop()
        STREAM_MGR.stop_all()

    # Select ruleset (FIBA or NBA)
    ruleset_key = config.get("ruleset", "fiba")
    rules = get_rules(ruleset_key)

    GAME_STATE = GameState(rules=rules)
    GAME_STATE.team_home_name = config.get("team_home", "Team Schwarz")
    GAME_STATE.team_away_name = config.get("team_away", "Team Gelb")

    STREAM_MGR = StreamManager()
    cameras = config.get("cameras", [])
    camera_ids = []
    audio_ids = []
    for cam in cameras:
        STREAM_MGR.add_camera(cam["id"], cam["source"])
        camera_ids.append(cam["id"])
        # Also extract audio from each video source for whistle detection
        audio_id = f"audio_{cam['id']}"
        STREAM_MGR.add_audio(audio_id, cam["source"])
        audio_ids.append(audio_id)
    STREAM_MGR.start_all()

    AGENTS = {}
    requested_agents = config.get("agents", [])
    for agent_name in requested_agents:
        cls = AGENT_CLASSES.get(agent_name)
        if cls:
            agent = cls(GAME_STATE, STREAM_MGR, camera_ids=camera_ids, audio_ids=audio_ids)
            agent.start()
            AGENTS[agent_name] = agent

    return {
        "status": "started",
        "cameras": camera_ids,
        "agents": list(AGENTS.keys()),
    }

@app.get("/game/state")
async def get_game_state():
    if GAME_STATE is None:
        return {"status": "no_game"}
    snapshot = GAME_STATE.snapshot()
    snapshot["active_agents"] = {
        name: agent.is_active for name, agent in AGENTS.items()
    }
    return snapshot

@app.post("/game/stop")
async def stop_game():
    global GAME_STATE, STREAM_MGR, AGENTS
    if STREAM_MGR:
        for a in AGENTS.values():
            a.stop()
        STREAM_MGR.stop_all()
    AGENTS = {}
    STREAM_MGR = None
    return {"status": "stopped"}

@app.post("/game/toggle_agent")
async def toggle_agent(data: dict):
    """Toggle an agent on/off. data = {"agent": "shot_clock", "active": true}"""
    agent_name = data.get("agent")
    active = data.get("active", True)
    if agent_name in AGENTS:
        AGENTS[agent_name].set_active(active)
        return {"agent": agent_name, "active": active}
    return {"status": "error", "message": f"Agent '{agent_name}' not found"}

# ── Legacy stream endpoint ──────────────────────────────────────────────────
@app.post("/analyze/stream")
async def analyze_stream(url: str = Form(...)):
    return {"status": "processing", "message": "Stream analysis started", "url": url}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning", access_log=False)
