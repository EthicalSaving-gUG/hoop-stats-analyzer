from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import uuid
import re
from hoop_stats.analyzer import BasketballAnalyzer
from hoop_stats.primitive_renderer import PrimitiveRenderer
from hoop_stats.game_state import GameState
from hoop_stats.stream_manager import StreamManager
from hoop_stats.basketball_rules import get_rules
from hoop_stats.agents.shot_clock_agent import ShotClockAgent
from hoop_stats.agents.game_clock_agent import GameClockAgent
from hoop_stats.agents.scoresheet_agent import ScoresheetAgent
from hoop_stats.agents.referee_agent import RefereeAgent
from hoop_stats.agents.pose_agent import PoseAnalysisAgent
from hoop_stats.agents.referee_signals_agent import RefereeSignalsAgent

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
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def secure_filename(filename: str) -> str:
    """Sanitize the filename to prevent path traversal and remove dangerous characters."""
    base_name = os.path.basename(filename)
    safe_name = re.sub(r'[^a-zA-Z0-9.\-_]', '_', base_name)
    safe_name = safe_name.strip('._')
    if not safe_name:
        safe_name = "uploaded_file"
    return safe_name

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

def run_background_visualize(job_id: str, file_path: str, output_path: str, mode: str):
    JOBS[job_id]["status"] = "Initializing renderer..."
    def on_progress(pct, msg):
        JOBS[job_id]["progress"] = pct
        JOBS[job_id]["status"] = msg
    renderer = PrimitiveRenderer(0, 0, mode=mode)  # render_video auto-adopts source resolution
    renderer.render_video(file_path, output_path, skip_frames=2, progress_callback=on_progress)
    JOBS[job_id]["progress"] = 100
    JOBS[job_id]["status"] = "Complete"
    JOBS[job_id]["result_file"] = output_path

@app.post("/analyze/file")
async def analyze_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "status": "Uploading...", "result_file": None}
    safe_name = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    output_name = f"results/stats_{safe_name.split('.')[0]}_{job_id[:8]}"
    calibration = {
        "video": [[100, 100], [1100, 100], [100, 900], [1100, 900]],
        "real": [[0, 0], [28, 0], [0, 15], [28, 15]]
    }
    background_tasks.add_task(run_background_analysis, job_id, file_path, output_name, calibration)
    return {"status": "success", "job_id": job_id}

@app.post("/analyze/visualize")
async def visualize_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("wireframe"),
):
    """Render a primitives-only visualization video. mode = 'wireframe' or 'overlay'."""
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "status": "Uploading...", "result_file": None}
    safe_name = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    output_path = f"results/viz_{safe_name.split('.')[0]}_{job_id[:8]}.mp4"
    background_tasks.add_task(run_background_visualize, job_id, file_path, output_path, mode)
    return {"status": "success", "job_id": job_id}

def run_background_export3d(job_id: str, file_path: str, output_path: str):
    JOBS[job_id]["status"] = "Initializing 3D export..."
    def on_progress(pct, msg):
        JOBS[job_id]["progress"] = pct
        JOBS[job_id]["status"] = msg
    renderer = PrimitiveRenderer(0, 0, mode="3d")
    renderer.export_3d_data(file_path, output_path, skip_frames=3, progress_callback=on_progress)
    JOBS[job_id]["progress"] = 100
    JOBS[job_id]["status"] = "Complete"
    JOBS[job_id]["result_file"] = output_path

@app.post("/analyze/export3d")
async def export3d_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Export per-frame 3D primitive vectors as JSON (no video rendering)."""
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "status": "Uploading...", "result_file": None}
    safe_name = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    output_path = f"results/vectors3d_{safe_name.split('.')[0]}_{job_id[:8]}.json"
    background_tasks.add_task(run_background_export3d, job_id, file_path, output_path)
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
    "pose":       PoseAnalysisAgent,
    "referee_signals": RefereeSignalsAgent,
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
    processing_fps = config.get("processing_fps", 10)

    for agent_name in requested_agents:
        cls = AGENT_CLASSES.get(agent_name)
        if cls:
            agent = cls(
                GAME_STATE, STREAM_MGR,
                camera_ids=camera_ids, audio_ids=audio_ids,
                tick_rate=processing_fps,
                pose_model=config.get("pose_model", "mediapipe")
            )
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

# ── Manual Stats Dashboard Mode ─────────────────────────────────────────────
from pydantic import BaseModel
from typing import List, Optional

class ManualPlayer(BaseModel):
    id: str
    name: str
    jersey: str
    team: str # "home" or "away"
    points_2pt: int = 0
    points_3pt: int = 0
    points_ft: int = 0
    fouls: int = 0
    rebounds_off: int = 0
    rebounds_def: int = 0
    assists: int = 0
    steals: int = 0
    blocks: int = 0
    turnovers: int = 0
    minutes: int = 0

class ManualState(BaseModel):
    team_home: str = "Team A"
    team_away: str = "Team B"
    players: List[ManualPlayer] = []
    video_url: Optional[str] = None

MANUAL_STATE = ManualState()

@app.post("/manual/start")
async def manual_start(data: dict):
    global MANUAL_STATE
    MANUAL_STATE = ManualState(
        team_home=data.get("team_home", "Team A"),
        team_away=data.get("team_away", "Team B"),
        players=[ManualPlayer(**p) for p in data.get("players", [])],
        video_url=data.get("video_url")
    )
    return {"status": "success"}

@app.get("/manual/state")
async def get_manual_state():
    return MANUAL_STATE.dict()

@app.post("/manual/update-player")
async def manual_update_player(data: dict):
    player_id = data.get("player_id")
    field = data.get("field")
    value = data.get("value", 0)

    for p in MANUAL_STATE.players:
        if p.id == player_id:
            setattr(p, field, getattr(p, field) + value)
            return {"status": "success", "player": p.dict()}
    return {"status": "error", "message": "Player not found"}

@app.post("/manual/add-player")
async def manual_add_player(player: ManualPlayer):
    MANUAL_STATE.players.append(player)
    return {"status": "success"}

@app.post("/manual/remove-player")
async def manual_remove_player(data: dict):
    player_id = data.get("player_id")
    MANUAL_STATE.players = [p for p in MANUAL_STATE.players if p.id != player_id]
    return {"status": "success"}

@app.get("/manual/export")
async def manual_export():
    return MANUAL_STATE.dict()

@app.post("/manual/upload-video")
async def manual_upload_video(file: UploadFile = File(...)):
    safe_name = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"video_url": f"http://localhost:8000/uploads/{safe_name}"}

# ── Legacy stream endpoint ──────────────────────────────────────────────────
@app.post("/analyze/stream")
async def analyze_stream(url: str = Form(...)):
    return {"status": "processing", "message": "Stream analysis started", "url": url}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning", access_log=False)
