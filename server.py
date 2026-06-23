from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import shutil
import os
import uuid
import re
import json
from pathlib import Path
import ultralytics
from hoop_stats.analyzer import BasketballAnalyzer
from hoop_stats.primitive_renderer import PrimitiveRenderer
from hoop_stats.game_state import GameState
from hoop_stats.stream_manager import StreamManager
from hoop_stats.basketball_rules import get_rules
from hoop_stats.detector import resolve_model_variant
from hoop_stats.grounding import (
    GroundingUnavailable,
    create_grounder,
    boxes_to_yolo_labels,
    grounding_status,
    parse_boxes,
    parse_points,
)
from hoop_stats.agents.shot_clock_agent import ShotClockAgent
from hoop_stats.agents.game_clock_agent import GameClockAgent
from hoop_stats.agents.scoresheet_agent import ScoresheetAgent
from hoop_stats.agents.referee_agent import RefereeAgent
from hoop_stats.agents.pose_agent import PoseAnalysisAgent
from hoop_stats.agents.referee_signals_agent import RefereeSignalsAgent

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ── Analysis Jobs (file upload mode) ────────────────────────────────────────
JOBS = {}
GROUNDER = None

@app.get("/health")
async def health():
    selected_model = resolve_model_variant()
    hoop_model = os.getenv("HOOP_STATS_HOOP_MODEL", "models/hoop_detector_yolo11.pt")
    grounding = grounding_status()
    return {
        "status": "ok",
        "ultralytics_version": ultralytics.__version__,
        "selected_model": selected_model,
        "selected_model_exists": Path(selected_model).exists(),
        "hoop_model": hoop_model,
        "hoop_model_exists": Path(hoop_model).exists(),
        "auto_export_enabled": os.getenv("HOOP_STATS_AUTO_EXPORT", "0") == "1",
        "grounding": grounding,
        "jobs": len(JOBS),
        "game_running": GAME_STATE is not None,
    }

def _safe_stem(filename: str) -> str:
    stem = Path(filename or "upload").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "upload"

def _safe_upload_path(filename: str) -> Path:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video extension '{ext or '<none>'}'. Allowed: {allowed}")
    return UPLOAD_DIR / f"{_safe_stem(filename)}_{uuid.uuid4().hex[:8]}{ext}"

def _store_upload(file: UploadFile) -> Path:
    file_path = _safe_upload_path(file.filename)
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

def _safe_image_path(filename: str) -> Path:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image extension '{ext or '<none>'}'. Allowed: {allowed}")
    return UPLOAD_DIR / f"{_safe_stem(filename)}_{uuid.uuid4().hex[:8]}{ext}"

def _store_image(file: UploadFile) -> Path:
    file_path = _safe_image_path(file.filename)
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(image_path) as image:
            return image.size
    except ImportError:
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Could not read uploaded image")
        height, width = image.shape[:2]
        return width, height

def _get_grounder():
    global GROUNDER
    if GROUNDER is None:
        GROUNDER = create_grounder()
    return GROUNDER

def _safe_result_child(base_dir: Path, filename: str, allowed_suffixes: set[str]) -> Path:
    candidate = (base_dir / Path(filename or "").name).resolve()
    base = base_dir.resolve()
    if base not in candidate.parents or candidate.suffix.lower() not in allowed_suffixes:
        raise ValueError("Unsupported result file")
    return candidate

@app.get("/grounding/status")
async def get_grounding_status():
    return grounding_status()

@app.post("/grounding/image")
async def ground_image(
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(...),
    task: str = Form("ground_multi"),
    mode: str = Form("hybrid"),
):
    try:
        image_path = _store_image(file)
        width, height = _image_size(image_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = _get_grounder().locate(image_path, task=task, query=query, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GroundingUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    answer = result.get("answer", "")
    boxes = [box.as_dict() for box in parse_boxes(answer, width, height)]
    points = [point.as_dict() for point in parse_points(answer, width, height)]
    annotation = {
        "image": image_path.name,
        "image_url": str(request.url_for("uploads", path=image_path.name)),
        "width": width,
        "height": height,
        "query": query,
        "task": task,
        "mode": mode,
        "prompt": result.get("prompt"),
        "backend": result.get("backend"),
        "answer": answer,
        "boxes": boxes,
        "points": points,
    }

    annotations_dir = RESULTS_DIR / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    annotation_path = annotations_dir / f"grounding_{image_path.stem}_{uuid.uuid4().hex[:8]}.json"
    annotation["annotation_file"] = annotation_path.name
    annotation["annotation_url"] = str(request.url_for("results", path=f"annotations/{annotation_path.name}"))
    annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    return annotation

@app.post("/grounding/export-yolo/{annotation_file}")
async def export_grounding_yolo(annotation_file: str, class_id: int = 0):
    annotations_dir = RESULTS_DIR / "annotations"
    labels_dir = RESULTS_DIR / "labels"
    try:
        annotation_path = _safe_result_child(annotations_dir, annotation_file, {".json"})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="Annotation file not found")

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    labels = boxes_to_yolo_labels(
        annotation.get("boxes", []),
        image_width=int(annotation["width"]),
        image_height=int(annotation["height"]),
        class_id=class_id,
    )
    labels_dir.mkdir(exist_ok=True)
    label_path = labels_dir / f"{annotation_path.stem}.txt"
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return {
        "status": "success",
        "annotation_file": annotation_path.name,
        "label_file": label_path.name,
        "labels": len(labels),
    }

def _mark_job_failed(job_id: str, exc: Exception):
    JOBS[job_id]["progress"] = 100
    JOBS[job_id]["status"] = "Failed"
    JOBS[job_id]["error"] = str(exc)
    JOBS[job_id]["result_file"] = None

def run_background_analysis(job_id: str, file_path: str, output_name: str, calibration: dict):
    try:
        JOBS[job_id]["status"] = "Initializing Models..."
        def on_progress(pct, msg):
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["status"] = msg
        analyzer = BasketballAnalyzer(file_path, output_name)
        analyzer.run(court_calibration_points=calibration, progress_callback=on_progress)
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"] = "Complete"
        JOBS[job_id]["result_file"] = Path(f"{output_name}.json").name
    except Exception as exc:
        _mark_job_failed(job_id, exc)

def run_background_visualize(job_id: str, file_path: str, output_path: str, mode: str):
    try:
        JOBS[job_id]["status"] = "Initializing renderer..."
        def on_progress(pct, msg):
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["status"] = msg
        renderer = PrimitiveRenderer(0, 0, mode=mode)  # render_video auto-adopts source resolution
        renderer.render_video(file_path, output_path, skip_frames=2, progress_callback=on_progress)
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"] = "Complete"
        JOBS[job_id]["result_file"] = Path(output_path).name
    except Exception as exc:
        _mark_job_failed(job_id, exc)

@app.post("/analyze/file")
async def analyze_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        file_path = _store_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "status": "Uploading...", "result_file": None, "error": None}
    output_name = str(RESULTS_DIR / f"stats_{file_path.stem}_{job_id[:8]}")
    calibration = {
        "video": [[100, 100], [1100, 100], [100, 900], [1100, 900]],
        "real": [[0, 0], [28, 0], [0, 15], [28, 15]]
    }
    background_tasks.add_task(run_background_analysis, job_id, str(file_path), output_name, calibration)
    return {"status": "success", "job_id": job_id}

@app.post("/analyze/visualize")
async def visualize_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("wireframe"),
):
    """Render a primitives-only visualization video. mode = 'wireframe' or 'overlay'."""
    try:
        file_path = _store_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "status": "Uploading...", "result_file": None, "error": None}
    output_path = str(RESULTS_DIR / f"viz_{file_path.stem}_{job_id[:8]}.mp4")
    background_tasks.add_task(run_background_visualize, job_id, str(file_path), output_path, mode)
    return {"status": "success", "job_id": job_id}

def run_background_export3d(job_id: str, file_path: str, output_path: str):
    try:
        JOBS[job_id]["status"] = "Initializing 3D export..."
        def on_progress(pct, msg):
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["status"] = msg
        renderer = PrimitiveRenderer(0, 0, mode="3d")
        renderer.export_3d_data(file_path, output_path, skip_frames=3, progress_callback=on_progress)
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"] = "Complete"
        JOBS[job_id]["result_file"] = Path(output_path).name
    except Exception as exc:
        _mark_job_failed(job_id, exc)

@app.post("/analyze/export3d")
async def export3d_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Export per-frame 3D primitive vectors as JSON (no video rendering)."""
    try:
        file_path = _store_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "status": "Uploading...", "result_file": None, "error": None}
    output_path = str(RESULTS_DIR / f"vectors3d_{file_path.stem}_{job_id[:8]}.json")
    background_tasks.add_task(run_background_export3d, job_id, str(file_path), output_path)
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
    if STREAM_MGR:
        snapshot["cameras"] = {
            cam_id: STREAM_MGR.get_info(cam_id) for cam_id in STREAM_MGR.list_cameras()
        }
    else:
        snapshot["cameras"] = {}
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

@app.get("/game/highlights")
async def get_highlights(request: Request):
    """
    Scan GAME_STATE.events for SHOT_MAKE or VIOLATION_ALERT events,
    slice the source video file around those frames, save to results/highlights/,
    and return metadata with download URLs.
    """
    if GAME_STATE is None:
        return {"status": "no_game", "highlights": []}

    highlights_dir = RESULTS_DIR / "highlights"
    highlights_dir.mkdir(exist_ok=True)

    # Find the video source from active cameras, or fallback to uploads
    video_source = None
    if STREAM_MGR and hasattr(STREAM_MGR, "_cameras") and STREAM_MGR._cameras:
        first_cam = list(STREAM_MGR._cameras.values())[0]
        video_source = first_cam.source
    else:
        # Check files in uploads
        if UPLOAD_DIR.exists():
            files = [str(UPLOAD_DIR / f) for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".mp4")]
            if files:
                video_source = files[0]

    if not video_source or not os.path.exists(video_source):
        # Fallback to standard names
        for path in [UPLOAD_DIR / "game.mp4", UPLOAD_DIR / "GebrunnSpiel.mp4"]:
            if path.exists():
                video_source = str(path)
                break

    if not video_source or not os.path.exists(video_source):
        return {"status": "error", "message": "No video source found to slice highlights", "highlights": []}

    highlights_list = []
    
    with GAME_STATE._lock:
        events = list(GAME_STATE.events)

    import cv2
    for idx, event in enumerate(events):
        if event.event_type not in ("SHOT_MAKE", "VIOLATION_ALERT", "REFEREE_WHISTLE"):
            continue
        if event.frame_ref is None:
            continue

        frame_ref = event.frame_ref
        event_type = event.event_type
        
        # Format a descriptive name
        desc = ""
        if event_type == "SHOT_MAKE":
            player = event.data.get("shooter_id", "Unbekannt")
            team = event.data.get("team", "home")
            team_name = GAME_STATE.team_home_name if team == "home" else GAME_STATE.team_away_name
            zone = event.data.get("zone", "Unbekannt")
            desc = f"Treffer von Spieler {player} ({team_name}) aus Zone '{zone}'"
        elif event_type == "VIOLATION_ALERT":
            v_type = event.data.get("type", "Regelverstoss")
            player = event.data.get("player_id", "Unbekannt")
            team = event.data.get("team", "home")
            team_name = GAME_STATE.team_home_name if team == "home" else GAME_STATE.team_away_name
            desc = f"Regelverstoss: {v_type} - Spieler {player} ({team_name})"
        else:
            desc = event.data.get("hint", "Schiedsrichter Pfiff")

        output_filename = f"highlight_{event_type}_{idx}_{frame_ref}.mp4"
        output_path = str(highlights_dir / output_filename)
        
        # Only slice if file doesn't exist yet (to speed up repeated calls)
        if not os.path.exists(output_path):
            cap = cv2.VideoCapture(video_source)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # 3s before, 2s after (assuming 30fps)
                frames_before = int(3 * fps)
                frames_after = int(2 * fps)
                
                start_frame = max(0, frame_ref - frames_before)
                end_frame = min(total_frames - 1, frame_ref + frames_after) if total_frames > 0 else (frame_ref + frames_after)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                current_frame = start_frame
                while current_frame <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    current_frame += 1
                
                out.release()
                cap.release()

        # If it was sliced successfully (or already exists)
        if os.path.exists(output_path):
            video_url = str(request.url_for("results", path=f"highlights/{output_filename}"))
            highlights_list.append({
                "id": idx,
                "type": event_type,
                "frame_ref": frame_ref,
                "video_url": video_url,
                "description": desc,
                "timestamp": event.timestamp
            })

    return {"status": "success", "highlights": highlights_list}


import time

@app.post("/game/voice-command")
async def voice_command(data: dict):
    """
    Processes a verbal override command parsed by the frontend,
    e.g., {"command": "foul_home", "player": "4"}, {"command": "pause_clock"}, etc.
    """
    global GAME_STATE
    if GAME_STATE is None:
        return {"status": "error", "message": "No active game"}
    
    cmd = data.get("command")
    player = data.get("player")
    
    with GAME_STATE._lock:
        if cmd == "foul_home":
            GAME_STATE.add_foul("home", player)
            GAME_STATE.append_event("voice_hud", "VOICE_OVERRIDE_FOUL", {"team": "home", "player": player, "message": f"Foul Heim Spieler {player}"})
        elif cmd == "foul_away":
            GAME_STATE.add_foul("away", player)
            GAME_STATE.append_event("voice_hud", "VOICE_OVERRIDE_FOUL", {"team": "away", "player": player, "message": f"Foul Gast Spieler {player}"})
        elif cmd == "toggle_clock":
            GAME_STATE.game_clock_running = not GAME_STATE.game_clock_running
            GAME_STATE.append_event("voice_hud", "VOICE_OVERRIDE_CLOCK", {"running": GAME_STATE.game_clock_running, "message": "Uhr gestartet" if GAME_STATE.game_clock_running else "Uhr gestoppt"})
        elif cmd == "timeout":
            GAME_STATE.game_clock_running = False
            GAME_STATE.append_event("voice_hud", "VOICE_OVERRIDE_TIMEOUT", {"message": "Auszeit genommen"})
        elif cmd == "sub":
            # Open substitution window for 15s
            GAME_STATE.substitution_window_open_until = time.time() + 15.0
            GAME_STATE.append_event("voice_hud", "VOICE_OVERRIDE_SUB", {"message": "Auswechselungsfenster geoeffnet"})
            
    return {"status": "success", "game_state": GAME_STATE.snapshot()}

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
MANUAL_STAT_FIELDS = {
    "points_2pt",
    "points_3pt",
    "points_ft",
    "fouls",
    "rebounds_off",
    "rebounds_def",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "minutes",
}

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
    if field not in MANUAL_STAT_FIELDS:
        raise HTTPException(status_code=400, detail=f"Unsupported stat field: {field}")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Stat update value must be an integer") from exc
    
    for p in MANUAL_STATE.players:
        if p.id == player_id:
            setattr(p, field, max(0, getattr(p, field) + value))
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
async def manual_upload_video(request: Request, file: UploadFile = File(...)):
    try:
        file_path = _store_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"video_url": str(request.url_for("uploads", path=file_path.name))}

# ── Legacy stream endpoint ──────────────────────────────────────────────────
@app.post("/analyze/stream")
async def analyze_stream(url: str = Form(...)):
    return {"status": "processing", "message": "Stream analysis started", "url": url}

# ── WebRTC Signaling Server ──────────────────────────────────────────────────
active_broadcasters = {}  # peer_id -> WebSocket
active_viewers = {}        # peer_id -> WebSocket

@app.websocket("/ws/webrtc")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    peer_id = str(uuid.uuid4())[:8]
    role = None
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "register":
                role = data.get("role")
                if role == "broadcaster":
                    active_broadcasters[peer_id] = websocket
                    await websocket.send_json({"type": "registered", "peer_id": peer_id})
                    # Notify existing viewers
                    for v_id, v_ws in active_viewers.items():
                        try:
                            await v_ws.send_json({"type": "broadcaster_connected", "peer_id": peer_id})
                        except Exception:
                            pass
                elif role == "viewer":
                    active_viewers[peer_id] = websocket
                    await websocket.send_json({"type": "registered", "peer_id": peer_id})
                    # Send list of active broadcasters
                    await websocket.send_json({
                        "type": "broadcasters_list",
                        "broadcasters": list(active_broadcasters.keys())
                    })
            
            elif msg_type == "offer":
                target_id = data.get("target_id")
                payload = data.get("payload")
                if target_id in active_broadcasters:
                    await active_broadcasters[target_id].send_json({
                        "type": "offer",
                        "sender_id": peer_id,
                        "payload": payload
                    })
            
            elif msg_type == "answer":
                target_id = data.get("target_id")
                payload = data.get("payload")
                if target_id in active_viewers:
                    await active_viewers[target_id].send_json({
                        "type": "answer",
                        "sender_id": peer_id,
                        "payload": payload
                    })
            
            elif msg_type == "candidate":
                target_id = data.get("target_id")
                payload = data.get("payload")
                # Route candidate to target
                if target_id in active_broadcasters:
                    await active_broadcasters[target_id].send_json({
                        "type": "candidate",
                        "sender_id": peer_id,
                        "payload": payload
                    })
                elif target_id in active_viewers:
                    await active_viewers[target_id].send_json({
                        "type": "candidate",
                        "sender_id": peer_id,
                        "payload": payload
                    })
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        # Cleanup
        if role == "broadcaster":
            if peer_id in active_broadcasters:
                del active_broadcasters[peer_id]
            # Notify viewers
            for v_id, v_ws in active_viewers.items():
                try:
                    await v_ws.send_json({"type": "broadcaster_disconnected", "peer_id": peer_id})
                except Exception:
                    pass
        elif role == "viewer":
            if peer_id in active_viewers:
                del active_viewers[peer_id]

@app.get("/mobile")
async def get_mobile():
    return FileResponse("mobile.html")

@app.get("/viewer")
async def get_viewer():
    return FileResponse("viewer.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning", access_log=False)
