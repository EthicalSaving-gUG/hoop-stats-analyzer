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
FRAMES_DIR = RESULTS_DIR / "frames"
ANNOTATIONS_DIR = RESULTS_DIR / "annotations"
LABELS_DIR = RESULTS_DIR / "labels"
REVIEW_STATE_FILE = RESULTS_DIR / "review_state.json"
CALIBRATION_FILE = RESULTS_DIR / "court_calibration.json"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REVIEW_STATUSES = {"pending", "accepted", "rejected", "needs_review"}

DEFAULT_CALIBRATION = {
    "video": [[100, 100], [1100, 100], [100, 900], [1100, 900]],
    "real": [[0, 0], [28, 0], [0, 15], [28, 15]],
}

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
FRAMES_DIR.mkdir(exist_ok=True)
ANNOTATIONS_DIR.mkdir(exist_ok=True)
LABELS_DIR.mkdir(exist_ok=True)

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

def _read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default

def _write_json_file(path: Path, data) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _load_review_state() -> dict:
    return _read_json_file(REVIEW_STATE_FILE, {"events": {}, "annotations": {}})

def _save_review_state(state: dict) -> None:
    _write_json_file(REVIEW_STATE_FILE, state)

def _load_calibration() -> dict:
    calibration = _read_json_file(CALIBRATION_FILE, DEFAULT_CALIBRATION)
    if "video" not in calibration or "real" not in calibration:
        return DEFAULT_CALIBRATION
    return calibration

def _quality_summary() -> dict:
    connected = 0
    cameras = {}
    if STREAM_MGR:
        for cam_id in STREAM_MGR.list_cameras():
            info = STREAM_MGR.get_info(cam_id)
            cameras[cam_id] = info
            if info.get("connected"):
                connected += 1

    warnings = []
    if STREAM_MGR and connected == 0:
        warnings.append("Keine Kameraquelle verbunden")
    if not Path(resolve_model_variant()).exists():
        warnings.append("Gewaehltes YOLO-Modell fehlt lokal")
    if grounding_status()["enabled"] is False:
        warnings.append("LocateAnything Grounding ist deaktiviert")

    event_count = len(GAME_STATE.events) if GAME_STATE else 0
    pending_reviews = 0
    if GAME_STATE:
        pending_reviews = sum(1 for event in GAME_STATE.events if event.review_status == "pending")
    review_state = _load_review_state()
    pending_reviews += sum(
        1 for item in review_state.get("annotations", {}).values()
        if item.get("review_status") == "pending"
    )

    return {
        "cameras": cameras,
        "connected_cameras": connected,
        "total_cameras": len(cameras),
        "events": event_count,
        "pending_reviews": pending_reviews,
        "jobs": JOBS,
        "warnings": warnings,
    }

def _event_to_review_dict(event, overrides: dict = None) -> dict:
    overrides = overrides or {}
    return {
        "id": event.event_id,
        "source": "live",
        "agent": event.agent,
        "type": event.event_type,
        "time": event.timestamp,
        "frame_ref": event.frame_ref,
        "data": event.data,
        "confidence": event.confidence,
        "evidence": event.evidence,
        "review_status": overrides.get("review_status", event.review_status),
        "review_note": overrides.get("review_note", ""),
        "source_camera": event.source_camera,
    }

def _offline_events_from_result(result_file: str) -> list[dict]:
    result_path = _safe_result_child(RESULTS_DIR, result_file, {".json"})
    if not result_path.exists():
        raise FileNotFoundError(result_file)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    events = []
    for idx, shot in enumerate(payload.get("shots", [])):
        event_id = f"{result_path.stem}:shot:{idx}"
        status = shot.get("status")
        events.append({
            "id": event_id,
            "source": "offline",
            "agent": "analyzer",
            "type": "SHOT_MAKE" if status == "made" else "SHOT_MISS",
            "time": None,
            "frame_ref": shot.get("start_frame"),
            "data": shot,
            "confidence": shot.get("confidence"),
            "evidence": {"result_file": result_path.name},
            "review_status": "pending",
            "review_note": "",
            "source_camera": None,
        })
    return events

def _resolve_video_source(source: str) -> Path:
    if not source:
        raise ValueError("Video source is required")
    if source.startswith("http://") or source.startswith("https://"):
        path_part = source.split("/uploads/", 1)[-1] if "/uploads/" in source else source
        candidate = UPLOAD_DIR / Path(path_part).name
    else:
        raw = Path(source)
        candidate = raw if raw.is_absolute() else (BASE_DIR / raw)
    candidate = candidate.resolve()
    allowed_roots = [UPLOAD_DIR.resolve(), RESULTS_DIR.resolve(), BASE_DIR.resolve()]
    if not any(root == candidate or root in candidate.parents for root in allowed_roots):
        raise ValueError("Video source is outside allowed workspace paths")
    if candidate.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video source")
    if not candidate.exists():
        raise FileNotFoundError(str(candidate))
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

    for idx, box in enumerate(boxes):
        box["id"] = f"box_{idx}"
        box["review_status"] = "pending"
        box["source"] = "grounding"

    annotation_path = ANNOTATIONS_DIR / f"grounding_{image_path.stem}_{uuid.uuid4().hex[:8]}.json"
    annotation["annotation_file"] = annotation_path.name
    annotation["annotation_url"] = str(request.url_for("results", path=f"annotations/{annotation_path.name}"))
    annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    return annotation

@app.post("/grounding/export-yolo/{annotation_file}")
async def export_grounding_yolo(annotation_file: str, class_id: int = 0, accepted_only: bool = False):
    try:
        annotation_path = _safe_result_child(ANNOTATIONS_DIR, annotation_file, {".json"})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="Annotation file not found")

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    boxes = annotation.get("boxes", [])
    if accepted_only:
        boxes = [box for box in boxes if box.get("review_status") == "accepted"]
    labels = boxes_to_yolo_labels(
        boxes,
        image_width=int(annotation["width"]),
        image_height=int(annotation["height"]),
        class_id=class_id,
    )
    LABELS_DIR.mkdir(exist_ok=True)
    label_path = LABELS_DIR / f"{annotation_path.stem}.txt"
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return {
        "status": "success",
        "annotation_file": annotation_path.name,
        "label_file": label_path.name,
        "labels": len(labels),
    }

@app.post("/grounding/annotation/{annotation_file}/boxes/{box_id}/review")
async def review_annotation_box(annotation_file: str, box_id: str, data: dict):
    status = data.get("review_status", "pending")
    if status not in REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail=f"Unsupported review status: {status}")
    try:
        annotation_path = _safe_result_child(ANNOTATIONS_DIR, annotation_file, {".json"})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="Annotation file not found")

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    for box in annotation.get("boxes", []):
        if box.get("id") == box_id:
            box["review_status"] = status
            if "label" in data:
                box["label"] = data["label"]
            if "review_note" in data:
                box["review_note"] = data["review_note"]
            annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")

            review_state = _load_review_state()
            review_state.setdefault("annotations", {})[f"{annotation_path.name}:{box_id}"] = {
                "review_status": status,
                "label": box.get("label"),
                "review_note": box.get("review_note", ""),
            }
            _save_review_state(review_state)
            return {"status": "success", "box": box}
    raise HTTPException(status_code=404, detail="Annotation box not found")

@app.get("/review/events")
async def list_review_events(result_file: str = None):
    review_state = _load_review_state()
    overrides = review_state.get("events", {})
    events = []
    if GAME_STATE:
        with GAME_STATE._lock:
            events.extend(_event_to_review_dict(event, overrides.get(event.event_id)) for event in GAME_STATE.events)
    if result_file:
        try:
            offline = _offline_events_from_result(result_file)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        for event in offline:
            event.update(overrides.get(event["id"], {}))
        events.extend(offline)
    return {"events": events, "count": len(events)}

@app.post("/review/events/{event_id}")
async def update_review_event(event_id: str, data: dict):
    status = data.get("review_status", "pending")
    if status not in REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail=f"Unsupported review status: {status}")

    review_state = _load_review_state()
    review_state.setdefault("events", {})[event_id] = {
        "review_status": status,
        "review_note": data.get("review_note", ""),
    }
    if GAME_STATE:
        with GAME_STATE._lock:
            for event in GAME_STATE.events:
                if event.event_id == event_id:
                    event.review_status = status
                    break
    _save_review_state(review_state)
    return {"status": "success", "event_id": event_id, "review_status": status}

@app.post("/video/extract-frame")
async def extract_frame(request: Request, data: dict):
    try:
        video_path = _resolve_video_source(data.get("video_url") or data.get("video_path") or data.get("source"))
        timestamp = float(data.get("timestamp", 0))
        frame_index = data.get("frame")
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video source")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if frame_index is None:
            frame_index = max(0, int(timestamp * fps))
        else:
            frame_index = max(0, int(frame_index))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Could not read requested frame")
        output_path = FRAMES_DIR / f"frame_{video_path.stem}_{frame_index}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(output_path), frame)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "success",
        "frame": frame_index,
        "timestamp": timestamp,
        "image": output_path.name,
        "image_url": str(request.url_for("results", path=f"frames/{output_path.name}")),
    }

@app.get("/calibration")
async def get_calibration():
    return _load_calibration()

@app.post("/calibration")
async def save_calibration(data: dict):
    video_points = data.get("video")
    real_points = data.get("real")
    if not isinstance(video_points, list) or not isinstance(real_points, list):
        raise HTTPException(status_code=400, detail="Calibration requires video and real point lists")
    if len(video_points) < 4 or len(real_points) < 4 or len(video_points) != len(real_points):
        raise HTTPException(status_code=400, detail="Calibration requires at least four matching points")
    calibration = {"video": video_points, "real": real_points}
    _write_json_file(CALIBRATION_FILE, calibration)
    return {"status": "success", "calibration": calibration}

@app.get("/quality")
async def quality():
    return _quality_summary()

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
    calibration = _load_calibration()
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
