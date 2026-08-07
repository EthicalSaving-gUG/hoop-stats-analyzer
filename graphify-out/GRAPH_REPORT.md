# Graph Report - .  (2026-07-09)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 859 nodes · 1886 edges · 51 communities (46 shown, 5 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 308 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `546ab4e1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- PrimitiveRenderer
- ShotTracker
- devDependencies
- App.jsx
- StreamManager
- RimDetector
- server.py
- FramePrescreener
- GameState
- grounding.py
- MultiRoiTracker
- PlayerCapsule
- CourtGeometry
- Camera3D
- __init__.py
- Point2D
- HoopDetector
- Path
- get_dominant_jersey_color
- Sphere3D
- primitives.py
- Ball
- ManualPlayer
- BasketballAnalyzer
- _segment_worker
- primitive_renderer.py
- conftest.py
- BaseAgent
- RefereeAgent
- Point3D
- PoseAnalysisAgent
- ground_image
- .render_frame_3d
- WhistleDetector
- bbox_to_player
- Cylinder3D
- base_agent.py
- Backboard
- dependencies
- RefereeSignalsAgent
- ScoresheetAgent
- TestHoopDetector
- bbox_to_ball
- TestDataExporter
- TestJerseyOCR
- ball_near_player
- TestCamera3DCalibration

## God Nodes (most connected - your core abstractions)
1. `Point2D` - 98 edges
2. `Ball` - 70 edges
3. `PlayerCapsule` - 64 edges
4. `Rim` - 62 edges
5. `PrimitiveRenderer` - 50 edges
6. `Point3D` - 47 edges
7. `ShotTracker` - 44 edges
8. `Camera3D` - 41 edges
9. `HoopDetector` - 30 edges
10. `Cylinder3D` - 30 edges

## Surprising Connections (you probably didn't know these)
- `ManualPlayer` --uses--> `PoseAnalysisAgent`  [INFERRED]
  server.py → hoop_stats/agents/pose_agent.py
- `ManualState` --uses--> `PoseAnalysisAgent`  [INFERRED]
  server.py → hoop_stats/agents/pose_agent.py
- `ManualPlayer` --uses--> `RefereeAgent`  [INFERRED]
  server.py → hoop_stats/agents/referee_agent.py
- `ManualState` --uses--> `RefereeAgent`  [INFERRED]
  server.py → hoop_stats/agents/referee_agent.py
- `ManualPlayer` --uses--> `RefereeSignalsAgent`  [INFERRED]
  server.py → hoop_stats/agents/referee_signals_agent.py

## Import Cycles
- None detected.

## Communities (51 total, 5 thin omitted)

### Community 0 - "PrimitiveRenderer"
Cohesion: 0.07
Nodes (25): PrimitiveRenderer, ndarray, Draw all court lines., Draw the ball as a filled circle with a soft glow., Draw fading trail behind the ball., Draw a player capsule as a rounded rectangle (tube)., Draw the rim as a circle outline with glow., Draw a heads-up display bar at the top. (+17 more)

### Community 1 - "ShotTracker"
Cohesion: 0.07
Nodes (23): Auto-trigger halftime swap if we've reached the halftime frame., balls: list of Ball primitives         player_capsules: dict of {player_id: Pla, Looks for a strong upward movement pattern indicating a shot release., Detects basketball jump shots, arcs, and makes/misses.      Strategy:     1., Update rim positions from RimDetector output., Swap which basket each team attacks (called at halftime)., Return 'home' or 'away' based on which team scores into this basket., ShotTracker (+15 more)

### Community 2 - "devDependencies"
Cohesion: 0.06
Nodes (33): dependencies, axios, framer-motion, lucide-react, @mediapipe/tasks-vision, onnxruntime-web, protobufjs, react (+25 more)

### Community 3 - "App.jsx"
Cohesion: 0.09
Nodes (16): App(), ClientAIBenchmark, AGENT_META, AgentCard(), EXAMPLE_PROMPTS, GroundingReview(), TASKS, LiveDashboard() (+8 more)

### Community 4 - "StreamManager"
Cohesion: 0.07
Nodes (11): AudioReader, CameraReader, ndarray, Multi-camera stream manager with audio support. Opens each camera source in a d, Extract audio from a video file using ffmpeg subprocess., Record from system microphone using sounddevice., Continuously reads frames from a single video source., Returns the latest audio chunk or None. (+3 more)

### Community 5 - "RimDetector"
Cohesion: 0.09
Nodes (23): ndarray, rim_detector.py =============== Detects basketball rims/hoops in video frames, Assign candidates to left/right rim and update EMA positions., Apply exponential moving average to rim position., Return (left_confidence, right_confidence)., Detects basketball rims in video frames using color + shape analysis.      The, True if at least one rim has been detected., frame_w, frame_h: video dimensions         smoothing: EMA factor (0 = no update (+15 more)

### Community 6 - "server.py"
Cohesion: 0.10
Nodes (22): resolve_model_variant(), _event_to_review_dict(), export_grounding_yolo(), get_calibration(), health(), list_review_events(), _load_calibration(), _load_review_state() (+14 more)

### Community 7 - "FramePrescreener"
Cohesion: 0.09
Nodes (18): FramePrescreener, ndarray, prescreener.py ============== Lightweight pre-screening pass applied to every, Reset internal state (call after a scene cut or seek)., Summarise a list of ScreenResults for logging / benchmarking., Fast per-frame quality and activity filter.      Parameters     ----------, ScreenResult, test_prescreener.py — tests for hoop_stats.prescreener (+10 more)

### Community 8 - "GameState"
Cohesion: 0.09
Nodes (19): BasketballRules, get_rules(), Basketball Rule Sets — FIBA 2024 and NBA 2024/25.  Provides dataclass-based ru, All rule parameters that differ between FIBA and NBA., Return a rule set by key ('fiba' or 'nba')., GameEvent, GameState, Thread-safe shared game state and event bus for multi-agent Kampfgericht. All a (+11 more)

### Community 9 - "grounding.py"
Cohesion: 0.12
Nodes (26): Any, boxes_to_yolo_labels(), build_prompt(), create_grounder(), grounding_status(), GroundingBox, GroundingPoint, GroundingUnavailable (+18 more)

### Community 10 - "MultiRoiTracker"
Cohesion: 0.12
Nodes (13): _create_opencv_tracker(), MultiRoiTracker, roi_tracker.py ============== Multi-object optical tracker with OpenCV tracker s, Initializes a CSRT tracker for a given bounding box.         bbox_xyxy: [x1, y1, Updates all tracked objects with the new frame.         Returns a dict of succe, Remove all trackers (e.g., before refreshing from YOLO)., Return the best available OpenCV tracker, or None if none are installed., _TemplateBackedTracker (+5 more)

### Community 11 - "PlayerCapsule"
Cohesion: 0.14
Nodes (16): Crops the player's torso area where the jersey number is likely.         Accept, ball_distance(), capsule_distance(), CourtLine, PlayerCapsule, Horizontal center (same as bbox center-X)., A single court marking line segment., Euclidean distance from a ball's center to a point. (+8 more)

### Community 12 - "CourtGeometry"
Cohesion: 0.12
Nodes (9): Scoresheet Agent — Kampfbericht (Anschreiber). Records all baskets, fouls, subs, CourtGeometry, Loads physical court zones from JSON and converts them to OpenCV contours., Calculates the homography matrix.         :param video_points: List of (x, y) p, Transforms pixel (x, y) to real-world (Mx, My)., Handles the transformation between video pixel coordinates and real-world basket, Calculates distance in meters between two transformed points., Given a real-world coordinate (mx, my) in meters, returns the name of the shot z (+1 more)

### Community 13 - "Camera3D"
Cohesion: 0.11
Nodes (13): Camera3D, Convenience: calibrate from 4 visible court floor corners.         Order: top-l, Project a pixel (px, py) onto the ground plane (Y = ground_y).         Returns, Estimate the real-world height (meters) of an object given its foot         and, Project a 3D world point to pixel coordinates in this camera., Lift a 2D PlayerCapsule to a 3D Cylinder., Lift a 2D Ball to a 3D Sphere.          If nearby_players are given, estimate, Create a 3D Ring for the basket rim at its known real-world position.         s (+5 more)

### Community 14 - "__init__.py"
Cohesion: 0.15
Nodes (8): DataExporter, Export everything including shots., JerseyOCR, Processes a crop of a player and attempts to find a number., exporter(), test_exporter.py — tests for hoop_stats.exporter (DataExporter), ocr(), test_ocr.py — tests for hoop_stats.ocr (JerseyOCR)

### Community 15 - "Point2D"
Cohesion: 0.13
Nodes (8): Point2D, The foot point — used for court-coordinate transforms., A 2-D point in pixel or meter space., update_position still accepts (mx, my) tuples., Player moves 1 m per frame at 30 fps → speed should ≈ 30 m/s., TestSpeedCalculation, TestUpdatePosition, TestPoint2D

### Community 16 - "HoopDetector"
Cohesion: 0.13
Nodes (10): Game Clock Agent — Spielzeiterfassung. Manages the game clock (4×10min quarters, 24-Sekunden Shot Clock Agent. Tracks ball possession and manages the 24-second, HoopDetector, Detects players and returns raw bounding boxes [x1, y1, x2, y2].         Kept f, Detects sports balls and returns raw bounding boxes.         Kept for backward, Detects basketball hoops (class 1 in our custom model)         and returns Rim p, bbox_to_rim(), Convert a bounding rectangle to a Rim circle (inscribed). (+2 more)

### Community 17 - "Path"
Cohesion: 0.20
Nodes (19): BackgroundTasks, Exception, analyze_file(), export3d_file(), manual_upload_video(), _mark_job_failed(), Path, Render a primitives-only visualization video. mode = 'wireframe' or 'overlay'. (+11 more)

### Community 18 - "get_dominant_jersey_color"
Cohesion: 0.15
Nodes (12): assign_teams_by_color(), _classify_color_name(), get_dominant_jersey_color(), Map a single HSV pixel to a color name., Crops the jersey (torso area) and returns the dominant color name., Given {player_id: color_name}, groups players into at most 2 teams.     Returns, test_team_colors.py — tests for hoop_stats.team_colors, Degenerate crop (zero size) should not crash – returns 'Unknown'. (+4 more)

### Community 19 - "Sphere3D"
Cohesion: 0.20
Nodes (11): Ball in 3D space — a sphere., Basket rim in 3D space — a horizontal ring at a fixed height., Ring3D, Sphere3D, Tests for camera3d module — 3D camera calibration and primitive lifting., Middle of the frame should map roughly to the court center., TestHeightEstimation, TestLiftBall (+3 more)

### Community 20 - "primitives.py"
Cohesion: 0.15
Nodes (7): PlayerMetrics, Update a player's position.         pos: Point2D (meter-space coordinates) or (, Returns simplified movement points for the player as (x, y) tuples., primitives.py ============= Lightweight geometric primitives for representing, metrics(), test_metrics.py — pure unit tests for hoop_stats.metrics (no video needed), TestGetTrajectoryLines

### Community 21 - "Ball"
Cohesion: 0.24
Nodes (8): Ball, Returns True if the ball overlaps the rim circle     (center-to-center distance, Basketball reduced to a center point and radius., Basket hoop reduced to a 2-D circle., Rim, rim_contains_ball(), TestLiftRim, TestRimContainsBall

### Community 22 - "ManualPlayer"
Cohesion: 0.16
Nodes (8): BaseModel, GameClockAgent, Whistle detected → stop clocks (dead ball)., ShotClockAgent, manual_add_player(), manual_start(), ManualPlayer, ManualState

### Community 23 - "BasketballAnalyzer"
Cohesion: 0.21
Nodes (8): BasketballAnalyzer, Accelerated pipeline that splits the video into segments processed in, main(), test_integration.py =================== End-to-end integration test: runs Bask, run_fast() on the first 300 frames should produce:     - a .json file that is v, Speed benchmark: run_fast() should be at least 1.5× faster than run()     on th, test_run_fast_is_faster_than_run(), test_run_fast_produces_valid_output()

### Community 24 - "_segment_worker"
Cohesion: 0.16
Nodes (11): Processes one video segment using Detect-Validate-Track paradigm.     Runs YOLO, _segment_worker(), Process an entire video file end-to-end and write the primitive view         as, Process a video and export per-frame 3D primitive vectors as JSON.         No v, get_segment_ranges(), get_video_info(), video_utils.py ============== Helpers for splitting a video into segments so e, Return basic metadata for the video file. (+3 more)

### Community 25 - "primitive_renderer.py"
Cohesion: 0.16
Nodes (9): make_virtual_camera(), ndarray, camera3d.py =========== Camera pose estimation and 2D → 3D lifting for the bas, Project a 3D world point to a different (virtual) camera., Create virtual camera parameters (rvec, tvec, K) for a preset viewing angle., primitive_renderer.py ===================== Renders simplified geometric views, Backboard in 3D — a vertical rectangle., Rect3D (+1 more)

### Community 26 - "conftest.py"
Cohesion: 0.14
Nodes (13): black_frame(), first_frame(), noisy_frame(), conftest.py — shared pytest fixtures for hoop-stats-analyzer tests., Return path to the upload video, or skip the test if not present., Extract and return the very first frame of the upload video., A completely black 720×1280 BGR frame (no motion, very blurry)., A completely white 720×1280 BGR frame. (+5 more)

### Community 27 - "BaseAgent"
Cohesion: 0.22
Nodes (5): BaseAgent, Abstract agent. Subclasses implement `process_frame()`.     The run-loop grabs, Process a single frame from one camera. Implement in subclass., Called when a referee whistle is detected. Override in subclass., Optional hook called once when the agent starts.

### Community 28 - "RefereeAgent"
Cohesion: 0.21
Nodes (6): NBA Rule: Defensive player cannot remain in the paint for > 3 seconds         w, FIBA Rule: 2 unsportsmanlike fouls OR 1 technical + 1 unsportsmanlike         =, Scan new events from other agents for referee-relevant triggers., Whistle detected → log referee whistle event., RefereeAgent, test_pass_and_possession_telemetry()

### Community 29 - "Point3D"
Cohesion: 0.22
Nodes (6): Point3D, A point in 3D world space. Y is up, X is right, Z is forward (into the scene)., Return (x, z) ground-plane position., A 3D point on the court should project to a valid pixel., TestPoint3D, TestProjection

### Community 30 - "PoseAnalysisAgent"
Cohesion: 0.26
Nodes (4): calculate_angle(), PoseAnalysisAgent, Pose Analysis Agent supporting both MediaPipe Pose (lightweight) and DensePose /, Calculate the 2D angle between three landmarks: a, b, c.     Returns the angle i

### Community 31 - "ground_image"
Cohesion: 0.25
Nodes (8): Request, extract_frame(), _get_grounder(), get_highlights(), ground_image(), _image_size(), Scan GAME_STATE.events for SHOT_MAKE or VIOLATION_ALERT events,     slice the s, _resolve_video_source()

### Community 32 - ".render_frame_3d"
Cohesion: 0.21
Nodes (6): Project a 3D point to pixel. Returns None if behind camera., Draw a 3D sphere as a projected circle with shading., Draw a 3D cylinder as a projected vertical shape with head circle., Draw a 3D ring as a projected ellipse., Draw court floor as a 3D projected grid., Render a 3D scene from a virtual camera angle.

### Community 33 - "WhistleDetector"
Cohesion: 0.20
Nodes (6): ndarray, Return the latest whistle-band confidence ratio (0.0 to ~10.0)., Analyses audio chunks for high-frequency whistle peaks.          Usage:, Args:             sample_rate: Audio sample rate in Hz.             min_freq /, Analyse an audio chunk and return True if a whistle is detected., WhistleDetector

### Community 34 - "bbox_to_player"
Cohesion: 0.29
Nodes (4): Detects players (class 0 in COCO) and returns PlayerCapsule primitives., bbox_to_player(), Convert a YOLO bounding box [x1, y1, x2, y2] to a PlayerCapsule.      Head = t, TestBboxToPlayer

### Community 35 - "Cylinder3D"
Cohesion: 0.27
Nodes (5): Cylinder3D, Player in 3D space — a vertical cylinder from base up to height., Center of the cylinder., TestCylinder3D, TestLiftPlayer

### Community 36 - "base_agent.py"
Cohesion: 0.22
Nodes (5): ABC, Abstract base class for all Kampfgericht agents. Each agent runs in its own thr, Referee Agent — Schiedsrichter-Modus. Watches for rule violations and flags war, Referee Signals Agent.  Uses Google MediaPipe PoseLandmarker (Tasks API) to tr, Whistle Detector — Schiedsrichter-Pfiff-Erkennung.  Detects referee whistles i

### Community 37 - "Backboard"
Cohesion: 0.22
Nodes (4): Backboard, bbox_to_backboard(), Convert a bounding rectangle to a Backboard., Backboard reduced to a 2-D rectangle.

### Community 38 - "dependencies"
Cohesion: 0.22
Nodes (8): dependencies, axios, framer-motion, lucide-react, devDependencies, autoprefixer, postcss, tailwindcss

### Community 41 - "TestHoopDetector"
Cohesion: 0.29
Nodes (3): Crop should return a non-empty ndarray., Should detect at least one person in a real game frame., TestHoopDetector

### Community 42 - "bbox_to_ball"
Cohesion: 0.33
Nodes (3): Detects sports balls (class 32 in COCO) and returns Ball primitives., bbox_to_ball(), Convert a YOLO bounding box [x1, y1, x2, y2] to a Ball primitive.

### Community 45 - "ball_near_player"
Cohesion: 0.50
Nodes (3): ball_near_player(), True if the ball center is within `threshold` pixels of the player center., TestBallNearPlayer

## Knowledge Gaps
- **40 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PrimitiveRenderer` connect `PrimitiveRenderer` to `.render_frame_3d`, `ShotTracker`, `Cylinder3D`, `Backboard`, `RimDetector`, `server.py`, `PlayerCapsule`, `CourtGeometry`, `Camera3D`, `Point2D`, `HoopDetector`, `Path`, `Sphere3D`, `Ball`, `ManualPlayer`, `_segment_worker`, `primitive_renderer.py`, `Point3D`?**
  _High betweenness centrality (0.122) - this node is a cross-community bridge._
- **Why does `Point2D` connect `Point2D` to `PrimitiveRenderer`, `ShotTracker`, `RimDetector`, `PlayerCapsule`, `CourtGeometry`, `Camera3D`, `__init__.py`, `HoopDetector`, `Sphere3D`, `primitives.py`, `Ball`, `BasketballAnalyzer`, `_segment_worker`, `primitive_renderer.py`, `Point3D`, `bbox_to_player`, `Cylinder3D`, `Backboard`, `ScoresheetAgent`, `bbox_to_ball`, `ball_near_player`, `TestCamera3DCalibration`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `BasketballAnalyzer` connect `BasketballAnalyzer` to `ShotTracker`, `server.py`, `MultiRoiTracker`, `PlayerCapsule`, `CourtGeometry`, `__init__.py`, `Point2D`, `HoopDetector`, `Path`, `primitives.py`, `Ball`, `ManualPlayer`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 40 inferred relationships involving `Point2D` (e.g. with `BasketballAnalyzer` and `Camera3D`) actually correct?**
  _`Point2D` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `Ball` (e.g. with `BasketballAnalyzer` and `Camera3D`) actually correct?**
  _`Ball` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `PlayerCapsule` (e.g. with `BasketballAnalyzer` and `Camera3D`) actually correct?**
  _`PlayerCapsule` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 33 inferred relationships involving `Rim` (e.g. with `Camera3D` and `HoopDetector`) actually correct?**
  _`Rim` has 33 INFERRED edges - model-reasoned connections that need verification._