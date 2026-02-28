import cv2
import numpy as np
import multiprocessing as mp
from .detector import HoopDetector
from .geometry import CourtGeometry
from .metrics import PlayerMetrics
from .ocr import JerseyOCR
from .exporter import DataExporter
from .team_colors import get_dominant_jersey_color, assign_teams_by_color
from .scoring import ScoringDetector
from .roi_tracker import MultiRoiTracker
from .video_utils import get_video_info, get_segment_ranges, read_segment

class BasketballAnalyzer:
    def __init__(self, video_path, output_name="game_stats"):
        self.video_path = video_path
        self.detector = HoopDetector()
        self.geometry = CourtGeometry()
        self.metrics = PlayerMetrics()
        self.ocr = JerseyOCR()
        self.exporter = DataExporter(output_name)

        self.player_metadata = {}      # {id: {"jersey": "...", "team": "..."}}
        self.player_colors   = {}      # {id: color_name} – raw dominant color
        self.scoring_detector = None   # initialised after first frame

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, court_calibration_points=None, max_frames=None, skip_frames=3):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {self.video_path}")
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.metrics.fps = fps
        self.metrics.dt  = 1.0 / fps

        if court_calibration_points:
            self.geometry.set_homography(
                court_calibration_points['video'],
                court_calibration_points['real']
            )

        self.scoring_detector = ScoringDetector(fw, fh)

        print(f"Video: {fw}x{fh} @ {fps:.1f}fps  |  {total} frames total")
        frame_idx = 0

        while cap.isOpened():
            if max_frames and frame_idx >= max_frames:
                print(f"Reached max_frames limit ({max_frames}). Stopping.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to speed up analysis if requested
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # We use a placeholder logic here. For actual ROI tracking,
            # this single-threaded `run()` method is mostly a fallback.
            # Real work happens in `_segment_worker()`.
            # To keep `run()` functional, we just use YOLO tracking every time.
            results = self.detector.model.track(
                frame, persist=True, classes=[0], verbose=False, conf=0.35
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy().astype(int)

                for box, obj_id in zip(boxes, ids):
                    # ── 2. Real-world position & physics ──────────────────
                    px = (box[0] + box[2]) / 2
                    py = box[3]            # ground contact point
                    mx, my = self.geometry.pixel_to_meter(px, py)
                    self.metrics.update_position(obj_id, mx, my, frame_idx)

                    # ── 3. Jersey number (OCR on torso crop) ──────────────
                    # Ensure defaults always present
                    if obj_id not in self.player_metadata:
                        self.player_metadata[obj_id] = {"jersey": "Unknown", "team": "TBD"}

                    if self.player_metadata[obj_id].get("jersey", "Unknown") == "Unknown":
                        crop   = self.detector.get_jersey_crop(frame, box)
                        number = self.ocr.identify_number(crop)
                        if number:
                            self.player_metadata[obj_id]["jersey"] = number

                    # ── 4. Jersey color (sampled every 30 frames) ─────────
                    if frame_idx % 30 == 0:
                        color = get_dominant_jersey_color(frame, box)
                        if color not in ("Unknown", "Mixed"):
                            self.player_colors[obj_id] = color

            # ── 5. Ball / scoring detection ───────────────────────────────
            ball_boxes = self.detector.detect_ball(frame)
            ball_centers = []
            for b_box in ball_boxes:
                bx = (b_box[0] + b_box[2]) / 2
                by = (b_box[1] + b_box[3]) / 2
                ball_centers.append((bx, by))
            
            self.scoring_detector.update(ball_centers, frame_idx)

            frame_idx += 1
            if frame_idx % 150 == 0:
                pct = (frame_idx / total * 100) if total else 0
                print(f"  Frame {frame_idx}/{total}  ({pct:.0f}%)  "
                      f"Players tracked: {len(self.player_metadata)}")

        cap.release()
        self._finalize_stats()

    # ─────────────────────────────────────────────────────────────────────────
    def _finalize_stats(self):
        # Assign teams based on dominant jersey colors
        team_assignments = assign_teams_by_color(self.player_colors)

        final_data = {}
        for pid, stats in self.metrics.metrics.items():
            meta   = self.player_metadata.get(pid, {})
            jersey = meta.get("jersey", "Unknown")
            team   = team_assignments.get(pid, "Unknown")
            color  = self.player_colors.get(pid, "Unknown")

            # Collect trajectory as simplified line segments (every 5th point)
            traj = self.metrics.get_trajectory_lines(pid)
            simplified = traj[::5]  # reduce resolution

            final_data[str(pid)] = {
                "jersey":         jersey,
                "team":           team,
                "jersey_color":   color,
                "max_speed_ms":   float(round(stats["max_speed"], 2)),
                "max_accel_ms2":  float(round(stats["max_accel"], 2)),
                "trajectory_pts": len(traj),
                "trail_sample":   simplified[:20]  # first 20 sampled points
            }

        # Add scoring summary
        scores = self.scoring_detector.get_total_scores() if self.scoring_detector else {}

        output = {
            "players": final_data,
            "scoring": scores,
            "team_summary": self._build_team_summary(final_data)
        }

        self.exporter.export_json(output)
        self.exporter.export_csv_players(final_data, scores)
        print("\n=== Analysis Complete ===")
        print(f"Players tracked : {len(final_data)}")
        print(f"Scoring summary : {scores}")
        print("Files saved.\n")

    def _build_team_summary(self, final_data):
        summary = {}
        for pid, d in final_data.items():
            team = d["team"]
            if team not in summary:
                summary[team] = {
                    "player_count":   0,
                    "avg_max_speed":  [],
                    "avg_max_accel":  [],
                }
            summary[team]["player_count"] += 1
            summary[team]["avg_max_speed"].append(d["max_speed_ms"])
            summary[team]["avg_max_accel"].append(d["max_accel_ms2"])

        for team, s in summary.items():
            speeds = s["avg_max_speed"]
            accels = s["avg_max_accel"]
            s["avg_max_speed"] = float(round(sum(speeds)/len(speeds), 2)) if speeds else 0.0
            s["avg_max_accel"] = float(round(sum(accels)/len(accels), 2)) if accels else 0.0
        return summary

    # ─────────────────────────────────────────────────────────────────────────
    def run_fast(
        self,
        court_calibration_points=None,
        max_frames=None,
        skip_frames=3,
        n_workers=None,
        yolo_interval=30
    ):
        """
        Accelerated pipeline that splits the video into segments processed in 
        parallel worker processes using Region-Of-Interest (ROI) optical tracking.

        Parameters
        ----------
        n_workers : int, optional
            Number of parallel processes. Defaults to min(4, cpu_count()).
        yolo_interval : int
            Run heavy YOLO detection every N frames to refresh the optical tracker.
        """
        info = get_video_info(self.video_path)
        fps = info["fps"]
        fw  = info["width"]
        fh  = info["height"]
        total = info["total_frames"]

        self.metrics.fps = fps
        self.metrics.dt  = 1.0 / fps

        if court_calibration_points:
            self.geometry.set_homography(
                court_calibration_points["video"],
                court_calibration_points["real"],
            )

        self.scoring_detector = ScoringDetector(fw, fh)

        if n_workers is None:
            n_workers = min(4, mp.cpu_count())

        segment_ranges = get_segment_ranges(
            self.video_path, n_segments=n_workers, max_frames=max_frames
        )

        print(f"[run_fast] {fw}x{fh} @ {fps:.1f}fps | {total} total frames")
        print(f"[run_fast] {len(segment_ranges)} segments, skip_frames={skip_frames}")

        worker_args = [
            (
                self.video_path,
                start,
                end,
                skip_frames,
                yolo_interval
            )
            for (start, end) in segment_ranges
        ]

        # ── Run segments in parallel processes ─────────────────────────────
        ctx = mp.get_context("spawn")  # spawn is safe on Windows
        with ctx.Pool(processes=n_workers) as pool:
            partial_results = pool.map(_segment_worker, worker_args)

        # ── Merge partial results into main state ───────────────────────────
        for partial in partial_results:
            # Merge player metadata (jersey, colors)
            for pid, meta in partial["player_metadata"].items():
                if pid not in self.player_metadata:
                    self.player_metadata[pid] = meta
                else:
                    # Only promote if we got a real jersey number
                    if (
                        self.player_metadata[pid].get("jersey", "Unknown") == "Unknown"
                        and meta.get("jersey", "Unknown") != "Unknown"
                    ):
                        self.player_metadata[pid]["jersey"] = meta["jersey"]

            for pid, color in partial["player_colors"].items():
                if pid not in self.player_colors:
                    self.player_colors[pid] = color

            # Merge trajectory / metrics
            for pid, traj in partial["trajectories"].items():
                if pid not in self.metrics.trajectories:
                    self.metrics.trajectories[pid] = traj
                    self.metrics.metrics[pid] = partial["metrics"][pid]
                else:
                    self.metrics.trajectories[pid].extend(traj)
                    # Keep the highest observed speed / accel
                    cur  = self.metrics.metrics[pid]
                    other = partial["metrics"][pid]
                    cur["max_speed"] = max(cur["max_speed"], other["max_speed"])
                    cur["max_accel"] = max(cur["max_accel"], other["max_accel"])

            # Merge scoring (sum per hoop zone)
            for zi, pts in enumerate(partial["scores"]):
                self.scoring_detector.scores[zi] += pts

        self._finalize_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker – must be picklable (top-level function, not a method)
# ─────────────────────────────────────────────────────────────────────────────
def _segment_worker(args):
    """
    Processes one video segment using Detect-Validate-Track paradigm.
    Runs YOLO occasionally, then optical ROI tracking in-between.
    """
    video_path, start_frame, end_frame, skip_frames, yolo_interval = args

    from .detector import HoopDetector
    from .geometry import CourtGeometry
    from .metrics import PlayerMetrics
    from .ocr import JerseyOCR
    from .team_colors import get_dominant_jersey_color
    from .scoring import ScoringDetector
    from .roi_tracker import MultiRoiTracker
    from .video_utils import get_video_info, read_segment

    info     = get_video_info(video_path)
    fps      = info["fps"]
    fw, fh   = info["width"], info["height"]

    detector   = HoopDetector()
    geometry   = CourtGeometry()
    metrics    = PlayerMetrics(fps=fps)
    ocr        = JerseyOCR()
    tracker    = MultiRoiTracker()
    scoring    = ScoringDetector(fw, fh)

    player_metadata = {}
    player_colors   = {}
    
    # Track ball separately (obj_id = -1, -2, etc. or just keep list of ball centers)
    # Actually, let's keep it simple: Optical Track ball as obj_id -1.

    for frame_idx, frame in read_segment(video_path, start_frame, end_frame, skip_frames):
        
        # ── Detect using YOLO (periodically) ──────────────────────────────
        if frame_idx % yolo_interval == 0 or tracker.active_count == 0:
            tracker.clear()
            
            # Detect Players
            det_results = detector.model.track(
                frame, persist=True, classes=[0], verbose=False, conf=0.35, tracker="botsort.yaml"
            )
            
            if det_results[0].boxes.id is not None:
                boxes = det_results[0].boxes.xyxy.cpu().numpy()
                ids   = det_results[0].boxes.id.cpu().numpy().astype(int)

                for box, obj_id in zip(boxes, ids):
                    tracker.add_tracker(frame, box, obj_id)
                    
                    if obj_id not in player_metadata:
                        player_metadata[obj_id] = {"jersey": "Unknown", "team": "TBD"}

                    if player_metadata[obj_id].get("jersey", "Unknown") == "Unknown":
                        crop   = detector.get_jersey_crop(frame, box)
                        number = ocr.identify_number(crop)
                        if number:
                            player_metadata[obj_id]["jersey"] = number

                    # Sample color during YOLO pulses
                    color = get_dominant_jersey_color(frame, box)
                    if color not in ("Unknown", "Mixed"):
                        player_colors[obj_id] = color

            # Detect Ball
            ball_boxes = detector.detect_ball(frame)
            for i, b_box in enumerate(ball_boxes):
                # Use negative IDs for balls
                tracker.add_tracker(frame, b_box, -1 - i)
                
            updated_boxes = tracker.update(frame) # init first frame outputs
            
        else:
            # ── Track using CSRT (Fast) ───────────────────────────────────
            updated_boxes = tracker.update(frame)

        # ── Process Tracked Boxes ─────────────────────────────────────────
        ball_centers = []
        for obj_id, box in updated_boxes.items():
            if obj_id < 0:
                # It's a ball
                bx = (box[0] + box[2]) / 2
                by = (box[1] + box[3]) / 2
                ball_centers.append((bx, by))
            else:
                # It's a player
                px = (box[0] + box[2]) / 2
                py = box[3]
                mx, my = geometry.pixel_to_meter(px, py)
                metrics.update_position(obj_id, mx, my, frame_idx)

        # ── Scoring ───────────────────────────────────────────────────────
        scoring.update(ball_centers, frame_idx)

    return {
        "player_metadata": player_metadata,
        "player_colors":   player_colors,
        "trajectories":    metrics.trajectories,
        "metrics":         metrics.metrics,
        "scores":          scoring.scores,
    }
