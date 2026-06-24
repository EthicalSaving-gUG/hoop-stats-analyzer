import cv2
import numpy as np
import multiprocessing as mp
from .detector import HoopDetector
from .geometry import CourtGeometry
from .metrics import PlayerMetrics
from .ocr import JerseyOCR
from .exporter import DataExporter
from .team_colors import get_dominant_jersey_color, assign_teams_by_color
from .shot_tracker import ShotTracker
from .roi_tracker import MultiRoiTracker
from .video_utils import get_video_info, get_segment_ranges, read_segment
from .primitives import Point2D, bbox_to_ball, bbox_to_player

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
        self.shot_tracker = None   # initialised after first frame

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, court_calibration_points=None, max_frames=None, skip_frames=3, progress_callback=None):
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

        self.shot_tracker = ShotTracker(fw, fh)

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

            results = self.detector.model.track(
                frame, persist=True, classes=[0], verbose=False, conf=0.35
            )

            player_capsules = {}
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy().astype(int)

                for box, obj_id in zip(boxes, ids):
                    # ── Convert to PlayerCapsule primitive ─────────────────
                    capsule = bbox_to_player(box[0], box[1], box[2], box[3], obj_id=obj_id)

                    # ── Real-world position & physics ─────────────────────
                    foot = capsule.ground_contact
                    mx, my = self.geometry.pixel_to_meter(foot.x, foot.y)
                    capsule.meter_pos = Point2D(mx, my)
                    self.metrics.update_position(obj_id, capsule.meter_pos, frame_idx)

                    player_capsules[obj_id] = capsule

                    # ── Jersey number (OCR on torso crop) ─────────────────
                    if obj_id not in self.player_metadata:
                        self.player_metadata[obj_id] = {"jersey": "Unknown", "team": "TBD"}

                    if self.player_metadata[obj_id].get("jersey", "Unknown") == "Unknown":
                        crop   = self.detector.get_jersey_crop(frame, capsule)
                        number = self.ocr.identify_number(crop)
                        if number:
                            self.player_metadata[obj_id]["jersey"] = number

                    # ── Jersey color (sampled every 30 frames) ────────────
                    if frame_idx % 30 == 0:
                        color = get_dominant_jersey_color(frame, box)
                        if color not in ("Unknown", "Mixed"):
                            self.player_colors[obj_id] = color

            # ── Ball / scoring detection ──────────────────────────────────
            balls = self.detector.detect_ball(frame)

            self.shot_tracker.update(balls, player_capsules, frame_idx)

            frame_idx += 1
            if frame_idx % 150 == 0:
                pct = (frame_idx / total * 100) if total else 0
                msg = f"  Frame {frame_idx}/{total}  ({pct:.0f}%)  Players tracked: {len(self.player_metadata)}"
                print(msg)
                if progress_callback:
                    progress_callback(min(99, int(pct)), msg)

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
        scores = self.shot_tracker.get_total_scores() if self.shot_tracker else {}
        shots_log = self.shot_tracker.completed_shots if self.shot_tracker else []

        output = {
            "players": final_data,
            "scoring": scores,
            "team_summary": self._build_team_summary(final_data, shots_log),
            "shots": shots_log
        }

        self.exporter.export_json(output)
        self.exporter.export_csv_full(output)
        print("\n=== Analysis Complete ===")
        print(f"Players tracked : {len(final_data)}")
        print(f"Scoring summary : {scores}")
        print("Files saved.\n")

    def _build_team_summary(self, final_data, shots_log):
        summary = {}
        for pid, d in final_data.items():
            team = d["team"]
            if team not in summary:
                summary[team] = {
                    "player_count":   0,
                    "avg_max_speed":  [],
                    "avg_max_accel":  [],
                    "FGA": 0,
                    "FGM": 0,
                    "3PA": 0,
                    "3PM": 0
                }
            summary[team]["player_count"] += 1
            summary[team]["avg_max_speed"].append(d["max_speed_ms"])
            summary[team]["avg_max_accel"].append(d["max_accel_ms2"])

        for shot in shots_log:
            pid = str(shot["shooter_id"])
            if pid in final_data:
                team = final_data[pid]["team"]
                if team in summary:
                    summary[team]["FGA"] += 1
                    if shot["status"] == "made":
                        summary[team]["FGM"] += 1
                    
                    if shot["zone"] in ("three_point", "3PT Wing"):
                        summary[team]["3PA"] += 1
                        if shot["status"] == "made":
                            summary[team]["3PM"] += 1

        for team, s in summary.items():
            speeds = s["avg_max_speed"]
            accels = s["avg_max_accel"]
            s["avg_max_speed"] = float(round(sum(speeds)/len(speeds), 2)) if speeds else 0.0
            s["avg_max_accel"] = float(round(sum(accels)/len(accels), 2)) if accels else 0.0
            
            fga = s["FGA"]
            fgm = s["FGM"]
            tpm = s["3PM"]
            s["FG%"] = float(round((fgm / fga) * 100, 1)) if fga > 0 else 0.0
            s["eFG%"] = float(round(((fgm + 0.5 * tpm) / fga) * 100, 1)) if fga > 0 else 0.0

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    def run_fast(
        self,
        court_calibration_points=None,
        max_frames=None,
        skip_frames=3,
        n_workers=None,
        yolo_interval=30,
        progress_callback=None
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

        self.shot_tracker = ShotTracker(fw, fh)

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
        partial_results = []
        with ctx.Pool(processes=n_workers) as pool:
            completed = 0
            for partial in pool.imap_unordered(_segment_worker, worker_args):
                partial_results.append(partial)
                completed += 1
                if progress_callback:
                    pct = int((completed / len(worker_args)) * 95) # Leave 5% for finalize
                    progress_callback(pct, f"Processed segment {completed}/{len(worker_args)}")

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
                self.shot_tracker.scores[zi] += pts
            
            # Merge completed shots
            if "shots" in partial:
                self.shot_tracker.completed_shots.extend(partial["shots"])

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
    from .shot_tracker import ShotTracker
    from .roi_tracker import MultiRoiTracker
    from .video_utils import get_video_info, read_segment
    from .primitives import Point2D, bbox_to_ball, bbox_to_player

    info     = get_video_info(video_path)
    fps      = info["fps"]
    fw, fh   = info["width"], info["height"]

    detector   = HoopDetector()
    geometry   = CourtGeometry()
    metrics    = PlayerMetrics(fps=fps)
    ocr        = JerseyOCR()
    tracker    = MultiRoiTracker()
    shot_tracker = ShotTracker(fw, fh)

    player_metadata = {}
    player_colors   = {}

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

            # Detect Ball — use raw bboxes for ROI tracker init
            ball_boxes_raw = detector.detect_ball_raw(frame)
            for i, b_box in enumerate(ball_boxes_raw):
                # Use negative IDs for balls
                tracker.add_tracker(frame, b_box, -1 - i)
                
            updated_boxes = tracker.update(frame) # init first frame outputs
            
        else:
            # ── Track using CSRT (Fast) ───────────────────────────────────
            updated_boxes = tracker.update(frame)

        # ── Process Tracked Boxes → Primitives ────────────────────────────
        balls = []
        player_capsules = {}
        for obj_id, box in updated_boxes.items():
            if obj_id < 0:
                # It's a ball → convert to Ball primitive
                balls.append(bbox_to_ball(box[0], box[1], box[2], box[3]))
            else:
                # It's a player → convert to PlayerCapsule primitive
                capsule = bbox_to_player(box[0], box[1], box[2], box[3], obj_id=obj_id)
                foot = capsule.ground_contact
                mx, my = geometry.pixel_to_meter(foot.x, foot.y)
                capsule.meter_pos = Point2D(mx, my)
                metrics.update_position(obj_id, capsule.meter_pos, frame_idx)
                player_capsules[obj_id] = capsule

        # ── Scoring ───────────────────────────────────────────────────────
        shot_tracker.update(balls, player_capsules, frame_idx)

    return {
        "player_metadata": player_metadata,
        "player_colors":   player_colors,
        "trajectories":    metrics.trajectories,
        "metrics":         metrics.metrics,
        "scores":          shot_tracker.scores,
        "shots":           shot_tracker.completed_shots,
    }
