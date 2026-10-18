"""
run_game_evaluation.py
======================
Runs the HoopStats video analytics pipeline on the official match video
and generates a comparative evaluation against the ground truth DBB scoresheet.
"""

import sys
import os
import json
import cv2
import argparse
from hoop_stats.court_calibrator import AutomatedCourtCalibrator
from hoop_stats.detector import HoopDetector
from hoop_stats.ocr import JerseyOCR
from hoop_stats.shot_tracker import ShotTracker
from hoop_stats.team_colors import get_dominant_jersey_color, assign_teams_by_color
from hoop_stats.metrics import calculate_shooting_splits, calculate_team_ratings
from hoop_stats.benchmark_evaluator import BenchmarkEvaluator, OFFICIAL_GROUND_TRUTH_UFFENHEIM


def run_evaluation(video_path: str, max_frames: int = 6000, skip_frames: int = 4):
    print(f"\n=================================================================")
    print(f"🏀 HOOP-STATS MATCH ANALYSIS & BENCHMARK EVALUATOR")
    print(f"Video: {video_path}")
    print(f"Max Frames: {max_frames} | Skip Frames: {skip_frames}")
    print(f"Ground Truth Team: SB Uffenheim (55 Pkt) vs. SG Oerlenbach (49 Pkt)")
    print(f"=================================================================\n")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Video Loaded: {fw}x{fh} @ {fps:.1f} FPS | Total: {total_frames} frames ({total_frames/(fps*60):.1f} min)")

    # 1. Automated Court Calibration on initial frame
    calibrator = AutomatedCourtCalibrator()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ret, sample_frame = cap.read()
    if ret:
        calib_res = calibrator.auto_calibrate(sample_frame)
        print(f"[Auto-Calibrator] Homography status: {calib_res.success} (Method: {calib_res.method}, Confidence: {calib_res.confidence:.2f})")
    else:
        print("[Auto-Calibrator] Warning: Could not read sample frame for calibration.")

    detector = HoopDetector()
    ocr = JerseyOCR()
    shot_tracker = ShotTracker(fw, fh)

    player_jerseys = {}
    player_colors = {}
    detected_shots = []

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("\nStarting video inference stream...")
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        # Player Detection & Tracking
        results = detector.model.track(frame, persist=True, classes=[0], verbose=False, conf=0.35)
        player_capsules = {}

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, obj_id in zip(boxes, ids):
                # Jersey OCR every 60 frames for active players
                if obj_id not in player_jerseys and (frame_idx % 60 == 0):
                    crop = detector.get_jersey_crop(frame, box)
                    num = ocr.identify_number(crop)
                    if num:
                        player_jerseys[obj_id] = num

                # Jersey Color every 60 frames
                if obj_id not in player_colors and (frame_idx % 60 == 0):
                    col = get_dominant_jersey_color(frame, box)
                    if col not in ("Unknown", "Mixed"):
                        player_colors[obj_id] = col

        # Ball Detection
        balls = detector.detect_ball(frame)
        rims = detector.detect_rims(frame)
        if rims:
            shot_tracker.update_rims(rims)

        shot_tracker.update(balls, player_capsules, frame_idx)

        frame_idx += 1
        if frame_idx % 600 == 0:
            print(f"  Processed {frame_idx}/{max_frames} frames ({(frame_idx/max_frames)*100:.1f}%) | Unique Players: {len(player_jerseys)}")

    cap.release()

    # Calculate detected stats
    detected_jerseys = list(player_jerseys.values())
    print("\n=== DETECTION RESULTS ===")
    print(f"Total Unique Jersey Numbers Detected: {sorted(list(set(detected_jerseys)))}")
    print(f"Completed Shots Tracked: {len(shot_tracker.completed_shots)}")

    evaluator = BenchmarkEvaluator()
    player_rec = evaluator.evaluate_player_recognition(detected_jerseys)
    print(f"\nPlayer Recognition Recall: {player_rec['recall_pct']}% (Matched: {player_rec['matched_jerseys']})")

    # Generate full comparison report against official scoresheet
    report = {
        "video": video_path,
        "processed_frames": frame_idx,
        "player_recognition": player_rec,
        "ground_truth": OFFICIAL_GROUND_TRUTH_UFFENHEIM,
    }

    os.makedirs("results", exist_ok=True)
    report_path = "results/match_oerlenbach_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\nDetailed evaluation report saved to: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="uploads/game_oerlenbach.mp4")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--skip", type=int, default=5)
    args = parser.parse_args()

    run_evaluation(args.video, max_frames=args.frames, skip_frames=args.skip)
