"""
run_game_evaluation.py
======================
Runs the HoopStats video analytics pipeline on the official match video
and generates a comparative evaluation against the ground truth DBB scoresheet.
Supports single-segment and multi-quarter whole-match evaluations with PlayerRegistry.
"""

import sys
import os
import json
import math
import cv2
import argparse
import numpy as np
from hoop_stats.court_calibrator import AutomatedCourtCalibrator
from hoop_stats.detector import HoopDetector
from hoop_stats.ocr import JerseyOCR
from hoop_stats.shot_tracker import ShotTracker
from hoop_stats.team_colors import get_dominant_jersey_color, assign_teams_by_color
from hoop_stats.metrics import calculate_shooting_splits, calculate_team_ratings
from hoop_stats.benchmark_evaluator import BenchmarkEvaluator, OFFICIAL_GROUND_TRUTH_UFFENHEIM
from hoop_stats.primitives import bbox_to_player, Point2D
from hoop_stats.geometry import CourtGeometry
from hoop_stats.player_registry import PlayerRegistry
from hoop_stats.event_detector import EventDetector
from hoop_stats.video_clipper import StatClipRegistry
from hoop_stats.coach_report import CoachReportGenerator


def run_evaluation(video_path: str, start_frame: int = 13000, max_frames: int = 4000, skip_frames: int = 3,
                   use_sahi: bool = False, multi_quarter: bool = False, full_game: bool = False,
                   frames_per_quarter: int = 1500):
    mode_str = "Komplettes Spiel (Vollständige 4 Viertel Q1-Q4)" if full_game else (
        f"Multi-Quarter Scan ({frames_per_quarter} Frames / Viertel)" if multi_quarter else f"Fokussiertes Segment ({start_frame} bis {start_frame + max_frames})"
    )
    print(f"\n=================================================================")
    print(f"🏀 HOOP-STATS MATCH ANALYSIS & BENCHMARK EVALUATOR")
    print(f"Video: {video_path}")
    print(f"Modus: {mode_str}")
    print(f"Skip Frames: {skip_frames} | Korb-Fokus: Aktiv | Free-Throw Formation: Aktiv")
    print(f"PlayerRegistry: Dynamic Late Identity Binding & Roster Constraint Aktiv")
    print(f"EventDetector: Schrittfehler (Traveling) & Fehlpass/Steal Detection Aktiv")
    print(f"Ground Truth Team: SB Uffenheim (55 Pkt) vs. SG Oerlenbach (49 Pkt)")
    print(f"=================================================================\n")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Video Loaded: {fw}x{fh} @ {fps:.1f} FPS | Total Frames: {total_frames} ({total_frames/(fps*60):.1f} min)")

    # 1. Automated Court Calibration on initial frame
    calibrator = AutomatedCourtCalibrator()
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(100, start_frame))
    ret, sample_frame = cap.read()
    court_geo = CourtGeometry()
    if ret:
        calib_res = calibrator.auto_calibrate(sample_frame)
        print(f"[Auto-Calibrator] Homography status: {calib_res.success} (Method: {calib_res.method}, Confidence: {calib_res.confidence:.2f})")
        if calib_res.success and calib_res.homography is not None:
            court_geo.homography_matrix = calib_res.homography
    else:
        print("[Auto-Calibrator] Warning: Could not read sample frame for calibration.")

    detector = HoopDetector()
    ocr = JerseyOCR()

    # Initialize PlayerRegistry with official Uffenheim roster from scoresheet
    roster_uffenheim = list(OFFICIAL_GROUND_TRUTH_UFFENHEIM.get("players", {}).keys())
    player_registry = PlayerRegistry(
        roster_home=roster_uffenheim,
        voting_threshold=0.85,
        single_hit_conf=0.65
    )

    shot_tracker = ShotTracker(fw, fh, court_geometry=court_geo, fps=fps, player_registry=player_registry, halftime_frame=999999999)
    event_detector = EventDetector(fw, fh, court_geometry=court_geo, player_registry=player_registry, fps=fps)
    clip_registry = StatClipRegistry(fps=fps, pre_pad_s=2.5, post_pad_s=2.0)
    coach_reporter = CoachReportGenerator(team_name="SB Uffenheim")

    player_jerseys = {}
    player_colors = {}
    player_teams = {}
    
    # Event Counters
    tracked_events = {
        "points": 0,
        "fgm": 0,
        "fga": 0,
        "fg2m": 0,
        "fg2a": 0,
        "fg3m": 0,
        "fg3a": 0,
        "ftm": 0,
        "fta": 0,
        "reb": 0,
        "oreb": 0,
        "dreb": 0,
        "assists": 0,
        "turnovers": 0,
        "turnovers_traveling": 0,
        "turnovers_bad_pass": 0,
        "turnovers_three_seconds": 0,
        "turnovers_backcourt": 0,
        "turnovers_out_of_bounds": 0,
        "turnovers_double_dribble": 0,
        "turnovers_eight_seconds": 0,
        "turnovers_five_seconds": 0,
        "steals": 0,
        "blocks": 0,
        "fouls_drawn": 0,
    }

    # Define segments to process
    if full_game:
        segments = [
            ("1. Viertel (Q1)", 13500, 30500),
            ("2. Viertel (Q2)", 33500, 48500),
            ("Halbzeit Pause", -1, -1),
            ("3. Viertel (Q3)", 63500, 78500),
            ("4. Viertel (Q4)", 83500, 107500),
        ]
    elif multi_quarter:
        q_len = frames_per_quarter if frames_per_quarter > 0 else 1500
        segments = [
            ("1. Viertel (Q1)", 13500, 13500 + q_len),
            ("2. Viertel (Q2)", 33500, 33500 + q_len),
            ("Halbzeit Pause", -1, -1),
            ("3. Viertel (Q3)", 63500, 63500 + q_len),
            ("4. Viertel (Q4)", 83500, 83500 + q_len),
        ]
    else:
        segments = [("Haupt-Spielphase", start_frame, min(total_frames, start_frame + max_frames))]

    total_inferences = 0
    last_possession_team = None
    prev_ft_mode = False
    last_foul_frame = -99999

    for seg_name, s_start, s_end in segments:
        if seg_name == "Halbzeit Pause":
            shot_tracker.swap_sides()
            continue

        print(f"\n--- Verarbeite {seg_name} (Frames {s_start} bis {s_end}) ---")
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_start)
        frame_idx = s_start
        shot_tracker.ball_history.clear()
        shot_tracker.ball_frame_history.clear()
        shot_tracker.active_shots.clear()
        shot_tracker.player_history.clear()

        while cap.isOpened() and frame_idx < s_end:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - s_start) % skip_frames != 0:
                frame_idx += 1
                continue

            total_inferences += 1

            # 1. Player Detection & ByteTrack (fast, no GMC warnings)
            results = detector.model.track(frame, persist=True, classes=[0], verbose=False, tracker="bytetrack.yaml", conf=0.35)
            player_capsules = {}

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.array([1000 + i for i in range(len(boxes))])

                ocr_candidates = []
                for box, obj_id in zip(boxes, ids):
                    capsule = bbox_to_player(box[0], box[1], box[2], box[3], obj_id=obj_id)
                    mx, my = court_geo.pixel_to_meter(capsule.ground_contact.x, capsule.ground_contact.y)
                    capsule.meter_pos = Point2D(mx, my)
                    player_capsules[obj_id] = capsule

                    # Team Color every 25 inferences
                    if obj_id not in player_colors and (total_inferences % 25 == 0):
                        col = get_dominant_jersey_color(frame, box)
                        if col not in ("Unknown", "Mixed"):
                            player_colors[obj_id] = col
                            if col in ("Black", "Dark"):
                                player_teams[str(obj_id)] = "home"  # Uffenheim
                            elif col in ("White", "Light", "Blue"):
                                player_teams[str(obj_id)] = "away"  # Oerlenbach

                    p_team = player_teams.get(str(obj_id), "home")
                    player_registry.register_track(obj_id, team=p_team)

                    if (total_inferences % 20 == 0) and (obj_id not in player_registry.track_to_jersey):
                        ocr_candidates.append((box, obj_id, p_team))

                # High-speed OCR: run on single clearest candidate (largest box) every 25 inferences
                all_found = len(player_registry.canonical_players) >= len(player_registry.roster_home)
                ocr_freq = 100 if all_found else 25
                if (total_inferences % ocr_freq == 0) and ocr_candidates:
                    best_cand = max(ocr_candidates, key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]))
                    box, obj_id, p_team = best_cand
                    crop = detector.get_jersey_crop(frame, box)
                    num, conf = ocr.identify_number_with_conf(crop)
                    if num and conf > 0.40:
                        bound = player_registry.record_ocr_observation(obj_id, num, confidence=conf, team=p_team)
                        if bound:
                            player_jerseys[obj_id] = bound
                            print(f"  🎯 [PlayerRegistry] Track ID #{obj_id} successfully bound to Jersey #{bound} ({p_team}, Conf: {conf:.2f})!", flush=True)

            # 2. Rim Detection & Korb-Fokus (Periodic check avoids redundant full-frame passes)
            if not shot_tracker._rims_detected or (total_inferences % 250 == 0):
                rims = detector.detect_rims(frame)
                if rims:
                    shot_tracker.update_rims(rims)
            rim_focus_boxes = shot_tracker.get_rim_focus_boxes()

            # 3. Dead-ball Free-throw Formation Detection (edge-triggered with 90s cooldown)
            ft_detected = shot_tracker.detect_free_throw_formation(player_capsules, frame_idx)
            if ft_detected and not prev_ft_mode and (frame_idx - last_foul_frame) > int(fps * 90.0):
                tracked_events["fouls_drawn"] += 1
                last_foul_frame = frame_idx
            prev_ft_mode = ft_detected

            # 4. Ball Detection with Korb-Fokus
            balls = detector.detect_ball(frame, rim_focus_boxes=rim_focus_boxes, use_sahi=use_sahi)

            # 5. Shot & Motion Tracking
            prev_completed = len(shot_tracker.completed_shots)
            events = shot_tracker.update(balls, player_capsules, frame_idx, player_teams=player_teams)

            # 6. Evaluate New Events from ShotTracker
            if len(shot_tracker.completed_shots) > prev_completed:
                latest_shot = shot_tracker.completed_shots[-1]
                status = latest_shot.get("status")
                zone = latest_shot.get("zone", "")
                is_three = "three" in str(zone).lower()
                is_ft = latest_shot.get("is_free_throw", False) and not is_three and "paint" not in str(zone).lower()
                shooter_id = latest_shot.get("shooter_id")
                s_team = player_teams.get(str(shooter_id), "home") if shooter_id is not None else "home"

                if is_three:
                    tracked_events["fg3a"] += 1
                    tracked_events["fga"] += 1
                    if status == "made":
                        tracked_events["fg3m"] += 1
                        tracked_events["fgm"] += 1
                        tracked_events["points"] += 3
                elif is_ft:
                    tracked_events["fta"] += 1
                    if status == "made":
                        tracked_events["ftm"] += 1
                        tracked_events["points"] += 1
                else:
                    tracked_events["fg2a"] += 1
                    tracked_events["fga"] += 1
                    if status == "made":
                        tracked_events["fg2m"] += 1
                        tracked_events["fgm"] += 1
                        tracked_events["points"] += 2

                # Link Shot to Video Clip
                shot_pts = 3 if is_three else (1 if is_ft else 2)
                clip_registry.add_event({
                    "type": "SHOT_MAKE" if status == "made" else "SHOT_MISS",
                    "shooter_id": shooter_id,
                    "team": s_team,
                    "frame": frame_idx,
                    "points": shot_pts if status == "made" else 0,
                    "zone": str(zone),
                    "description": f"{'3PT' if is_three else ('FT' if is_ft else '2PT')} {'Made' if status == 'made' else 'Missed'} ({zone})"
                }, player_registry=player_registry, total_frames=total_frames)

                if status == "made" and not is_ft:
                    # Assist check: Did a teammate pass immediately lead to this made shot (2P or 3P)?
                    s_start_frame = latest_shot.get("start_frame", frame_idx)
                    assist_evt = event_detector.check_assist_on_made_shot(shooter_id, frame_idx, shooter_team=s_team, shot_start_frame=s_start_frame)
                    if assist_evt:
                        p_team = assist_evt.get("team", "home")
                        if p_team == "home":
                            tracked_events["assists"] += 1
                        clip_registry.add_event(assist_evt, player_registry=player_registry, total_frames=total_frames)

                # Rebound Detection on Missed Shot
                if status == "missed":
                    rim_idx = latest_shot.get("target_hoop", 0)
                    rim = shot_tracker.rims[rim_idx]
                    closest_p = None
                    min_p_dist = 9999.0
                    for pid, p_cap in player_capsules.items():
                        d = math.hypot(p_cap.center.x - rim.center.x, p_cap.center.y - rim.center.y)
                        if d < min_p_dist:
                            min_p_dist = d
                            closest_p = pid
                    
                    if closest_p is not None and min_p_dist < (fw * 0.38):
                        tracked_events["reb"] += 1
                        p_team = player_teams.get(str(closest_p), "home")
                        if p_team == "home":
                            tracked_events["dreb"] += 1
                            player_registry.record_action(closest_p, "DEFENSIVE_REBOUND")
                        else:
                            tracked_events["oreb"] += 1
                            player_registry.record_action(closest_p, "OFFENSIVE_REBOUND")

            # 7. Advanced Event & Violation Detection (Schrittfehler, Fehlpässe, Steals, Blocks)
            adv_events = event_detector.update(
                balls, player_capsules, frame_idx,
                player_teams=player_teams,
                active_shots=shot_tracker.active_shots
            )
            for a_evt in adv_events:
                evt_t = a_evt["type"]
                clip_registry.add_event(a_evt, player_registry=player_registry, total_frames=total_frames)
                if evt_t in ("TRAVELING", "FEHLPASS_AUS", "THREE_SECONDS_VIOLATION", "BACKCOURT_VIOLATION", "PLAYER_OUT_OF_BOUNDS", "DOUBLE_DRIBBLE", "EIGHT_SECONDS_VIOLATION", "FIVE_SECONDS_VIOLATION"):
                    p_team = a_evt.get("team", "home")
                    if p_team == "home":
                        tracked_events["turnovers"] += 1
                        if evt_t == "TRAVELING":
                            tracked_events["turnovers_traveling"] += 1
                        elif evt_t == "FEHLPASS_AUS":
                            tracked_events["turnovers_bad_pass"] += 1
                        elif evt_t == "THREE_SECONDS_VIOLATION":
                            tracked_events["turnovers_three_seconds"] += 1
                        elif evt_t == "BACKCOURT_VIOLATION":
                            tracked_events["turnovers_backcourt"] += 1
                        elif evt_t == "PLAYER_OUT_OF_BOUNDS":
                            tracked_events["turnovers_out_of_bounds"] += 1
                        elif evt_t == "DOUBLE_DRIBBLE":
                            tracked_events["turnovers_double_dribble"] += 1
                        elif evt_t == "EIGHT_SECONDS_VIOLATION":
                            tracked_events["turnovers_eight_seconds"] += 1
                        elif evt_t == "FIVE_SECONDS_VIOLATION":
                            tracked_events["turnovers_five_seconds"] += 1
                elif evt_t == "FEHLPASS_STEAL":
                    if a_evt.get("passer_team") == "home":
                        tracked_events["turnovers"] += 1
                        tracked_events["turnovers_bad_pass"] += 1
                    if a_evt.get("stealer_team") == "home":
                        tracked_events["steals"] += 1
                elif evt_t == "BLOCK":
                    if a_evt.get("blocker_team") == "home":
                        tracked_events["blocks"] += 1

            frame_idx += 1
            if total_inferences % 250 == 0:
                pct = ((frame_idx - s_start) / max(1, s_end - s_start)) * 100.0
                print(f"  [{seg_name}] Frame {frame_idx}/{s_end} ({pct:.1f}%) | Shots: {len(shot_tracker.completed_shots)} | Pkt: {tracked_events['points']} | REB: {tracked_events['reb']} | TO: {tracked_events['turnovers']} | Roster: {len(player_registry.canonical_players)}/6", flush=True)

    cap.release()

    # 8. Evaluation against Official Ground Truth
    detected_jerseys = sorted(list(set(list(player_registry.canonical_players.keys()) + list(player_jerseys.values()))))
    evaluator = BenchmarkEvaluator()
    comparison = evaluator.evaluate_team_totals(tracked_events)
    player_rec = evaluator.evaluate_player_recognition(detected_jerseys)

    print("\n" + "="*82)
    print("📊 ERGEBNISSE DES TESTLAUFS & BENCHMARK-VERGLEICH (SB UFFENHEIM)")
    print("="*82)
    print(f"{'Kategorie / Event':<32} | {'Soll (DBB)':<10} | {'Ist (CV)':<10} | {'Diff':<8} | {'Erfassungsquote':<15}")
    print("-" * 82)
    for k, v in comparison.items():
        print(f"{v['label']:<32} | {v['ground_truth']:<10} | {v['detected']:<10} | {v['delta']:<+8.1f} | {v['capture_rate_pct']:>6.1f} %")
    print("-" * 82)
    print(f"Turnovers Breakdown: {tracked_events['turnovers_traveling']} Schrittfehler, {tracked_events['turnovers_bad_pass']} Fehlpässe, {tracked_events['turnovers_three_seconds']} 3-Sekunden, {tracked_events['turnovers_backcourt']} Rückspiel, {tracked_events['turnovers_out_of_bounds']} Aus, {tracked_events['turnovers_double_dribble']} Doppeldribbel, {tracked_events['turnovers_eight_seconds']} 8-Sek, {tracked_events['turnovers_five_seconds']} 5-Sek")
    print(f"Spielererkennung (Recall): {player_rec['recall_pct']}% ({len(player_rec['matched_jerseys'])} von {len(player_rec['gt_jerseys'])} Trikotnummern erkannt: {player_rec['matched_jerseys']})")
    
    canonical_summary = player_registry.get_canonical_roster_summary(team="home")
    if canonical_summary:
        print("\n📋 INDIVIDUELLE SPIELER-STATISTIKEN (SB UFFENHEIM):")
        print(f"{'Trikot':<8} | {'Punkte':<8} | {'FGM/FGA':<10} | {'REB (O/D)':<10} | {'Tracklets'}")
        print("-" * 55)
        for j_num, p_stats in sorted(canonical_summary.items()):
            print(f"#{j_num:<7} | {p_stats['points']:<8} | {p_stats['fgm']}/{p_stats['fga']:<8} | {p_stats['reb']} ({p_stats['oreb']}/{p_stats['dreb']}) | {p_stats['associated_tracks']}")
    print("="*82 + "\n")

    os.makedirs("results", exist_ok=True)

    # 9. Stat-Linked Video Clips Manifest (SportsVisio / Hudl Parity)
    manifest_path = "results/stat_linked_clips_manifest.json"
    clip_registry.export_manifest(manifest_path)
    print(f"🎬 Stat-verknüpfte Video-Clips exportiert ({len(clip_registry.clips)} Clips): {manifest_path}")

    # 10. AI Coach Game Summary & Four Factors (SportsVisio Coach Mode Parity)
    coach_data = coach_reporter.generate_report(tracked_events, canonical_summary or {})
    coach_report_md_path = "results/coach_game_report.md"
    coach_reporter.export_markdown_report(coach_data, coach_report_md_path)
    coach_report_json_path = "results/coach_game_report.json"
    with open(coach_report_json_path, "w", encoding="utf-8") as f:
        json.dump(coach_data, f, indent=4)
    print(f"📊 AI Coach Game Report (Four Factors) exportiert: {coach_report_md_path}")

    report = {
        "video": video_path,
        "mode": "multi_quarter" if multi_quarter else "single_segment",
        "processed_frames": total_inferences,
        "comparison": comparison,
        "player_recognition": player_rec,
        "canonical_players": canonical_summary,
        "detected_totals": tracked_events,
        "four_factors": coach_data["four_factors"],
        "coach_summary": coach_data["summary"],
        "total_stat_clips": len(clip_registry.clips),
        "ground_truth": OFFICIAL_GROUND_TRUTH_UFFENHEIM,
    }

    report_path = "results/match_oerlenbach_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        def json_default(obj):
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        json.dump(report, f, indent=4, default=json_default)

    print(f"Detaillierter JSON-Report gespeichert unter: {report_path}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="uploads/game_oerlenbach.mp4")
    parser.add_argument("--start", type=int, default=13000)
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--skip", type=int, default=3)
    parser.add_argument("--sahi", action="store_true", default=False)
    parser.add_argument("--multi-quarter", action="store_true", default=False)
    parser.add_argument("--full-game", action="store_true", default=False)
    parser.add_argument("--frames-per-quarter", type=int, default=1500)
    args = parser.parse_args()

    run_evaluation(
        args.video,
        start_frame=args.start,
        max_frames=args.frames,
        skip_frames=args.skip,
        use_sahi=args.sahi,
        multi_quarter=args.multi_quarter,
        full_game=args.full_game,
        frames_per_quarter=args.frames_per_quarter
    )
