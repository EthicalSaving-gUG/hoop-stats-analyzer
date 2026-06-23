"""
Scoresheet Agent — Kampfbericht (Anschreiber).
Records all baskets, fouls, substitutions, and generates the official score sheet.
Listens to events from other agents and maintains the running Kampfbericht.
"""
import time
import json
import os
from .base_agent import BaseAgent
from ..detector import HoopDetector
from ..shot_tracker import ShotTracker
from ..geometry import CourtGeometry
from ..primitives import bbox_to_player, Point2D


class ScoresheetAgent(BaseAgent):
    AGENT_NAME = "scoresheet"

    def __init__(self, game_state, stream_manager, camera_ids=None, audio_ids=None, **kwargs):
        super().__init__(game_state, stream_manager, camera_ids, audio_ids, tick_rate=10, **kwargs)
        self._detector = None
        self._geometry = None
        self._shot_tracker = None
        self._last_event_count = 0
        self._known_players = set()
        self._fused_players = {}

    def on_start(self):
        self._detector = HoopDetector()
        self._geometry = CourtGeometry()

    def process_frame(self, camera_id, frame, frame_idx):
        h, w = frame.shape[:2]

        # Initialize shot tracker on first frame
        if self._shot_tracker is None:
            self._shot_tracker = ShotTracker(w, h, court_geometry=self._geometry)

        # Detect players for substitution tracking
        det_results = self._detector.model.track(
            frame, persist=True, classes=[0], verbose=False,
            conf=0.35, tracker="botsort.yaml"
        )

        player_capsules = {}
        current_players = set()

        if det_results[0].boxes.id is not None:
            boxes = det_results[0].boxes.xyxy.cpu().numpy()
            ids = det_results[0].boxes.id.cpu().numpy().astype(int)

            for obj_id, box in zip(ids, boxes):
                # Construct proper PlayerCapsule
                capsule = bbox_to_player(box[0], box[1], box[2], box[3], obj_id=obj_id)
                
                # Transform coordinates to real-world meters
                foot = capsule.ground_contact
                mx, my = self._geometry.pixel_to_meter(foot.x, foot.y)
                
                # Spatial Fusion Matcher
                matched_global_id = None
                min_dist = float('inf')
                for gid, data in self._fused_players.items():
                    gx, gy = data["pos"]
                    dist = ((mx - gx)**2 + (my - gy)**2)**0.5
                    if dist < 1.5 and dist < min_dist:
                        min_dist = dist
                        matched_global_id = gid
                
                if matched_global_id is not None:
                    # Update existing global player with alpha smoothing (spatial moving average)
                    alpha = 0.3
                    gx, gy = self._fused_players[matched_global_id]["pos"]
                    new_x = (1 - alpha) * gx + alpha * mx
                    new_y = (1 - alpha) * gy + alpha * my
                    self._fused_players[matched_global_id]["pos"] = [new_x, new_y]
                    self._fused_players[matched_global_id]["last_update"] = frame_idx
                    self._fused_players[matched_global_id]["local_ids"][camera_id] = obj_id
                    global_id = matched_global_id
                else:
                    # Create new global player (starting from 1000 to avoid conflicting with YOLO ids)
                    global_id = 1000 + len(self._fused_players)
                    self._fused_players[global_id] = {
                        "pos": [mx, my],
                        "last_update": frame_idx,
                        "local_ids": {camera_id: obj_id}
                    }
                
                capsule.obj_id = global_id
                capsule.meter_pos = Point2D(self._fused_players[global_id]["pos"][0], self._fused_players[global_id]["pos"][1])
                
                player_capsules[global_id] = capsule
                pid = str(global_id)
                current_players.add(pid)

                # Register new players
                if pid not in self.game_state.player_stats:
                    self.game_state.player_stats[pid] = {
                        "jersey": "Unknown",
                        "team": "Unknown",
                        "points": 0,
                        "fouls": 0,
                        "fg_made": 0,
                        "fg_att": 0,
                    }

            # Prune old fused players (not updated in 150 frames)
            self._fused_players = {
                gid: data for gid, data in self._fused_players.items()
                if frame_idx - data["last_update"] < 150
            }

        # Update GameState active players
        self.game_state.active_players = current_players

        # Track substitutions
        new_in = current_players - self._known_players
        new_out = self._known_players - current_players

        # Phase 8: Official Substitutions.
        # Only log these if the referee authorized a substitution recently.
        # Otherwise, we just silently update _known_players to fix tracking flickering.
        if self._known_players and self.game_state.is_substitution_window_open:
            for pid in new_in:
                self.game_state.append_event(
                    self.AGENT_NAME, "OFFICIAL_SUB_IN",
                    {"player_id": pid}, frame_idx
                )

            for pid in new_out:
                if len(new_out) < 3:  # avoid false triggers from camera glitches
                    self.game_state.append_event(
                        self.AGENT_NAME, "OFFICIAL_SUB_OUT",
                        {"player_id": pid}, frame_idx
                    )

        self._known_players = current_players

        # Detect ball
        ball_boxes = self._detector.detect_ball(frame)
        ball_primitives = []
        for ball in ball_boxes:
            # Map ball position to real-world meters
            mx, my = self._geometry.pixel_to_meter(ball.center.x, ball.center.y)
            ball.meter_pos = Point2D(mx, my)
            ball_primitives.append(ball)

        # Update court tracking telemetry in GameState
        with self.game_state._lock:
            player_positions = {}
            for pid, cap in player_capsules.items():
                team = self.game_state.player_stats.get(str(pid), {}).get("team", "Unknown")
                if cap.meter_pos:
                    player_positions[str(pid)] = {
                        "team": team,
                        "pos": [round(cap.meter_pos.x, 2), round(cap.meter_pos.y, 2)]
                    }
            self.game_state.court_tracking["players"] = player_positions

            if ball_primitives and ball_primitives[0].meter_pos:
                b_pos = ball_primitives[0].meter_pos
                self.game_state.court_tracking["ball"] = [round(b_pos.x, 2), round(b_pos.y, 2)]
            else:
                self.game_state.court_tracking["ball"] = None

        # Get team assignments for shot tracker to calculate defender distance
        player_teams = {
            str(pid): self.game_state.player_stats.get(str(pid), {}).get("team", "Unknown")
            for pid in player_capsules
        }
        events = self._shot_tracker.update(ball_primitives, player_capsules, frame_idx, player_teams=player_teams)

        # Record scores and shot events
        for ev in events:
            if ev["type"] == "SHOT_MAKE":
                team = ev["team"]
                points = ev["points"]
                self.game_state.add_score(team, points)
                
                # Update player statistics
                sid = str(ev["shooter_id"])
                if sid in self.game_state.player_stats:
                    self.game_state.player_stats[sid]["fg_made"] += 1
                    self.game_state.player_stats[sid]["fg_att"] += 1
                    
                self.game_state.append_event(
                    self.AGENT_NAME, "SHOT_MAKE",
                    ev, frame_idx
                )
            elif ev["type"] == "SHOT_MISS":
                # Update player statistics
                sid = str(ev["shooter_id"])
                if sid in self.game_state.player_stats:
                    self.game_state.player_stats[sid]["fg_att"] += 1
                    
                self.game_state.append_event(
                    self.AGENT_NAME, "SHOT_MISS",
                    ev, frame_idx
                )

    def on_stop(self):
        """Generate the Kampfbericht JSON on stop."""
        report = self.generate_report()
        path = os.path.join("results", "kampfbericht.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[{self.AGENT_NAME}] Kampfbericht saved to {path}")

    def generate_report(self) -> dict:
        gs = self.game_state
        return {
            "game_info": {
                "home": gs.team_home_name,
                "away": gs.team_away_name,
                "final_score": f"{gs.score_home} : {gs.score_away}",
                "quarters_played": gs.quarter,
            },
            "team_fouls": {
                "home": dict(gs.team_fouls_home),
                "away": dict(gs.team_fouls_away),
            },
            "player_stats": dict(gs.player_stats),
            "event_log": [
                {
                    "time": e.timestamp,
                    "agent": e.agent,
                    "type": e.event_type,
                    "data": e.data,
                }
                for e in gs.events
            ],
        }
