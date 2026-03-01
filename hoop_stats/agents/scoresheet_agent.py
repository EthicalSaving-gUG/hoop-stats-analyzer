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


class ScoresheetAgent(BaseAgent):
    AGENT_NAME = "scoresheet"

    def __init__(self, game_state, stream_manager, camera_ids=None, audio_ids=None, **kwargs):
        super().__init__(game_state, stream_manager, camera_ids, audio_ids, tick_rate=10, **kwargs)
        self._detector = None
        self._shot_tracker = None
        self._last_event_count = 0
        self._known_players = set()

    def on_start(self):
        self._detector = HoopDetector()

    def process_frame(self, camera_id, frame, frame_idx):
        h, w = frame.shape[:2]

        # Initialize shot tracker on first frame
        if self._shot_tracker is None:
            self._shot_tracker = ShotTracker(w, h)

        # Detect players for substitution tracking
        det_results = self._detector.model.track(
            frame, persist=True, classes=[0], verbose=False,
            conf=0.35, tracker="botsort.yaml"
        )

        player_positions = {}
        current_players = set()

        if det_results[0].boxes.id is not None:
            boxes = det_results[0].boxes.xyxy.cpu().numpy()
            ids = det_results[0].boxes.id.cpu().numpy().astype(int)

            for obj_id, box in zip(ids, boxes):
                pid = str(obj_id)
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                player_positions[obj_id] = (cx, cy, cx, cy)
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

        # Detect ball + update shot tracker
        ball_boxes = self._detector.detect_ball(frame)
        ball_centers = []
        for b in ball_boxes:
            bx = (b[0] + b[2]) / 2
            by = (b[1] + b[3]) / 2
            ball_centers.append((bx, by))

        events = self._shot_tracker.update(ball_centers, player_positions, frame_idx)

        # Record scores
        for (zone_idx, score) in events:
            team = self.game_state.possession or "unknown"
            points = 2  # simplified; 3pt detection would need zone info
            self.game_state.add_score(team, points)
            self.game_state.append_event(
                self.AGENT_NAME, "SCORE",
                {"team": team, "points": points, "zone": zone_idx}, frame_idx
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
