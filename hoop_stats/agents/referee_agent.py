"""
Referee Agent — Schiedsrichter-Modus.
Watches for rule violations and flags warnings.
Supports both FIBA 2024 and NBA 2024-25 rule sets.
"""
from .base_agent import BaseAgent
from ..detector import HoopDetector
from ..geometry import CourtGeometry
import time


class RefereeAgent(BaseAgent):
    AGENT_NAME = "referee"

    def __init__(self, game_state, stream_manager, camera_ids=None, audio_ids=None, **kwargs):
        super().__init__(game_state, stream_manager, camera_ids, audio_ids, tick_rate=5, **kwargs)
        self._last_checked_event_count = 0
        self._alerted_violations = set()
        self._whistle_count = 0
        self._rules = game_state.rules  # FIBA or NBA
        self._paint_times = {}
        self._detector = None
        self._geometry = None

    def on_start(self):
        super().on_start()
        self._detector = HoopDetector()
        self._geometry = CourtGeometry()

    def on_whistle(self):
        """Whistle detected → log referee whistle event."""
        self._whistle_count += 1
        self.game_state.append_event(
            self.AGENT_NAME, "REFEREE_WHISTLE",
            {"whistle_number": self._whistle_count,
             "hint": "Mögliches Foul, Ausball oder Timeout",
             "ruleset": self._rules.name}
        )

    def process_frame(self, camera_id, frame, frame_idx):
        gs = self.game_state
        rules = self._rules

        # 1. Check for shot clock violation + other agent events
        self._check_events(frame_idx)

        # 2. Personal foul disqualification (FIBA: 5 fouls, NBA: 6 fouls)
        for pid, fouls in gs.player_fouls.items():
            alert_key = f"disqualified_{pid}"
            if fouls >= rules.personal_foul_limit and alert_key not in self._alerted_violations:
                self._alerted_violations.add(alert_key)
                gs.append_event(
                    self.AGENT_NAME, "PLAYER_DISQUALIFIED",
                    {"player_id": pid, "fouls": fouls,
                     "limit": rules.personal_foul_limit,
                     "ruleset": rules.name}, frame_idx
                )

        # 3. Team foul bonus (FIBA & NBA: 5th team foul triggers bonus)
        q = gs.quarter
        for team_label, fouls_dict in [("home", gs.team_fouls_home),
                                         ("away", gs.team_fouls_away)]:
            team_fouls = fouls_dict.get(q, 0)
            alert_key = f"bonus_{team_label}_q{q}"
            if team_fouls >= rules.team_foul_bonus_threshold and alert_key not in self._alerted_violations:
                self._alerted_violations.add(alert_key)
                gs.append_event(
                    self.AGENT_NAME, "TEAM_FOUL_BONUS",
                    {"team": team_label, "quarter": q, "fouls": team_fouls,
                     "threshold": rules.team_foul_bonus_threshold,
                     "ruleset": rules.name},
                    frame_idx
                )

        # 4. Defensive 3-second violation (NBA only)
        if rules.defensive_three_seconds:
            self._check_defensive_three_seconds(frame, frame_idx)

        # 5. Unsportsmanlike foul accumulation (FIBA only)
        if rules.has_unsportsmanlike_fouls:
            self._check_unsportsmanlike_disqualification(frame_idx)

    def _check_defensive_three_seconds(self, frame, frame_idx):
        """
        NBA Rule: Defensive player cannot remain in the paint for > 3 seconds
        without actively guarding an opponent. (Simplified detection)
        """
        if not self.game_state.game_clock_running:
            self._paint_times.clear()
            return

        # Use the video/game clock tick rate or frame index to calculate the duration
        # instead of wall-clock time which can vary depending on processing speed.
        now = frame_idx
        fps = getattr(self.game_state, "fps", getattr(self.stream_manager, "fps", 30))
        # fallback if we don't have exact fps. The tick rate is 5 fps by default
        tick_rate = self.tick_rate if hasattr(self, "tick_rate") else 5.0
        time_per_tick = 1.0 / tick_rate

        det_results = self._detector.model.track(
            frame, persist=True, classes=[0], verbose=False,
            conf=0.35, tracker="botsort.yaml"
        )

        current_players_in_paint = set()

        if det_results[0].boxes.id is not None:
            boxes = det_results[0].boxes.xyxy.cpu().numpy()
            ids = det_results[0].boxes.id.cpu().numpy().astype(int)

            for obj_id, box in zip(ids, boxes):
                pid = str(obj_id)

                # Bottom center (foot position)
                cx = (box[0] + box[2]) / 2
                cy = box[3]

                mx, my = self._geometry.pixel_to_meter(cx, cy)
                zone = self._geometry.get_zone(mx, my)

                if zone == "paint":
                    current_players_in_paint.add(pid)
                    if pid not in self._paint_times:
                        self._paint_times[pid] = now
                    else:
                        duration = (now - self._paint_times[pid]) * time_per_tick
                        alert_key = f"def_3sec_{pid}_{int(now)}"
                        if duration > 3.0 and alert_key not in self._alerted_violations:
                            # Simplified: We don't know who is on defense right now
                            # without advanced ball possession/team association.
                            # So we flag it as a generic 3-second violation.
                            self._alerted_violations.add(alert_key)
                            self.game_state.append_event(
                                self.AGENT_NAME, "DEFENSIVE_3_SECONDS",
                                {"player_id": pid, "duration": round(duration, 1),
                                 "ruleset": self._rules.name},
                                frame_idx
                            )
                            # Reset time so we don't spam
                            self._paint_times[pid] = now

        # Remove players who left the paint
        for pid in list(self._paint_times.keys()):
            if pid not in current_players_in_paint:
                del self._paint_times[pid]


    def _check_unsportsmanlike_disqualification(self, frame_idx):
        """
        FIBA Rule: 2 unsportsmanlike fouls OR 1 technical + 1 unsportsmanlike
        = disqualification. Check the event log for these combinations.
        """
        gs = self.game_state
        # Count unsportsmanlike events per player from the event log
        unsportsmanlike_counts = {}
        technical_counts = {}
        for evt in gs.events:
            if evt.event_type == "UNSPORTSMANLIKE_FOUL":
                pid = evt.data.get("player_id", "")
                unsportsmanlike_counts[pid] = unsportsmanlike_counts.get(pid, 0) + 1
            elif evt.event_type == "TECHNICAL_FOUL":
                pid = evt.data.get("player_id", "")
                technical_counts[pid] = technical_counts.get(pid, 0) + 1

        for pid, count in unsportsmanlike_counts.items():
            alert_key = f"unsportsmanlike_disq_{pid}"
            if alert_key in self._alerted_violations:
                continue

            # 2 unsportsmanlike fouls → disqualification
            if count >= self._rules.unsportsmanlike_disqualify_count:
                self._alerted_violations.add(alert_key)
                gs.append_event(
                    self.AGENT_NAME, "PLAYER_DISQUALIFIED",
                    {"player_id": pid, "reason": "2x Unsportsmanlike Foul",
                     "ruleset": self._rules.name}, frame_idx
                )
            # 1 technical + 1 unsportsmanlike → disqualification
            elif count >= 1 and technical_counts.get(pid, 0) >= 1:
                self._alerted_violations.add(alert_key)
                gs.append_event(
                    self.AGENT_NAME, "PLAYER_DISQUALIFIED",
                    {"player_id": pid,
                     "reason": "Technical + Unsportsmanlike Foul Kombination",
                     "ruleset": self._rules.name}, frame_idx
                )

    def _check_events(self, frame_idx):
        """Scan new events from other agents for referee-relevant triggers."""
        gs = self.game_state
        rules = self._rules
        events = gs.events[self._last_checked_event_count:]
        self._last_checked_event_count = len(gs.events)

        for evt in events:
            if evt.event_type == "SHOT_CLOCK_VIOLATION":
                alert_key = f"scv_{evt.timestamp}"
                if alert_key not in self._alerted_violations:
                    self._alerted_violations.add(alert_key)
                    gs.append_event(
                        self.AGENT_NAME, "VIOLATION_ALERT",
                        {"type": f"{int(rules.shot_clock_s)}-Sekunden Verstoß",
                         "team": evt.data.get("team", "Unknown"),
                         "ruleset": rules.name},
                        frame_idx
                    )

            elif evt.event_type == "QUARTER_END":
                # Reset alerted bonus flags for new quarter
                q_next = evt.data.get("quarter", 0) + 1
                self._alerted_violations.discard(f"bonus_home_q{q_next}")
                self._alerted_violations.discard(f"bonus_away_q{q_next}")
