"""
Referee Agent — Schiedsrichter-Modus.
Watches for rule violations and flags warnings.
"""
from .base_agent import BaseAgent


class RefereeAgent(BaseAgent):
    AGENT_NAME = "referee"

    def __init__(self, game_state, stream_manager, camera_ids=None, audio_ids=None, **kwargs):
        super().__init__(game_state, stream_manager, camera_ids, audio_ids, tick_rate=5, **kwargs)
        self._last_checked_event_count = 0
        self._alerted_violations = set()
        self._whistle_count = 0

    def on_whistle(self):
        """Whistle detected → log referee whistle event."""
        self._whistle_count += 1
        self.game_state.append_event(
            self.AGENT_NAME, "REFEREE_WHISTLE",
            {"whistle_number": self._whistle_count,
             "hint": "Mögliches Foul, Ausball oder Timeout"}
        )

    def process_frame(self, camera_id, frame, frame_idx):
        gs = self.game_state

        # 1. Check for shot clock violation (from shot_clock_agent events)
        self._check_events(frame_idx)

        # 2. Check 5-foul disqualification
        for pid, fouls in gs.player_fouls.items():
            alert_key = f"disqualified_{pid}"
            if fouls >= 5 and alert_key not in self._alerted_violations:
                self._alerted_violations.add(alert_key)
                gs.append_event(
                    self.AGENT_NAME, "PLAYER_DISQUALIFIED",
                    {"player_id": pid, "fouls": fouls}, frame_idx
                )

        # 3. Check team foul bonus (>=5 team fouls in a quarter → free throws)
        q = gs.quarter
        for team_label, fouls_dict in [("home", gs.team_fouls_home),
                                         ("away", gs.team_fouls_away)]:
            team_fouls = fouls_dict.get(q, 0)
            alert_key = f"bonus_{team_label}_q{q}"
            if team_fouls >= 5 and alert_key not in self._alerted_violations:
                self._alerted_violations.add(alert_key)
                gs.append_event(
                    self.AGENT_NAME, "TEAM_FOUL_BONUS",
                    {"team": team_label, "quarter": q, "fouls": team_fouls},
                    frame_idx
                )

        # 4. Backcourt violation detection (simplified)
        # If possession is known and ball crosses back over half-court
        # This would need ball position tracking; for now we check events
        # from the shot_clock_agent for possession changes that go backward

    def _check_events(self, frame_idx):
        """Scan new events from other agents for referee-relevant triggers."""
        gs = self.game_state
        events = gs.events[self._last_checked_event_count:]
        self._last_checked_event_count = len(gs.events)

        for evt in events:
            if evt.event_type == "SHOT_CLOCK_VIOLATION":
                alert_key = f"scv_{evt.timestamp}"
                if alert_key not in self._alerted_violations:
                    self._alerted_violations.add(alert_key)
                    gs.append_event(
                        self.AGENT_NAME, "VIOLATION_ALERT",
                        {"type": "24-Sekunden Verstoß",
                         "team": evt.data.get("team", "Unknown"),
                         "original_event": evt.event_type},
                        frame_idx
                    )

            elif evt.event_type == "QUARTER_END":
                # Reset alerted bonus flags for new quarter
                q_next = evt.data.get("quarter", 0) + 1
                self._alerted_violations.discard(f"bonus_home_q{q_next}")
                self._alerted_violations.discard(f"bonus_away_q{q_next}")
