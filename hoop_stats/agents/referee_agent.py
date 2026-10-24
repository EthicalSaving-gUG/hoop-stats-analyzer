"""
Referee Agent — Schiedsrichter-Modus.
Watches for rule violations and flags warnings.
Supports both FIBA 2024 and NBA 2024-25 rule sets.
"""
from .base_agent import BaseAgent


class RefereeAgent(BaseAgent):
    AGENT_NAME = "referee"

    def __init__(self, game_state, stream_manager, camera_ids=None, audio_ids=None, **kwargs):
        super().__init__(game_state, stream_manager, camera_ids, audio_ids, tick_rate=5, **kwargs)
        self._last_checked_event_count = 0
        self._alerted_violations = set()
        self._whistle_count = 0
        self._rules = game_state.rules  # FIBA or NBA

        # AI Referee Pro State variables
        self._possession_player = None
        self._possession_team = None
        self._steps_count = 0
        self._last_player_pos = None
        self._dribble_state = "NONE"
        self._held_time = 0
        self._established_frontcourt = False

        # Pass and Dribble Tracking
        self._last_possessor = None
        self._last_possessor_frame = -999
        self._last_possessor_pos = None
        self._last_processed_frame = None

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

        # 6. AI Referee Pro Rules (Traveling, Double-Dribble, Backcourt Violation)
        self._check_ai_referee_pro_rules(frame_idx)

    def _check_ai_referee_pro_rules(self, frame_idx):
        gs = self.game_state
        ct = getattr(gs, "court_tracking", None)
        if not ct:
            return

        ball = ct.get("ball")
        players = ct.get("players", {})

        if not ball or not players:
            return

        # Track tick time delta
        if self._last_processed_frame is None:
            self._last_processed_frame = frame_idx
            dt = 0.0
        else:
            frame_delta = frame_idx - self._last_processed_frame
            self._last_processed_frame = frame_idx
            dt = frame_delta / 30.0 if frame_delta > 0 else 0.0

        # 1. Determine player in possession
        # Find player closest to the ball
        closest_pid = None
        closest_dist = float('inf')
        for pid, pdata in players.items():
            if pdata.get("team") == "Referee" or pdata.get("role") == "referee":
                continue
            ppos = pdata.get("pos")
            if ppos:
                dist = ((ball[0] - ppos[0])**2 + (ball[1] - ppos[1])**2)**0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pid = pid

        # A player possesses the ball if they are within 1.2 meters
        current_possessor = None
        if closest_dist < 1.2:
            current_possessor = closest_pid

        if current_possessor:
            possessor_data = players[current_possessor]
            team = possessor_data.get("team", "Unknown")
            ppos = possessor_data.get("pos")
            
            # Switch possession to possessor's team
            if gs.possession != team and team in ("home", "away"):
                gs.set_possession(team)

            # Detect Teammate Pass
            if self._possession_player is None:
                # Ball was loose/in-flight, now caught
                if self._last_possessor is not None and self._last_possessor != current_possessor:
                    time_delta = (frame_idx - self._last_possessor_frame) / 30.0
                    if 0.1 < time_delta < 1.5:
                        if ppos and self._last_possessor_pos:
                            dist_m = ((ppos[0] - self._last_possessor_pos[0])**2 + (ppos[1] - self._last_possessor_pos[1])**2)**0.5
                            velocity_ms = dist_m / time_delta if time_delta > 0 else 0.0
                            
                            # Check teams
                            passer_team = gs.player_stats.get(str(self._last_possessor), {}).get("team", "Unknown")
                            receiver_team = gs.player_stats.get(str(current_possessor), {}).get("team", "Unknown")
                            
                            if passer_team == receiver_team and passer_team != "Unknown":
                                gs.append_event(
                                    self.AGENT_NAME, "PASS_DETECTED",
                                    {
                                        "passer_id": self._last_possessor,
                                        "receiver_id": current_possessor,
                                        "team": passer_team,
                                        "velocity_ms": round(velocity_ms, 2),
                                        "distance_m": round(dist_m, 2),
                                        "duration_s": round(time_delta, 2)
                                    },
                                    frame_idx
                                )
                                print(f"  📉 PASS DETECTED from Player {self._last_possessor} to {current_possessor} ({passer_team}) at {velocity_ms:.2f} m/s!")

            # If player carrier changed, reset state machine
            if self._possession_player != current_possessor:
                self._possession_player = current_possessor
                self._possession_team = team
                self._steps_count = 0
                self._last_player_pos = ppos
                self._dribble_state = "DRIBBLING"
                self._held_time = 0
                
                # Check if frontcourt is established immediately
                if team == "home":
                    self._established_frontcourt = (ppos[0] > 14.0) if ppos else False
                else:
                    self._established_frontcourt = (ppos[0] < 14.0) if ppos else False
            else:
                # Same player still has possession. Update their dribble duration
                pid_str = str(current_possessor)
                if pid_str not in gs.player_stats:
                    gs.player_stats[pid_str] = {
                        "jersey": "Unknown",
                        "team": "Unknown",
                        "points": 0,
                        "fouls": 0,
                        "fg_made": 0,
                        "fg_att": 0,
                        "dribble_duration_s": 0.0
                    }
                if "dribble_duration_s" not in gs.player_stats[pid_str]:
                    gs.player_stats[pid_str]["dribble_duration_s"] = 0.0
                gs.player_stats[pid_str]["dribble_duration_s"] += dt

                # ── Schrittfehler (Traveling) & Dribbelfehler (Double-Dribble) Heuristics ──
                
                # A. Track steps / movement
                if self._last_player_pos and ppos:
                    move_dist = ((ppos[0] - self._last_player_pos[0])**2 + (ppos[1] - self._last_player_pos[1])**2)**0.5
                    # A step is a significant movement of the player
                    if move_dist > 0.6:
                        # If player is in HELD state, they can only take 2 steps
                        if self._dribble_state == "HELD":
                            self._steps_count += 1
                            self._last_player_pos = ppos
                            
                            if self._steps_count > 2:
                                alert_key = f"traveling_{current_possessor}_{frame_idx}"
                                if not any(k.startswith(f"traveling_{current_possessor}") for k in self._alerted_violations):
                                    self._alerted_violations.add(alert_key)
                                    gs.append_event(
                                        self.AGENT_NAME, "VIOLATION_ALERT",
                                        {
                                            "type": "Schrittfehler (Traveling)",
                                            "team": team,
                                            "player_id": current_possessor,
                                            "steps": self._steps_count,
                                            "ruleset": self._rules.name
                                        },
                                        frame_idx
                                    )
                        else:
                            self._last_player_pos = ppos

                # B. Track Dribble State Transitions
                # If the ball remains very close and stable relative to player, player is holding (HELD).
                if closest_dist < 0.4:
                    self._held_time += 1
                    if self._held_time > 4: # held for ~0.4s (at 10fps)
                        if self._dribble_state == "DRIBBLING":
                            self._dribble_state = "HELD"
                            self._steps_count = 0  # Start step count for holding
                else:
                    self._held_time = 0
                    if self._dribble_state == "HELD" and closest_dist < 1.0:
                        # Player was holding the ball, but starts dribbling again!
                        alert_key = f"doubledribble_{current_possessor}_{frame_idx}"
                        if not any(k.startswith(f"doubledribble_{current_possessor}") for k in self._alerted_violations):
                            self._alerted_violations.add(alert_key)
                            gs.append_event(
                                self.AGENT_NAME, "VIOLATION_ALERT",
                                {
                                    "type": "Dribbelfehler (Double-Dribble)",
                                    "team": team,
                                    "player_id": current_possessor,
                                    "ruleset": self._rules.name
                                },
                                frame_idx
                            )
                            self._dribble_state = "DRIBBLING"

                # ── Rückspiel (Backcourt Violation) Heuristics ──
                if team == "home":
                    is_frontcourt = (ppos[0] > 14.0) if ppos else False
                else:
                    is_frontcourt = (ppos[0] < 14.0) if ppos else False

                if is_frontcourt and not self._established_frontcourt:
                    self._established_frontcourt = True
                
                elif self._established_frontcourt and not is_frontcourt:
                    # Attacking team had established frontcourt, but went back to backcourt!
                    alert_key = f"backcourt_{team}_{frame_idx}"
                    if not any(k.startswith(f"backcourt_{team}") for k in self._alerted_violations):
                        self._alerted_violations.add(alert_key)
                        gs.append_event(
                            self.AGENT_NAME, "VIOLATION_ALERT",
                            {
                                "type": "Rückspiel (Backcourt Violation)",
                                "team": team,
                                "player_id": current_possessor,
                                "coord_x": round(ppos[0], 2),
                                "ruleset": self._rules.name
                            },
                            frame_idx
                        )
                        self._established_frontcourt = False
        else:
            # Ball has no possessor (in flight or loose ball)
            if self._possession_player is not None:
                self._last_possessor = self._possession_player
                self._last_possessor_frame = frame_idx
                self._last_possessor_pos = self._last_player_pos
                self._possession_player = None
                self._possession_team = None

    def _check_defensive_three_seconds(self, frame, frame_idx):
        """
        NBA Rule: Defensive player cannot remain in the paint for > 3 seconds
        without actively guarding an opponent. (Simplified detection placeholder)
        """
        # Full implementation would track each defender's time in the paint zone
        # using player position data from the detector.
        # For now, this is a placeholder that can be extended.
        pass

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
