"""
event_detector.py
=================
Event Detector for Advanced Basketball Events:
- Schrittfehler (Traveling Violations, FIBA Art. 25)
- Fehlpässe (Bad Passes: Interceptions & Out-of-Bounds Passes)
- Steals (Defensive Ballgewinne / Ball Interceptions)
- Assists (Vorlagen zu erfolgreichen Korbwürfen)
- Blocks (Wurfblöcke gegen gegnerische Korbversuche)
"""

import math
from typing import Dict, List, Optional, Tuple
from .primitives import Ball, PlayerCapsule, Point2D
from .player_registry import PlayerRegistry


class EventDetector:
    """
    Detects complex basketball game events and rule violations:
    1. Traveling (Schrittfehler): Ball handler displaces > 1.8m without dribbling.
    2. Bad Pass (Fehlpass) & Steal: In-flight ball intercepted by opponent or passing out-of-bounds.
    3. Assist: Pass directly leading to a made basket within 2.5s.
    4. Block: Defensive deflection of an offensive shot attempt in rising phase.
    """

    def __init__(self,
                 frame_w: int,
                 frame_h: int,
                 court_geometry=None,
                 player_registry: Optional[PlayerRegistry] = None,
                 fps: float = 30.0,
                 traveling_dist_m: float = 2.6,
                 possession_radius_px: float = 85.0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.geometry = court_geometry
        self.player_registry = player_registry
        self.fps = fps
        self.traveling_dist_m = traveling_dist_m
        self.possession_radius_px = possession_radius_px

        # Possession state
        self.current_handler_id: Optional[int] = None
        self.current_handler_team: str = "unknown"
        self.possession_start_frame: int = 0
        self.handler_start_pos: Optional[Tuple[float, float]] = None
        self.handler_last_pos: Optional[Tuple[float, float]] = None
        self.handler_accumulated_dist_m: float = 0.0
        self.dribble_bounces: int = 0
        self.last_ball_y: Optional[float] = None
        self.ball_is_descending: bool = False

        # Pass tracking state
        self.active_pass: Optional[dict] = None
        self.last_completed_pass: Optional[dict] = None  # {passer_id, receiver_id, frame}
        self.defender_intercept_candidate: Optional[int] = None
        self.defender_intercept_count: int = 0

        # Event cooldowns: track_id -> frames remaining
        self.cooldowns: Dict[int, int] = {}
        # Global event log
        self.logged_events: List[dict] = []

        # Floor contact & FIBA rule violation tracking
        self.home_attacking_basket: int = 1
        self.team_frontcourt_established: Dict[str, bool] = {"home": False, "away": False}
        self.offensive_paint_time: Dict[int, int] = {}  # track_id -> frame_idx entered

        # Step-based Traveling & Ball Handling state (FIBA Art. 24 & 25)
        self.steps_since_gather: int = 0
        self.last_step_plant_pos: Optional[Tuple[float, float]] = None
        self.dribble_state: str = "none"  # "none", "dribbling", "gathered"
        self.last_bounce_frame: int = 0
        self.stride_threshold_m: float = 0.65  # Discrete step stride threshold

        # Time-based rule violations (FIBA Art. 27 & 28)
        self.backcourt_possession_start_frame: Optional[int] = None
        self.closely_guarded_start_frame: Optional[int] = None

    def swap_sides(self):
        """Swaps attacking court halves at halftime."""
        self.home_attacking_basket = 1 - self.home_attacking_basket
        self.team_frontcourt_established = {"home": False, "away": False}
        self.offensive_paint_time.clear()
        self.backcourt_possession_start_frame = None
        self.closely_guarded_start_frame = None

    def update(self,
               balls: List[Ball],
               player_capsules: Dict[int, PlayerCapsule],
               frame_idx: int,
               player_teams: Optional[Dict[str, str]] = None,
               active_shots: Optional[List[dict]] = None) -> List[dict]:
        """
        Processes current frame detections and emits newly recognized events.
        """
        events = []
        player_teams = player_teams or {}

        # Decrement cooldowns
        for tid in list(self.cooldowns.keys()):
            if self.cooldowns[tid] > 0:
                self.cooldowns[tid] -= 1
            else:
                del self.cooldowns[tid]

        if not balls or not player_capsules:
            return events

        ball = balls[0]
        ball_pos = (ball.center.x, ball.center.y)

        # ── 1. Find Closest Player to Ball ─────────────────────────────────
        closest_pid = None
        min_p_dist = float("inf")
        for pid, cap in player_capsules.items():
            d_center = ball.center.distance_to(cap.center)
            d_head = ball.center.distance_to(cap.head)
            d_foot = ball.center.distance_to(cap.foot)
            d = min(d_center, d_head, d_foot)
            if d < min_p_dist:
                min_p_dist = d
                closest_pid = pid

        thresh = max(self.possession_radius_px, player_capsules[closest_pid].height * 0.85) if closest_pid else self.possession_radius_px
        is_near_player = (closest_pid is not None) and (min_p_dist <= thresh)

        # ── 2. Dribble & Step Tracking (Traveling / Schrittfehler) ─────────
        if is_near_player and closest_pid is not None:
            cap = player_capsules[closest_pid]
            p_pos = (cap.center.x, cap.center.y)
            p_team = player_teams.get(str(closest_pid), "home")

            # Check if possession is starting or continuing
            if self.current_handler_id != closest_pid:
                # New possession begins
                self.current_handler_id = closest_pid
                self.current_handler_team = p_team
                self.possession_start_frame = frame_idx
                self.handler_start_pos = p_pos
                self.handler_last_pos = p_pos
                self.last_step_plant_pos = p_pos
                self.steps_since_gather = 0
                self.dribble_state = "none"
                self.handler_accumulated_dist_m = 0.0
                self.dribble_bounces = 0
                self.last_ball_y = ball.center.y
                self.ball_is_descending = False
                self.backcourt_possession_start_frame = frame_idx
                self.closely_guarded_start_frame = None

            possession_len = frame_idx - self.possession_start_frame

            if self.current_handler_id == closest_pid and self.handler_last_pos:
                # Track player movement displacement & discrete step cadence (FIBA Art. 25)
                if self.geometry and cap.meter_pos:
                    dist_m = math.hypot(cap.meter_pos.x - self.handler_last_pos[0],
                                        cap.meter_pos.y - self.handler_last_pos[1])
                else:
                    h_px = max(40.0, cap.height)
                    scale_m_per_px = 1.85 / h_px
                    dx = p_pos[0] - self.handler_last_pos[0]
                    dy = p_pos[1] - self.handler_last_pos[1]
                    dist_m = math.hypot(dx, dy) * scale_m_per_px

                if dist_m > 0.02:
                    self.handler_accumulated_dist_m += dist_m

                self.handler_last_pos = p_pos

                # Discrete Step Count Tracking (Gather Step 0 -> Step 1 Pivot -> Step 2 -> Step 3 Violation)
                if self.last_step_plant_pos:
                    if self.geometry and cap.meter_pos:
                        step_dist_m = math.hypot(cap.meter_pos.x - self.last_step_plant_pos[0],
                                                 cap.meter_pos.y - self.last_step_plant_pos[1])
                    else:
                        h_px = max(40.0, cap.height)
                        scale_m_per_px = 1.85 / h_px
                        step_dist_m = math.hypot(p_pos[0] - self.last_step_plant_pos[0],
                                                 p_pos[1] - self.last_step_plant_pos[1]) * scale_m_per_px

                    # A discrete foot plant occurs when displacement exceeds stride threshold (~0.65m)
                    if step_dist_m >= self.stride_threshold_m:
                        if self.dribble_state != "dribbling":
                            self.steps_since_gather += 1
                        self.last_step_plant_pos = (cap.meter_pos.x, cap.meter_pos.y) if (self.geometry and cap.meter_pos) else p_pos

                # Track ball vertical oscillation (dribble bounce)
                if self.last_ball_y is not None:
                    dy = ball.center.y - self.last_ball_y
                    if dy > 2.0:
                        self.ball_is_descending = True
                    elif dy < -2.0 and self.ball_is_descending:
                        # Ball bounced off the floor and is moving upward
                        self.ball_is_descending = False

                        # Check for Double Dribble (Zweimaliges Dribbeln, FIBA Art. 24):
                        # Player already completed a dribble (gathered ball) and begins bouncing again!
                        if self.dribble_state == "gathered" and self.dribble_bounces >= 1 and self.cooldowns.get("turnover", 0) == 0:
                            evt = {
                                "type": "DOUBLE_DRIBBLE",
                                "player_id": closest_pid,
                                "team": p_team,
                                "frame": frame_idx,
                                "description": f"Zweimaliges Dribbeln (Double Dribble, FIBA Art. 24) by Player {closest_pid} ({p_team})"
                            }
                            events.append(evt)
                            self.logged_events.append(evt)
                            self.cooldowns[closest_pid] = int(self.fps * 2.5)
                            self.cooldowns["turnover"] = int(self.fps * 2.5)
                            if self.player_registry:
                                self.player_registry.record_action(closest_pid, "DOUBLE_DRIBBLE", details={"reason": "double_dribble"})
                            print(f"  🚨 [EVENT] {evt['description']}")

                        self.dribble_bounces += 1
                        self.dribble_state = "dribbling"
                        self.last_bounce_frame = frame_idx
                        # Active legal dribble resets step counter
                        self.steps_since_gather = 0
                        self.handler_accumulated_dist_m = 0.0
                        self.closely_guarded_start_frame = None

                self.last_ball_y = ball.center.y

                # Detect when dribble ceases / ball gathered into hands (> 0.45s without bounce)
                if self.dribble_state == "dribbling" and (frame_idx - self.last_bounce_frame) > int(self.fps * 0.45):
                    self.dribble_state = "gathered"
                    self.steps_since_gather = 0  # Gather step 0 established upon termination
                    self.last_step_plant_pos = (cap.meter_pos.x, cap.meter_pos.y) if (self.geometry and cap.meter_pos) else p_pos

                # Check for Traveling (Schrittfehler, FIBA Art. 25)
                # Player takes >= 3 discrete steps after ball gathering without releasing shot/pass/dribble,
                # or displaces >= traveling_dist_m with 0 dribble bounces.
                is_shooting = any(s.get("shooter_id") == closest_pid and s.get("status") == "rising"
                                  for s in (active_shots or []))
                is_traveling_steps = (self.steps_since_gather >= 3 and self.dribble_state != "dribbling")
                is_traveling_dist = (self.handler_accumulated_dist_m >= self.traveling_dist_m and self.dribble_bounces == 0)

                if ((is_traveling_steps or is_traveling_dist) and
                        possession_len >= 10 and
                        not is_shooting and
                        self.cooldowns.get(closest_pid, 0) == 0 and
                        self.cooldowns.get("turnover", 0) == 0):

                    steps_taken = max(3, self.steps_since_gather)
                    evt = {
                        "type": "TRAVELING",
                        "player_id": closest_pid,
                        "team": p_team,
                        "frame": frame_idx,
                        "steps": steps_taken,
                        "distance_m": round(self.handler_accumulated_dist_m, 2),
                        "bounces": self.dribble_bounces,
                        "description": f"Schrittfehler (Traveling, FIBA Art. 25) by Player {closest_pid} ({p_team}): {steps_taken} Schritte nach Ballaufnahme ({self.handler_accumulated_dist_m:.2f}m)"
                    }
                    events.append(evt)
                    self.logged_events.append(evt)
                    self.cooldowns[closest_pid] = int(self.fps * 2.5)  # 2.5s cooldown
                    self.handler_accumulated_dist_m = 0.0
                    self.steps_since_gather = 0

                    if self.player_registry:
                        self.player_registry.record_action(
                            closest_pid, "TURNOVER",
                            details={"reason": "traveling", "steps": steps_taken, "distance_m": evt["distance_m"]}
                        )
                    print(f"  🚨 [EVENT] {evt['description']}")

            # ── 2b. Floor Contact Violations (FIBA Art. 23, 26, 30) ────────
            target_basket = self.home_attacking_basket if p_team == "home" else (1 - self.home_attacking_basket)
            floor_status = None
            if self.geometry and hasattr(self.geometry, "evaluate_player_floor_status"):
                floor_status = self.geometry.evaluate_player_floor_status(cap, attacking_basket=target_basket)

            # A. Out of Bounds on Ball Handler (FIBA Art. 23)
            # The boundary lines belong to out-of-bounds! Stepping on the line with the ball is a turnover.
            is_oob = False
            if floor_status:
                is_oob = floor_status["is_out_of_bounds"]
            elif self.geometry and hasattr(self.geometry, "is_out_of_bounds"):
                m_pos = self.geometry.pixel_to_meter(cap.ground_contact.x, cap.ground_contact.y)
                is_oob = self.geometry.is_out_of_bounds(m_pos[0], m_pos[1])
            else:
                gc = cap.ground_contact
                is_oob = (gc.x < 35 or gc.x > (self.frame_w - 35) or gc.y > (self.frame_h - 25))

            if is_oob and possession_len >= 5 and self.cooldowns.get("turnover", 0) == 0 and self.cooldowns.get(closest_pid, 0) == 0:
                evt = {
                    "type": "PLAYER_OUT_OF_BOUNDS",
                    "player_id": closest_pid,
                    "team": p_team,
                    "frame": frame_idx,
                    "description": f"Ballführer im Aus (Player Out of Bounds) by Player {closest_pid} ({p_team})"
                }
                events.append(evt)
                self.logged_events.append(evt)
                self.cooldowns[closest_pid] = int(self.fps * 2.5)
                self.cooldowns["turnover"] = int(self.fps * 2.5)
                if self.player_registry:
                    self.player_registry.record_action(
                        closest_pid, "PLAYER_OUT_OF_BOUNDS",
                        details={"reason": "player_out_of_bounds"}
                    )
                print(f"  🚨 [EVENT] {evt['description']}")

            # B. Backcourt Violation / Rückspiel (FIBA Art. 30)
            # Frontcourt is established only when both feet of the ballhandler are in the frontcourt.
            # Once established, neither the player nor the ball may touch the centerline or backcourt floor!
            if floor_status:
                if floor_status["frontcourt_established"]:
                    self.team_frontcourt_established[p_team] = True

                if self.team_frontcourt_established.get(p_team, False):
                    touches_backcourt = floor_status["touches_backcourt"] or floor_status["touches_centerline"]
                    if touches_backcourt and possession_len >= 6 and self.cooldowns.get("turnover", 0) == 0 and self.cooldowns.get(closest_pid, 0) == 0:
                        evt = {
                            "type": "BACKCOURT_VIOLATION",
                            "player_id": closest_pid,
                            "team": p_team,
                            "frame": frame_idx,
                            "description": f"Rückspiel (Backcourt Violation / Over-and-Back) by Player {closest_pid} ({p_team})"
                        }
                        events.append(evt)
                        self.logged_events.append(evt)
                        self.cooldowns[closest_pid] = int(self.fps * 2.5)
                        self.cooldowns["turnover"] = int(self.fps * 2.5)
                        self.team_frontcourt_established[p_team] = False
                        if self.player_registry:
                            self.player_registry.record_action(
                                closest_pid, "BACKCOURT_VIOLATION",
                                details={"reason": "backcourt_violation"}
                            )
                        print(f"  🚨 [EVENT] {evt['description']}")

            # C. Three Seconds in the Key / 3 Sekunden in der Zone (FIBA Art. 26)
            # An offensive player cannot remain in the opponent's restricted area > 3.0 consecutive seconds
            # while their team has control in the frontcourt.
            if self.team_frontcourt_established.get(p_team, False) and not active_shots:
                for pid, p_cap in player_capsules.items():
                    tm_team = player_teams.get(str(pid), "unknown")
                    if tm_team != "unknown" and tm_team != p_team:
                        continue  # In FIBA rules, ONLY offensive players can commit 3-second violations!

                    if self.geometry and hasattr(self.geometry, "evaluate_player_floor_status"):
                        p_floor = self.geometry.evaluate_player_floor_status(p_cap, attacking_basket=target_basket)
                        # Even one foot touching the paint line => in the restricted area
                        if p_floor["in_paint"]:
                            if pid not in self.offensive_paint_time:
                                self.offensive_paint_time[pid] = frame_idx
                            else:
                                duration_s = (frame_idx - self.offensive_paint_time[pid]) / self.fps
                                if duration_s >= 3.0 and self.cooldowns.get(pid, 0) == 0 and self.cooldowns.get("turnover", 0) == 0:
                                    evt = {
                                        "type": "THREE_SECONDS_VIOLATION",
                                        "player_id": pid,
                                        "team": p_team,
                                        "frame": frame_idx,
                                        "duration_s": round(duration_s, 1),
                                        "description": f"3-Sekunden-Verletzung (3 Seconds in Key) by Player {pid} ({p_team}): in paint for {duration_s:.1f}s"
                                    }
                                    events.append(evt)
                                    self.logged_events.append(evt)
                                    self.cooldowns[pid] = int(self.fps * 3.0)
                                    self.cooldowns["turnover"] = int(self.fps * 2.5)
                                    self.offensive_paint_time.pop(pid, None)
                                    if self.player_registry:
                                        self.player_registry.record_action(
                                            pid, "THREE_SECONDS_VIOLATION",
                                            details={"reason": "three_seconds", "duration_s": duration_s}
                                        )
                                    print(f"  🚨 [EVENT] {evt['description']}")
                        elif p_floor["paint_cleared"]:
                            # Both feet completely outside the restricted area clears the count
                            self.offensive_paint_time.pop(pid, None)

            # D. Eight Seconds Backcourt Rule (8 Sekunden im Rückfeld, FIBA Art. 28)
            # A team that gains ball control in the backcourt must advance the ball across the centerline within 8.0s.
            is_in_backcourt = False
            if floor_status:
                is_in_backcourt = floor_status["touches_backcourt"] or floor_status["touches_centerline"]
            elif self.geometry and hasattr(self.geometry, "is_backcourt"):
                m_pos = self.geometry.pixel_to_meter(cap.ground_contact.x, cap.ground_contact.y)
                is_in_backcourt = self.geometry.is_backcourt(m_pos[0], m_pos[1], target_basket)

            if is_in_backcourt and not self.team_frontcourt_established.get(p_team, False):
                if self.backcourt_possession_start_frame is None:
                    self.backcourt_possession_start_frame = frame_idx
                else:
                    bc_sec = (frame_idx - self.backcourt_possession_start_frame) / self.fps
                    if bc_sec >= 8.0 and self.cooldowns.get("turnover", 0) == 0 and self.cooldowns.get(closest_pid, 0) == 0:
                        evt = {
                            "type": "EIGHT_SECONDS_VIOLATION",
                            "player_id": closest_pid,
                            "team": p_team,
                            "frame": frame_idx,
                            "duration_s": round(bc_sec, 1),
                            "description": f"8-Sekunden-Übertretung im Rückfeld (FIBA Art. 28) by {p_team} ({closest_pid}): {bc_sec:.1f}s"
                        }
                        events.append(evt)
                        self.logged_events.append(evt)
                        self.cooldowns[closest_pid] = int(self.fps * 2.5)
                        self.cooldowns["turnover"] = int(self.fps * 2.5)
                        self.backcourt_possession_start_frame = None
                        if self.player_registry:
                            self.player_registry.record_action(closest_pid, "EIGHT_SECONDS_VIOLATION", details={"reason": "eight_seconds"})
                        print(f"  🚨 [EVENT] {evt['description']}")
            elif floor_status and floor_status["frontcourt_established"]:
                self.backcourt_possession_start_frame = None

            # E. Five Seconds Closely Guarded Rule (5 Sekunden eng bewacht, FIBA Art. 27)
            # A closely guarded player (defender within 1.0m) holding the ball must pass, shoot or dribble within 5.0s.
            min_def_dist_m = float("inf")
            for pid, def_cap in player_capsules.items():
                if pid == closest_pid:
                    continue
                d_team = player_teams.get(str(pid), "unknown")
                if d_team != "unknown" and d_team == p_team:
                    continue
                if self.geometry and def_cap.meter_pos and cap.meter_pos:
                    d_m = cap.meter_pos.distance_to(def_cap.meter_pos)
                else:
                    d_px = cap.center.distance_to(def_cap.center)
                    h = max(40.0, cap.height)
                    d_m = d_px * (1.85 / h)
                if d_m < min_def_dist_m:
                    min_def_dist_m = d_m

            if min_def_dist_m <= 1.05 and self.dribble_state != "dribbling":
                if self.closely_guarded_start_frame is None:
                    self.closely_guarded_start_frame = frame_idx
                else:
                    cg_sec = (frame_idx - self.closely_guarded_start_frame) / self.fps
                    if cg_sec >= 5.0 and self.cooldowns.get("turnover", 0) == 0 and self.cooldowns.get(closest_pid, 0) == 0:
                        evt = {
                            "type": "FIVE_SECONDS_VIOLATION",
                            "player_id": closest_pid,
                            "team": p_team,
                            "frame": frame_idx,
                            "duration_s": round(cg_sec, 1),
                            "description": f"5-Sekunden-Übertretung eng gedeckt (FIBA Art. 27) by Player {closest_pid} ({p_team}): {cg_sec:.1f}s"
                        }
                        events.append(evt)
                        self.logged_events.append(evt)
                        self.cooldowns[closest_pid] = int(self.fps * 2.5)
                        self.cooldowns["turnover"] = int(self.fps * 2.5)
                        self.closely_guarded_start_frame = None
                        if self.player_registry:
                            self.player_registry.record_action(closest_pid, "FIVE_SECONDS_VIOLATION", details={"reason": "five_seconds"})
                        print(f"  🚨 [EVENT] {evt['description']}")
            else:
                self.closely_guarded_start_frame = None

        # ── 3. Passing, Fehlpässe (Bad Passes) & Steals ─────────────────────
        # Check if ball has left current handler (potential pass in flight)
        if self.current_handler_id is not None and not is_near_player:
            handler_cap = player_capsules.get(self.current_handler_id)
            if handler_cap:
                dist_from_handler = ball.center.distance_to(handler_cap.center)
                is_shot = bool(active_shots)

                if dist_from_handler > 90.0 and not is_shot and self.active_pass is None:
                    # Pass initiated!
                    self.active_pass = {
                        "passer_id": self.current_handler_id,
                        "passer_team": self.current_handler_team,
                        "start_frame": frame_idx,
                        "start_pos": ball_pos,
                    }
                    self.current_handler_id = None

        # Check Active Pass Resolution
        if active_shots:
            self.active_pass = None
            self.offensive_paint_time.clear()

        if self.active_pass is not None:
            passer_id = self.active_pass["passer_id"]
            passer_team = self.active_pass["passer_team"]
            pass_duration = frame_idx - self.active_pass["start_frame"]

            # A. Check Out of Bounds (Fehlpass ins Aus)
            # A pass goes out of bounds only if the ball reaches near the camera boundary
            # or crosses beyond the physical court lines on the floor.
            # NOTE: self.geometry.get_zone() checks offensive shooting zones (paint, mid, 3pt)
            # and MUST NOT be used for court boundaries, as in-bounds passes in the backcourt/wings
            # are not inside shooting zones.
            is_out_of_bounds = False
            margin = 30
            # A pass goes out of bounds when the ball physically crosses near the camera frame boundaries
            if ball_pos[0] < margin or ball_pos[0] > (self.frame_w - margin) or ball_pos[1] < margin or ball_pos[1] > (self.frame_h - margin):
                is_out_of_bounds = True

            if is_out_of_bounds and pass_duration >= 6 and self.cooldowns.get("turnover", 0) == 0:
                evt = {
                    "type": "FEHLPASS_AUS",
                    "player_id": passer_id,
                    "team": passer_team,
                    "frame": frame_idx,
                    "description": f"Fehlpass ins Aus (Bad Pass Out of Bounds) by Player {passer_id} ({passer_team})"
                }
                events.append(evt)
                self.logged_events.append(evt)
                if self.player_registry:
                    self.player_registry.record_action(
                        passer_id, "TURNOVER",
                        details={"reason": "bad_pass_out_of_bounds"}
                    )
                print(f"  ⚠️ [EVENT] {evt['description']}")
                self.cooldowns["turnover"] = int(self.fps * 2.5)
                self.active_pass = None

            # B. Check Catch by Teammate or Opponent (Steal / Interception)
            elif is_near_player and closest_pid is not None and closest_pid != passer_id:
                receiver_team = player_teams.get(str(closest_pid), "unknown")
                if passer_team == "home":
                    is_opponent = (receiver_team == "away")
                    stealer_team = "away"
                elif passer_team == "away":
                    is_opponent = (receiver_team in ("home", "unknown"))
                    stealer_team = "home"
                else:
                    is_opponent = (receiver_team != "unknown" and receiver_team != passer_team)
                    stealer_team = receiver_team

                if is_opponent:
                    if self.defender_intercept_candidate == closest_pid:
                        self.defender_intercept_count += 1
                    else:
                        self.defender_intercept_candidate = closest_pid
                        self.defender_intercept_count = 1

                    if (min_p_dist <= 45.0 or self.defender_intercept_count >= 2) and self.cooldowns.get("turnover", 0) == 0:
                        evt = {
                            "type": "FEHLPASS_STEAL",
                            "passer_id": passer_id,
                            "passer_team": passer_team,
                            "stealer_id": closest_pid,
                            "stealer_team": stealer_team,
                            "frame": frame_idx,
                            "description": f"Fehlpass abgefangen! Steal by Player {closest_pid} ({stealer_team}) from Player {passer_id} ({passer_team})"
                        }
                        events.append(evt)
                        self.logged_events.append(evt)

                        if self.player_registry:
                            self.player_registry.record_action(
                                passer_id, "TURNOVER",
                                details={"reason": "bad_pass_intercepted", "interceptor": closest_pid}
                            )
                            self.player_registry.record_action(
                                closest_pid, "STEAL",
                                details={"stolen_from": passer_id}
                            )
                        print(f"  🖐️ [EVENT] {evt['description']}")
                        self.cooldowns["turnover"] = int(self.fps * 2.5)
                        self.team_frontcourt_established = {"home": False, "away": False}
                        self.offensive_paint_time.clear()
                        self.active_pass = None
                        self.defender_intercept_candidate = None
                        self.defender_intercept_count = 0
                else:
                    # Teammate catch (pass completed)
                    self.last_completed_pass = {
                        "passer_id": passer_id,
                        "passer_team": passer_team,
                        "receiver_id": closest_pid,
                        "frame": frame_idx,
                    }
                    self.active_pass = None
                    self.defender_intercept_candidate = None
                    self.defender_intercept_count = 0
            # Timeout expired pass (> 40 frames / ~1.3s)
            elif pass_duration > 40:
                self.active_pass = None
                self.defender_intercept_candidate = None
                self.defender_intercept_count = 0

        # ── 4. Shot Block Detection ────────────────────────────────────────
        if active_shots:
            for shot in active_shots:
                if shot.get("status") == "rising":
                    shooter_id = shot.get("shooter_id")
                    shooter_team = player_teams.get(str(shooter_id), "home")

                    # Check if an opposing defender is in blocking position
                    for pid, cap in player_capsules.items():
                        if pid == shooter_id:
                            continue
                        def_team = player_teams.get(str(pid), "unknown")
                        if def_team != "unknown" and def_team == shooter_team:
                            continue

                        # Defender distance to ball in flight
                        d_ball = ball.center.distance_to(cap.center)
                        d_head = ball.center.distance_to(cap.head)
                        if min(d_ball, d_head) < 70.0:
                            # Defender contesting ball closely during rising shot
                            # Check if ball abruptly drops or reverses trajectory
                            if len(balls) > 1 or (self.last_ball_y and (ball.center.y - self.last_ball_y) > 8.0):
                                shot["status"] = "blocked"
                                evt = {
                                    "type": "BLOCK",
                                    "blocker_id": pid,
                                    "blocker_team": def_team,
                                    "shooter_id": shooter_id,
                                    "frame": frame_idx,
                                    "description": f"Wurfblock (Shot Block) by Defender {pid} ({def_team}) against Shooter {shooter_id}"
                                }
                                events.append(evt)
                                self.logged_events.append(evt)
                                if self.player_registry:
                                    self.player_registry.record_action(pid, "BLOCK", details={"blocked_shooter": shooter_id})
                                print(f"  🛑 [EVENT] {evt['description']}")
                                break

        return events

    def check_assist_on_made_shot(self, shooter_id: Optional[int], frame_idx: int, shooter_team: Optional[str] = None, shot_start_frame: Optional[int] = None) -> Optional[dict]:
        """
        When a shot is made, checks if the shooter received a pass leading to the basket.
        If pass occurred within 4.5s of shot release, records an ASSIST for the passer.
        """
        if self.last_completed_pass is not None:
            passer_id = self.last_completed_pass["passer_id"]
            receiver_id = self.last_completed_pass["receiver_id"]
            passer_team = self.last_completed_pass["passer_team"]
            pass_frame = self.last_completed_pass["frame"]

            ref_frame = shot_start_frame if shot_start_frame is not None else frame_idx
            is_valid_recipient = (
                receiver_id == shooter_id
                or shooter_id is None
                or (shooter_team is not None and shooter_team == passer_team)
            )

            if is_valid_recipient and (ref_frame - pass_frame) <= int(self.fps * 4.5):
                evt = {
                    "type": "ASSIST",
                    "passer_id": passer_id,
                    "team": passer_team,
                    "shooter_id": shooter_id,
                    "frame": frame_idx,
                    "description": f"Vorlage (Assist) by Player {passer_id} ({passer_team}) to Shooter {shooter_id}"
                }
                self.logged_events.append(evt)
                if self.player_registry:
                    self.player_registry.record_action(passer_id, "ASSIST", details={"recipient": shooter_id})
                print(f"  🎯 [EVENT] {evt['description']}")
                self.last_completed_pass = None
                return evt

        return None
