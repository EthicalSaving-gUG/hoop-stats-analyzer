import numpy as np
from .primitives import Ball, Rim, PlayerCapsule, Point2D, rim_contains_ball


class ShotTracker:
    """
    Detects basketball jump shots, arcs, and makes/misses.

    Strategy:
    1. Maintain history of ball positions.
    2. Detect an arc: ball travels upward (decreasing Y), reaches an apex, then falls.
    3. At the start of the upward motion, attach the shot to the closest player.
    4. If the falling ball intersects the hoop Rim circle -> MADE shot.
    5. If the falling ball falls below the rim zone without intersection -> MISSED shot.

    Halftime side-swap:
    Teams switch which basket they attack at halftime. The tracker
    maintains a team→basket mapping that can be flipped via swap_sides().
    Team scores are accumulated correctly across the whole game.
    """
    def __init__(self, frame_w, frame_h, court_geometry=None, cooldown_frames=45, player_registry=None,
                 halftime_frame=None, fps=30.0, quarter_length_min=10.0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.geometry = court_geometry
        self.cooldown_frames = cooldown_frames
        self.player_registry = player_registry

        # Rim circles — initialized with fallback positions,
        # updated dynamically by RimDetector when available
        rim_radius = frame_w * 0.08
        self.rims = [
            Rim(center=Point2D(frame_w * 0.115, frame_h * 0.40), radius=rim_radius),  # Left  (index 0)
            Rim(center=Point2D(frame_w * 0.885, frame_h * 0.40), radius=rim_radius),  # Right (index 1)
        ]
        self._rim_bottom_y = [
            self.rims[0].center.y + rim_radius,
            self.rims[1].center.y + rim_radius,
        ]
        self._rims_detected = False

        # ── Team-to-basket mapping ───────────────────────────────────────
        # Before halftime: home attacks right basket (index 1),
        #                  away attacks left basket  (index 0).
        # After swap_sides(): these flip.
        self._home_basket = 1  # basket index that home team scores into
        self._away_basket = 0  # basket index that away team scores into
        self._sides_swapped = False

        # Halftime auto-detection
        self._halftime_frame = halftime_frame  # user-provided frame, or None
        self._fps = fps
        self._quarter_length_min = quarter_length_min
        self._halftime_triggered = False

        # Ball trajectory tracking
        self.ball_history = []
        self.ball_frame_history = []

        self.active_shots = []
        self.completed_shots = []

        # Per-basket raw scores (for backward compat)
        self.scores = [0, 0]
        # Per-team cumulative scores (correct across halftime)
        self.team_scores = {"home": 0, "away": 0}

        self.cooldown = {0: 0, 1: 0}
        self.rim_focus_radius = frame_w * 0.16  # approx 200px Korb-Fokus
        self.is_free_throw_mode = False
        self.free_throws_made = {"home": 0, "away": 0}
        self.free_throws_attempted = {"home": 0, "away": 0}
        self.player_history = {}

    # ── Rim management ────────────────────────────────────────────────────

    def get_rim_focus_boxes(self, focus_radius=None):
        """Returns high-resolution focus boxes around the rims for Korb-Fokus-Tracking."""
        r = focus_radius or self.rim_focus_radius
        boxes = []
        for rim in self.rims:
            rx, ry = rim.center.x, rim.center.y
            x1 = max(0, int(rx - r))
            y1 = max(0, int(ry - r))
            x2 = min(self.frame_w, int(rx + r))
            y2 = min(self.frame_h, int(ry + r))
            boxes.append((x1, y1, x2, y2))
        return boxes

    def detect_free_throw_formation(self, player_capsules, frame_idx) -> bool:
        """Detects dead-ball free-throw formation at the free-throw line."""
        if not player_capsules:
            return False
        for zi in [0, 1]:
            rim = self.rims[zi]
            shooters = []
            lane_top = []
            lane_bottom = []
            for pid, cap in player_capsules.items():
                dist_x = abs(cap.center.x - rim.center.x)
                dy = cap.center.y - rim.center.y
                # Shooter is in front of the rim along center corridor
                if 140 < dist_x < 280 and (abs(dy) < 60 or 60 <= dy <= 260):
                    shooters.append(pid)
                # Rebounders along lane lines (above and below the rim)
                elif dist_x < 340:
                    if dy < -30:
                        lane_top.append(pid)
                    elif dy > 50:
                        lane_bottom.append(pid)

            if len(shooters) >= 1 and (len(lane_top) + len(lane_bottom) >= 2):
                self.is_free_throw_mode = True
                return True
        self.is_free_throw_mode = False
        return False

    def update_rims(self, detected_rims: list):
        """Update rim positions from RimDetector output."""
        if not detected_rims:
            return
        mid_x = self.frame_w / 2
        for rim in detected_rims:
            if rim.center.x < mid_x:
                self.rims[0] = rim
                self._rim_bottom_y[0] = rim.center.y + rim.radius
            else:
                self.rims[1] = rim
                self._rim_bottom_y[1] = rim.center.y + rim.radius
        self._rims_detected = True

    # ── Side management ──────────────────────────────────────────────────

    def swap_sides(self):
        """Swap which basket each team attacks (called at halftime)."""
        self._home_basket, self._away_basket = self._away_basket, self._home_basket
        self._sides_swapped = not self._sides_swapped
        print("  🔄 HALFTIME: Teams swap sides!")

    def _basket_to_team(self, basket_idx: int) -> str:
        """Return 'home' or 'away' based on which team scores into this basket."""
        if basket_idx == self._home_basket:
            return "home"
        return "away"

    def _check_halftime(self, frame_idx):
        """Auto-trigger halftime swap if we've reached the halftime frame."""
        if self._halftime_triggered:
            return

        halftime_frame = self._halftime_frame
        if halftime_frame is None:
            # Estimate: halftime = 2 quarters worth of frames
            halftime_frame = int(self._fps * self._quarter_length_min * 60 * 2)

        if frame_idx >= halftime_frame:
            self.swap_sides()
            self._halftime_triggered = True
            self.completed_shots.append({
                "id": -1,
                "event": "HALFTIME",
                "frame": frame_idx,
                "status": "halftime",
                "shooter_id": None,
                "zone": None,
                "coord_m": None,
            })

    # ── Core update ──────────────────────────────────────────────────────

    def update(self, balls, player_capsules, frame_idx, player_teams=None):
        """
        balls: list of Ball primitives
        player_capsules: dict of {player_id: PlayerCapsule}
        player_teams: dict of {player_id_str: team_name}
        """
        events = []

        # Check for halftime
        self._check_halftime(frame_idx)

        # Decrement cooldowns
        for zi in [0, 1]:
            if self.cooldown[zi] > 0:
                self.cooldown[zi] -= 1

        if player_capsules:
            self.player_history[frame_idx] = player_capsules
            if len(self.player_history) > 60:
                del self.player_history[next(iter(self.player_history))]

        # Track primary ball with motion continuity filtering (rejects bench/teleporting balls)
        best_ball = None
        if balls:
            if self.ball_history:
                last_ball = self.ball_history[-1]
                # Filter out impossible jumps (> 280px between sampled frames)
                valid_balls = [b for b in balls if b.center.distance_to(last_ball.center) < 280]
                if valid_balls:
                    best_ball = min(valid_balls, key=lambda b: b.center.distance_to(last_ball.center))
                else:
                    # Pick ball closest to active players on court
                    if player_capsules:
                        best_ball = min(balls, key=lambda b: min(b.center.distance_to(p.center) for p in player_capsules.values()))
                    else:
                        best_ball = balls[0]
            else:
                best_ball = balls[0]

        if best_ball:
            ball = best_ball
            self.ball_history.append(ball)
            self.ball_frame_history.append(frame_idx)
            if len(self.ball_history) > 30:
                self.ball_history.pop(0)
                self.ball_frame_history.pop(0)

            # 1. Detect Shot Release
            self._detect_release(player_capsules, frame_idx, player_teams)

            # 2. Track Active Shots & Detect Outcomes
            events.extend(self._evaluate_active_shots(ball, frame_idx))

        return events

    def _detect_release(self, player_capsules, frame_idx, player_teams=None):
        """Looks for a strong upward movement pattern indicating a shot release."""
        if len(self.ball_history) < 5:
            return

        if any(s["status"] == "rising" for s in self.active_shots):
            return

        recent_balls = self.ball_history[-5:]
        recent_frames = self.ball_frame_history[-5:]
        y_vals = [b.center.y for b in recent_balls]
        ydiff = y_vals[0] - y_vals[-1]
        start_ball = recent_balls[0]

        # Check proximity to nearest rim
        d0 = start_ball.center.distance_to(self.rims[0].center)
        d1 = start_ball.center.distance_to(self.rims[1].center)
        min_rim_dist = min(d0, d1)
        is_near_rim_paint = min_rim_dist < (self.rims[0].radius * 2.2)

        standard_shot = (ydiff > 16) and all(y_vals[i] >= y_vals[i+1] for i in range(4))
        layup_shot = is_near_rim_paint and (ydiff > 13) and (sum(1 for i in range(4) if y_vals[i] >= y_vals[i+1]) >= 3)

        if standard_shot or layup_shot:
            start_ball = recent_balls[0]
            start_frame = recent_frames[0]
            capsules_at_release = self.player_history.get(start_frame) or player_capsules
            shooter_id = self._get_closest_player(start_ball, capsules_at_release) or self._get_closest_player(start_ball, player_capsules)

            # Calculate defender distance at release moment
            defender_dist = 999.0
            if shooter_id is not None and shooter_id in capsules_at_release:
                shooter_cap = capsules_at_release[shooter_id]
                shooter_pos = shooter_cap.meter_pos
                if shooter_pos:
                    shooter_team = player_teams.get(str(shooter_id), "Unknown") if player_teams else "Unknown"
                    min_dist = float('inf')
                    for pid, cap in player_capsules.items():
                        if pid == shooter_id:
                            continue
                        if cap.meter_pos:
                            other_team = player_teams.get(str(pid), "Unknown") if player_teams else "Unknown"
                            # If team assignments are known, only treat opposing team members as defenders
                            if player_teams and shooter_team != "Unknown" and other_team != "Unknown":
                                if other_team == shooter_team:
                                    continue
                            d = shooter_pos.distance_to(cap.meter_pos)
                            if d < min_dist:
                                min_dist = d
                    if min_dist != float('inf'):
                        defender_dist = min_dist

            # Target hoop selection:
            # Shots taken in offensive court zones target the near rim
            if start_ball.center.x > 800:
                target_hoop = 1  # Right rim
            elif start_ball.center.x < 480:
                target_hoop = 0  # Left rim
            else:
                target_hoop = 0 if start_ball.center.x > self.frame_w / 2 else 1

            # Free throw verification: must be in FT mode and at FT distance (150-260px)
            is_ft_shot = False
            if self.is_free_throw_mode:
                target_rim = self.rims[target_hoop]
                start_dist = start_ball.center.distance_to(target_rim.center)
                if 150 < start_dist < 260:
                    is_ft_shot = True
            shot = {
                "id": len(self.completed_shots) + len(self.active_shots),
                "shooter_id": shooter_id,
                "start_frame": start_frame,
                "start_px": start_ball.center.as_tuple(),
                "apex_y": start_ball.center.y,
                "status": "rising",
                "target_hoop": target_hoop,
                "defender_dist": defender_dist,
                "is_free_throw": is_ft_shot
            }

            # Determine Shot Zone (3-Point vs 2-Point) via Takeoff Floor Contact (FIBA Art. 16.2.3)
            # A field goal's value is determined exclusively by where the shooter touched the floor prior to takeoff!
            shooter_takeoff_cap = capsules_at_release.get(shooter_id) if shooter_id is not None else None
            if shooter_takeoff_cap is None and shooter_id is not None:
                shooter_takeoff_cap = player_capsules.get(shooter_id)

            zone_determined = None
            if shooter_takeoff_cap and self.geometry:
                if shooter_takeoff_cap.meter_pos:
                    mx, my = shooter_takeoff_cap.meter_pos.x, shooter_takeoff_cap.meter_pos.y
                else:
                    foot_pt = shooter_takeoff_cap.ground_contact
                    mx, my = self.geometry.pixel_to_meter(foot_pt.x, foot_pt.y)

                shot["coord_m"] = (float(mx), float(my))
                shot["takeoff_ground_m"] = (float(mx), float(my))

                if hasattr(self.geometry, "evaluate_player_floor_status"):
                    floor_status = self.geometry.evaluate_player_floor_status(shooter_takeoff_cap, attacking_basket=shot["target_hoop"])
                    shot["floor_status"] = floor_status
                    if floor_status.get("is_three"):
                        zone_determined = "above_the_break_three"
                    elif floor_status.get("in_paint"):
                        zone_determined = "paint"
                    else:
                        zone_determined = "mid_range"
                elif hasattr(self.geometry, "get_zone"):
                    z = self.geometry.get_zone(mx, my)
                    if z and z not in ("out_of_bounds", "Unknown"):
                        zone_determined = z
            else:
                shot["coord_m"] = None

            if zone_determined:
                shot["zone"] = zone_determined
            elif self.geometry is not None:
                target_rim = self.rims[shot["target_hoop"]] if (self.rims and len(self.rims) > shot["target_hoop"]) else None
                if target_rim:
                    s_cap = shooter_takeoff_cap
                    pos_x = s_cap.ground_contact.x if s_cap else start_ball.center.x
                    pos_y = s_cap.ground_contact.y if s_cap else start_ball.center.y
                    dx = abs(pos_x - target_rim.center.x)
                    dy = abs(pos_y - target_rim.center.y)

                    # FIBA court calibrated dimensions (39.1 px/m):
                    # 3-point line apex: 6.75m => dx >= 260px
                    # Corner three: dx >= 130px and dy >= 220px
                    is_corner_3 = (dx >= 130.0 and dy >= 220.0)
                    is_arc_3 = (dx >= 260.0)

                    if is_arc_3 or is_corner_3:
                        shot["zone"] = "above_the_break_three"
                    elif dx < 110.0 and dy < 160.0:
                        shot["zone"] = "paint"
                    else:
                        shot["zone"] = "mid_range"
                else:
                    shot["zone"] = "paint_or_midrange"
            else:
                shot["zone"] = "paint_or_midrange"

            # Calculate Expected Points (xPTS) and Shot Quality
            is_three = shot.get("zone") and "three" in str(shot.get("zone")).lower()
            if defender_dist > 2.5:
                shot_quality = 0.38 if is_three else 0.65
            elif defender_dist < 1.2:
                shot_quality = 0.20 if is_three else 0.40
            else:
                # Interpolate between 1.2m and 2.5m
                t = (defender_dist - 1.2) / (2.5 - 1.2)
                q_min = 0.20 if is_three else 0.40
                q_max = 0.38 if is_three else 0.65
                shot_quality = q_min + t * (q_max - q_min)

            shot["shot_quality"] = shot_quality
            shot["expected_points"] = (3.0 if is_three else 2.0) * shot_quality

            self.active_shots.append(shot)

    def _trajectory_crosses_rim(self, rim_idx: int, prev_ball: Ball, curr_ball: Ball) -> bool:
        """Checks whether the ball trajectory segment passed through the rim circle."""
        if prev_ball is None or curr_ball is None:
            return False
        rim = self.rims[rim_idx]
        y1, y2 = prev_ball.center.y, curr_ball.center.y
        # Ball must be falling downwards across the rim plane
        if y1 <= rim.center.y <= y2 and (y2 - y1) > 0:
            t = (rim.center.y - y1) / (y2 - y1)
            x_cross = prev_ball.center.x + t * (curr_ball.center.x - prev_ball.center.x)
            if abs(x_cross - rim.center.x) <= (rim.radius * 1.50):
                return True
        # Robust cylinder check: ball falling through or directly inside the rim net area
        if abs(curr_ball.center.x - rim.center.x) <= (rim.radius * 1.40) and (rim.center.y - 10) <= curr_ball.center.y <= (rim.center.y + 35) and (y2 - y1) >= 0:
            return True
        return False

    def _evaluate_active_shots(self, ball: Ball, frame_idx):
        events = []
        prev_ball = self.ball_history[-2] if len(self.ball_history) >= 2 else None

        for shot in self.active_shots[:]:
            zi = shot["target_hoop"]
            rim = self.rims[zi]

            if ball.center.y < shot["apex_y"]:
                shot["apex_y"] = ball.center.y

            if ball.center.y > shot["apex_y"] + 10 and shot["status"] == "rising":
                shot["status"] = "falling"

            is_near_rim = abs(ball.center.x - rim.center.x) <= (rim.radius * 1.4) and abs(ball.center.y - rim.center.y) <= (rim.radius + 25)
            is_falling_or_rim = (shot["status"] == "falling") or (is_near_rim and ball.center.y >= shot["apex_y"])
            if is_falling_or_rim:
                is_rim_make = rim_contains_ball(rim, ball) or self._trajectory_crosses_rim(zi, prev_ball, ball)
                if is_rim_make and self.cooldown[zi] == 0:
                    # MADE SHOT
                    shot["status"] = "made"
                    team = self._basket_to_team(zi)
                    shot["team"] = team
                    self.active_shots.remove(shot)
                    self.completed_shots.append(shot)
                    
                    is_three = shot.get("zone") and "three" in str(shot.get("zone")).lower()
                    is_ft = shot.get("is_free_throw", False) and not is_three
                    if is_ft:
                        points = 1
                        self.free_throws_made[team] += 1
                        self.free_throws_attempted[team] += 1
                        evt_type = "FREE_THROW_MAKE"
                    elif is_three:
                        points = 3
                        evt_type = "SHOT_MAKE"
                    else:
                        points = 2
                        evt_type = "SHOT_MAKE"

                    self.scores[zi] += points
                    self.team_scores[team] += points
                    self.cooldown[zi] = self.cooldown_frames
                    self.is_free_throw_mode = False
                    if self.player_registry and shot["shooter_id"] is not None:
                        self.player_registry.record_action(shot["shooter_id"], evt_type, points=points, details={"zone": shot["zone"]})
                    
                    events.append({
                        "type": evt_type,
                        "shooter_id": shot["shooter_id"],
                        "team": team,
                        "zone": "free_throw" if is_ft else shot["zone"],
                        "coord_m": shot["coord_m"],
                        "points": points,
                        "shot_quality": round(shot.get("shot_quality", 0.5), 2),
                        "expected_points": round(shot.get("expected_points", 1.0), 2),
                        "defender_dist": round(shot.get("defender_dist", 999.0), 2)
                    })
                    print(f"  🏀 MADE SHOT by Player {shot['shooter_id']} "
                          f"({team}) at zone {shot['zone']}!")

                elif ball.center.y > self._rim_bottom_y[zi] and frame_idx - shot["start_frame"] > 15:
                    shot["status"] = "missed"
                    team = self._basket_to_team(zi)
                    shot["team"] = team
                    self.active_shots.remove(shot)
                    self.completed_shots.append(shot)
                    
                    is_three = shot.get("zone") and "three" in str(shot.get("zone")).lower()
                    is_ft = shot.get("is_free_throw", False) and not is_three
                    if is_ft:
                        self.free_throws_attempted[team] += 1
                        evt_type = "FREE_THROW_MISS"
                    else:
                        evt_type = "SHOT_MISS"
                    self.is_free_throw_mode = False
                    if self.player_registry and shot["shooter_id"] is not None:
                        self.player_registry.record_action(shot["shooter_id"], evt_type, points=0, details={"zone": shot["zone"]})

                    events.append({
                        "type": evt_type,
                        "shooter_id": shot["shooter_id"],
                        "team": team,
                        "zone": "free_throw" if is_ft else shot["zone"],
                        "coord_m": shot["coord_m"],
                        "points": 0,
                        "shot_quality": round(shot.get("shot_quality", 0.5), 2),
                        "expected_points": round(shot.get("expected_points", 1.0), 2),
                        "defender_dist": round(shot.get("defender_dist", 999.0), 2)
                    })
                    print(f"  ❌ MISSED SHOT by Player {shot['shooter_id']} "
                          f"({team}) from {shot['zone']}.")

        return events

    def _get_closest_player(self, ball: Ball, player_capsules):
        closest_id = None
        min_dist = float('inf')
        for pid, cap in player_capsules.items():
            d_head = ball.center.distance_to(cap.head)
            d_center = ball.center.distance_to(cap.center)
            d_foot = ball.center.distance_to(cap.foot)
            dist = min(d_head, d_center, d_foot)
            if dist < min_dist:
                min_dist = dist
                closest_id = pid

        if min_dist < 450:
            return closest_id
        return None

    def get_total_scores(self):
        return {
            "home_pts": self.team_scores["home"],
            "away_pts": self.team_scores["away"],
            "left_basket_pts": self.scores[0],
            "right_basket_pts": self.scores[1],
            "total_baskets": (self.team_scores["home"] + self.team_scores["away"]) // 2,
            "sides_swapped": self._sides_swapped,
        }
