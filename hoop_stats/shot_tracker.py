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
    def __init__(self, frame_w, frame_h, court_geometry=None, cooldown_frames=45,
                 halftime_frame=None, fps=30.0, quarter_length_min=10.0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.geometry = court_geometry
        self.cooldown_frames = cooldown_frames

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

    # ── Rim management ────────────────────────────────────────────────────

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

    def update(self, balls, player_capsules, frame_idx):
        """
        balls: list of Ball primitives
        player_capsules: dict of {player_id: PlayerCapsule}
        """
        events = []

        # Check for halftime
        self._check_halftime(frame_idx)

        # Decrement cooldowns
        for zi in [0, 1]:
            if self.cooldown[zi] > 0:
                self.cooldown[zi] -= 1

        # Track primary ball
        if balls:
            ball = balls[0]
            self.ball_history.append(ball)
            self.ball_frame_history.append(frame_idx)
            if len(self.ball_history) > 30:
                self.ball_history.pop(0)
                self.ball_frame_history.pop(0)

            # 1. Detect Shot Release
            self._detect_release(player_capsules, frame_idx)

            # 2. Track Active Shots & Detect Outcomes
            events.extend(self._evaluate_active_shots(ball, frame_idx))

        return events

    def _detect_release(self, player_capsules, frame_idx):
        """Looks for a strong upward movement pattern indicating a shot release."""
        if len(self.ball_history) < 5:
            return

        if any(s["status"] == "rising" for s in self.active_shots):
            return

        recent_balls = self.ball_history[-5:]
        recent_frames = self.ball_frame_history[-5:]
        y_vals = [b.center.y for b in recent_balls]
        ydiff = y_vals[0] - y_vals[-1]

        if ydiff > 15 and all(y_vals[i] >= y_vals[i+1] for i in range(4)):
            start_ball = recent_balls[0]
            start_frame = recent_frames[0]
            shooter_id = self._get_closest_player(start_ball, player_capsules)

            shot = {
                "id": len(self.completed_shots) + len(self.active_shots),
                "shooter_id": shooter_id,
                "start_frame": start_frame,
                "start_px": start_ball.center.as_tuple(),
                "apex_y": start_ball.center.y,
                "status": "rising",
                "target_hoop": 0 if start_ball.center.x > self.frame_w / 2 else 1
            }

            if shooter_id is not None and self.geometry and shooter_id in player_capsules:
                cap = player_capsules[shooter_id]
                if cap.meter_pos:
                    shot["zone"] = self.geometry.get_zone(cap.meter_pos.x, cap.meter_pos.y)
                    shot["coord_m"] = cap.meter_pos.as_tuple()
                else:
                    shot["zone"] = "Unknown"
                    shot["coord_m"] = None
            else:
                shot["zone"] = "Unknown"
                shot["coord_m"] = None

            self.active_shots.append(shot)

    def _evaluate_active_shots(self, ball: Ball, frame_idx):
        events = []
        for shot in self.active_shots[:]:
            zi = shot["target_hoop"]
            rim = self.rims[zi]

            if ball.center.y < shot["apex_y"]:
                shot["apex_y"] = ball.center.y

            if ball.center.y > shot["apex_y"] + 10 and shot["status"] == "rising":
                shot["status"] = "falling"

            if shot["status"] == "falling":
                if rim_contains_ball(rim, ball) and self.cooldown[zi] == 0:
                    # MADE SHOT
                    shot["status"] = "made"
                    team = self._basket_to_team(zi)
                    shot["team"] = team
                    self.active_shots.remove(shot)
                    self.completed_shots.append(shot)
                    self.scores[zi] += 2
                    self.team_scores[team] += 2
                    self.cooldown[zi] = self.cooldown_frames
                    events.append((zi, self.scores[zi], team, self.team_scores[team]))
                    print(f"  🏀 MADE SHOT by Player {shot['shooter_id']} "
                          f"({team}) at zone {shot['zone']}!")

                elif ball.center.y > self._rim_bottom_y[zi] and frame_idx - shot["start_frame"] > 15:
                    shot["status"] = "missed"
                    team = self._basket_to_team(zi)
                    shot["team"] = team
                    self.active_shots.remove(shot)
                    self.completed_shots.append(shot)
                    print(f"  ❌ MISSED SHOT by Player {shot['shooter_id']} "
                          f"({team}) from {shot['zone']}.")

        return events

    def _get_closest_player(self, ball: Ball, player_capsules):
        closest_id = None
        min_dist = float('inf')
        for pid, cap in player_capsules.items():
            dist = ball.center.distance_to(cap.center)
            if dist < min_dist:
                min_dist = dist
                closest_id = pid

        if min_dist < 200:
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
