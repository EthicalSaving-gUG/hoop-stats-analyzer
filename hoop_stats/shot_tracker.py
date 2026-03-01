import numpy as np

class ShotTracker:
    """
    Detects basketball jump shots, arcs, and makes/misses.
    
    Strategy:
    1. Maintain history of ball positions.
    2. Detect an arc: ball travels upward (decreasing Y), reaches an apex, then falls.
    3. At the start of the upward motion, attach the shot to the closest player.
    4. If the falling ball intersects the hoop zone -> MADE shot.
    5. If the falling ball falls below the hoop zone without intersection -> MISSED shot.
    """
    def __init__(self, frame_w, frame_h, court_geometry=None, cooldown_frames=45):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.geometry = court_geometry
        self.cooldown_frames = cooldown_frames

        # (x1, y1, x2, y2)
        self.hoop_zones = [
            (int(frame_w * 0.03), int(frame_h * 0.25), int(frame_w * 0.20), int(frame_h * 0.55)), # Left
            (int(frame_w * 0.80), int(frame_h * 0.25), int(frame_w * 0.97), int(frame_h * 0.55)), # Right
        ]

        # Ball trajectory tracking: {ball_tracker_id: [(x, y, frame_idx), ...]}
        # Since balls are transient, we just track the active ball trajectory list
        self.ball_history = []  # Keep last 30 frames of the primary ball
        
        self.active_shots = []  # List of dicts representing in-air shots
        self.completed_shots = [] # List of outcome dicts
        
        self.scores = [0, 0]
        self.cooldown = {0: 0, 1: 0}

    def update(self, ball_centers, player_positions, frame_idx):
        """
        ball_centers: list of (px, py)
        player_positions: dict of {player_id: (px, py, mx, my)}
        """
        events = []
        
        # Decrement cooldowns
        for zi in [0, 1]:
            if self.cooldown[zi] > 0:
                self.cooldown[zi] -= 1

        # Track primary ball
        if ball_centers:
            # Assume first ball is primary for now
            bx, by = ball_centers[0]
            self.ball_history.append((bx, by, frame_idx))
            if len(self.ball_history) > 30:
                self.ball_history.pop(0)
            
            # 1. Detect Shot Release
            self._detect_release(player_positions, frame_idx)
            
            # 2. Track Active Shots & Detect Outcomes
            events.extend(self._evaluate_active_shots(bx, by, frame_idx))
            
        return events

    def _detect_release(self, player_positions, frame_idx):
        """Looks for a strong upward movement pattern indicating a shot release."""
        if len(self.ball_history) < 5:
            return
            
        # If we already have a shot going up, don't trigger another one immediately
        if any(s["status"] == "rising" for s in self.active_shots):
            return

        # Look at last 5 frames. If highly monotonic upward (decreasing Y), trigger release.
        recent = self.ball_history[-5:]
        y_vals = [p[1] for p in recent]
        ydiff = y_vals[0] - y_vals[-1] # positive means it went up
        
        # Arbitrary threshold: moved up by at least 15 pixels over 5 frames
        if ydiff > 15 and all(y_vals[i] >= y_vals[i+1] for i in range(4)):
            # Finds closest player (who took the shot)
            start_x, start_y, start_f = recent[0]
            shooter_id = self._get_closest_player(start_x, start_y, player_positions)
            
            shot = {
                "id": len(self.completed_shots) + len(self.active_shots),
                "shooter_id": shooter_id,
                "start_frame": start_f,
                "start_px": (start_x, start_y),
                "apex_y": start_y,
                "status": "rising", # rising -> falling -> made/missed
                "target_hoop": 0 if start_x > self.frame_w / 2 else 1 # Shooting towards left or right?
            }
            
            # Calculate real-world zone if geometry available
            if shooter_id is not None and self.geometry:
                _, _, mx, my = player_positions[shooter_id]
                shot["zone"] = self.geometry.get_zone(mx, my)
                shot["coord_m"] = (float(mx), float(my))
            else:
                shot["zone"] = "Unknown"
                shot["coord_m"] = None
                
            self.active_shots.append(shot)

    def _evaluate_active_shots(self, bx, by, frame_idx):
        events = []
        for shot in self.active_shots[:]:
            zi = shot["target_hoop"]
            hx1, hy1, hx2, hy2 = self.hoop_zones[zi]
            
            # Update Apex
            if by < shot["apex_y"]:
                shot["apex_y"] = by
                
            # Transition to falling
            if by > shot["apex_y"] + 10 and shot["status"] == "rising":
                shot["status"] = "falling"
                
            # Evaluate hit or miss
            if shot["status"] == "falling":
                in_zone_x = hx1 <= bx <= hx2
                in_zone_y = hy1 <= by <= hy2
                
                if in_zone_x and in_zone_y and self.cooldown[zi] == 0:
                    # MADE SHOT
                    shot["status"] = "made"
                    self.active_shots.remove(shot)
                    self.completed_shots.append(shot)
                    self.scores[zi] += 2
                    self.cooldown[zi] = self.cooldown_frames
                    events.append((zi, self.scores[zi]))
                    print(f"  🏀 MADE SHOT by Player {shot['shooter_id']} at zone {shot['zone']}!")
                    
                elif by > hy2 and frame_idx - shot["start_frame"] > 15:
                    # MISSED SHOT (fell below hoop height)
                    shot["status"] = "missed"
                    self.active_shots.remove(shot)
                    self.completed_shots.append(shot)
                    print(f"  ❌ MISSED SHOT by Player {shot['shooter_id']} from {shot['zone']}.")

        return events

    def _get_closest_player(self, bx, by, player_positions):
        closest_id = None
        min_dist = float('inf')
        for pid, (px, py, mx, my) in player_positions.items():
            dist = (px - bx)**2 + (py - by)**2
            if dist < min_dist:
                min_dist = dist
                closest_id = pid
        
        # Only attach if reasonably close (e.g., within 200 pixels)
        if min_dist < 40000:
            return closest_id
        return None

    def get_total_scores(self):
        return {
            "left_basket_pts": self.scores[0],
            "right_basket_pts": self.scores[1],
            "total_baskets": sum(self.scores) // 2
        }
