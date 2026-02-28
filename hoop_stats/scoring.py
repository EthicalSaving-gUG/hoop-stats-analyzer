import numpy as np
import cv2

class ScoringDetector:
    """
    Detects when the basketball passes through a basket.

    Strategy:
    - Detect the ball (COCO class 32 = sports ball) each frame.
    - Define two 'hoop zones' (one per end of the court) as rectangular
      regions of interest in the frame.
    - A score is counted when the ball's centre moves FROM above the hoop zone
      TO inside / below the hoop zone within a short window (the ball 'drops').
    - A cooldown prevents double-counting the same basket.
    """

    def __init__(self, frame_w, frame_h, cooldown_frames=45):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cooldown_frames = cooldown_frames

        # ── Hoop zones (pixel rectangles) ──────────────────────────────────
        # These are reasonable defaults for a centre-mounted single camera.
        # Both hoops are at ~y=30-50% height; left hoop x≈5-20%, right x≈80-95%.
        # Adjust these if the camera angle is very different.
        self.hoop_zones = [
            # (x1, y1, x2, y2)  – left basket
            (int(frame_w * 0.03), int(frame_h * 0.25),
             int(frame_w * 0.20), int(frame_h * 0.55)),
            # right basket
            (int(frame_w * 0.80), int(frame_h * 0.25),
             int(frame_w * 0.97), int(frame_h * 0.55)),
        ]

        self.scores = [0, 0]            # scores[0] = left end, scores[1] = right end
        self.last_ball_y = {}           # {zone_idx: last_ball_y_above_zone}
        self.cooldown = {0: 0, 1: 0}   # frames until next detection is allowed

    def update(self, ball_centers, frame_idx):
        """
        Call every frame with a list of (bx, by) tuples representing
        the center coordinates of tracked basketballs.
        Returns list of scoring events this frame as (zone_idx, new_score).
        """
        events = []
        for zi, (zx1, zy1, zx2, zy2) in enumerate(self.hoop_zones):
            # Decrement cooldown
            if self.cooldown[zi] > 0:
                self.cooldown[zi] -= 1
                continue

            for bx, by in ball_centers:
                in_zone_x = zx1 <= bx <= zx2
                above_zone = by < zy1 and in_zone_x
                in_zone    = in_zone_x and zy1 <= by <= zy2

                prev_y = self.last_ball_y.get(zi)

                # Score: ball was above hoop zone and now enters the zone
                if in_zone and prev_y is not None and prev_y < zy1:
                    self.scores[zi] += 2   # 2 pts per basket (simplification)
                    self.cooldown[zi] = self.cooldown_frames
                    events.append((zi, self.scores[zi]))
                    self.last_ball_y[zi] = None  # reset
                    print(f"  🏀 BASKET at zone {zi}! Score: {self.scores}")
                elif above_zone:
                    self.last_ball_y[zi] = by

        return events

    def get_total_scores(self):
        return {"left_basket_pts": self.scores[0],
                "right_basket_pts": self.scores[1],
                "total_baskets": sum(self.scores) // 2}
