"""
rim_detector.py
===============
Detects basketball rims/hoops in video frames using computer vision.

Uses a combination of:
  1. Orange/red color filtering (rim is typically orange)
  2. Hough circle detection
  3. Backboard rectangle detection (white/clear rectangle above rim)
  4. Position smoothing across frames (exponential moving average)

Outputs detected Rim primitives with actual pixel positions instead
of relying on hardcoded percentages.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from .primitives import Rim, Point2D


class RimDetector:
    """
    Detects basketball rims in video frames using color + shape analysis.

    The detector maintains a running smoothed position for each rim (left/right)
    to handle frames where the rim is occluded or not detected.
    """

    def __init__(self, frame_w: int, frame_h: int, smoothing: float = 0.3):
        """
        frame_w, frame_h: video dimensions
        smoothing: EMA factor (0 = no update, 1 = instant update)
        """
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.smoothing = smoothing

        # ── Expected rim properties ──────────────────────────────────────
        # A standard basketball rim is ~45.7cm diameter.
        # In a 1920px wide frame showing a full court (~28m), that's roughly:
        # (0.457 / 28) * 1920 ≈ 31 pixels diameter.
        # But the rim can appear 20-80px depending on zoom and camera angle.
        self.min_rim_radius = max(int(frame_w * 0.008), 8)   # ~15px min
        self.max_rim_radius = max(int(frame_w * 0.05), 40)   # ~96px max

        # ── HSV ranges for orange/red rim color ──────────────────────────
        # Orange rim in HSV: H=5-25, S=100-255, V=100-255
        self.rim_hsv_lower1 = np.array([0, 80, 80])
        self.rim_hsv_upper1 = np.array([20, 255, 255])
        self.rim_hsv_lower2 = np.array([160, 80, 80])  # Red wraps around
        self.rim_hsv_upper2 = np.array([180, 255, 255])

        # ── Smoothed rim positions (EMA) ─────────────────────────────────
        # Initialize with fallback positions (will be updated when detected)
        self._left_rim: Optional[Rim] = None
        self._right_rim: Optional[Rim] = None
        self._left_confidence = 0.0
        self._right_confidence = 0.0

        # Detection history for stability
        self._detection_count = 0

    def detect(self, frame: np.ndarray) -> List[Rim]:
        """
        Detect rims in a single frame.
        Returns a list of Rim primitives (typically 0-2).
        Updates internal smoothed positions.
        """
        candidates = self._find_rim_candidates(frame)
        self._update_smoothed(candidates)
        self._detection_count += 1

        result = []
        if self._left_rim is not None:
            result.append(self._left_rim)
        if self._right_rim is not None:
            result.append(self._right_rim)
        return result

    def get_rims(self) -> List[Rim]:
        """Return current best-estimate rim positions."""
        result = []
        if self._left_rim:
            result.append(self._left_rim)
        if self._right_rim:
            result.append(self._right_rim)
        return result

    # ── Internal detection pipeline ──────────────────────────────────────

    def _find_rim_candidates(self, frame: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Find potential rim circles in the frame using color + shape.
        Returns list of (cx, cy, radius) tuples.
        """
        mask = self._get_color_mask(frame)
        candidates = self._find_candidates_from_mask(mask)

        # If color detection fails to find 2 rims, try Hough circles on edges
        if len(candidates) < 2:
            candidates = self._find_candidates_from_hough(frame, candidates)

        return candidates

    def _get_color_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create a cleaned up color mask for orange/red rims."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.rim_hsv_lower1, self.rim_hsv_upper1)
        mask2 = cv2.inRange(hsv, self.rim_hsv_lower2, self.rim_hsv_upper2)
        mask = mask1 | mask2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _find_candidates_from_mask(self, mask: np.ndarray) -> List[Tuple[float, float, float]]:
        """Find and filter contours from the color mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)

            if radius < self.min_rim_radius or radius > self.max_rim_radius:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.2:
                continue

            # rims are usually in top 70% of frame
            if cy > self.frame_h * 0.7:
                continue

            candidates.append((float(cx), float(cy), float(radius)))

        return candidates

    def _find_candidates_from_hough(
        self, frame: np.ndarray, candidates: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Use Hough circles as a fallback when color detection fails."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)

        search_regions = [
            (0, int(self.frame_w * 0.3), 0, int(self.frame_h * 0.6)),  # left
            (int(self.frame_w * 0.7), self.frame_w, 0, int(self.frame_h * 0.6)),  # right
        ]

        for x1, x2, y1, y2 in search_regions:
            roi = blur[y1:y2, x1:x2]
            circles = cv2.HoughCircles(
                roi, cv2.HOUGH_GRADIENT, dp=1.5,
                minDist=50, param1=100, param2=40,
                minRadius=self.min_rim_radius,
                maxRadius=self.max_rim_radius,
            )

            if circles is not None:
                for circle in circles[0]:
                    cx, cy, r = circle
                    cx += x1
                    cy += y1

                    is_new = True
                    for ex_cx, ex_cy, _ in candidates:
                        if abs(cx - ex_cx) < 30 and abs(cy - ex_cy) < 30:
                            is_new = False
                            break
                    if is_new:
                        candidates.append((float(cx), float(cy), float(r)))

        return candidates

    def _update_smoothed(self, candidates: List[Tuple[float, float, float]]):
        """Assign candidates to left/right rim and update EMA positions."""
        mid_x = self.frame_w / 2

        left_candidates = [(cx, cy, r) for cx, cy, r in candidates if cx < mid_x]
        right_candidates = [(cx, cy, r) for cx, cy, r in candidates if cx >= mid_x]

        # Pick the best candidate for each side (closest to the edge)
        if left_candidates:
            # Prefer candidate closest to left edge (actual rim position)
            best_left = min(left_candidates, key=lambda c: c[0])
            self._smooth_rim("left", best_left)
        else:
            # Decay confidence
            self._left_confidence *= 0.95

        if right_candidates:
            # Prefer candidate closest to right edge
            best_right = max(right_candidates, key=lambda c: c[0])
            self._smooth_rim("right", best_right)
        else:
            self._right_confidence *= 0.95

    def _smooth_rim(self, side: str, detection: Tuple[float, float, float]):
        """Apply exponential moving average to rim position."""
        cx, cy, r = detection
        alpha = self.smoothing

        if side == "left":
            if self._left_rim is None:
                self._left_rim = Rim(center=Point2D(cx, cy), radius=r)
                self._left_confidence = 1.0
            else:
                new_cx = self._left_rim.center.x * (1 - alpha) + cx * alpha
                new_cy = self._left_rim.center.y * (1 - alpha) + cy * alpha
                new_r = self._left_rim.radius * (1 - alpha) + r * alpha
                self._left_rim = Rim(center=Point2D(new_cx, new_cy), radius=new_r)
                self._left_confidence = min(1.0, self._left_confidence + 0.1)
        else:
            if self._right_rim is None:
                self._right_rim = Rim(center=Point2D(cx, cy), radius=r)
                self._right_confidence = 1.0
            else:
                new_cx = self._right_rim.center.x * (1 - alpha) + cx * alpha
                new_cy = self._right_rim.center.y * (1 - alpha) + cy * alpha
                new_r = self._right_rim.radius * (1 - alpha) + r * alpha
                self._right_rim = Rim(center=Point2D(new_cx, new_cy), radius=new_r)
                self._right_confidence = min(1.0, self._right_confidence + 0.1)

    @property
    def confidence(self) -> Tuple[float, float]:
        """Return (left_confidence, right_confidence)."""
        return (self._left_confidence, self._right_confidence)

    @property
    def detected(self) -> bool:
        """True if at least one rim has been detected."""
        return self._left_rim is not None or self._right_rim is not None
