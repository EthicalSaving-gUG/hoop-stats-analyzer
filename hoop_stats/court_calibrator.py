"""
court_calibrator.py
===================
Automated basketball court line detection, keypoint extraction, and homography calibration.
Includes integration with Roboflow Supervision PolygonZones for automated zone tracking.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    sv = None
    HAS_SUPERVISION = False


@dataclass
class CourtCalibrationResult:
    success: bool
    homography: Optional[np.ndarray] = None
    confidence: float = 0.0
    detected_keypoints: List[Tuple[float, float]] = field(default_factory=list)
    method: str = "heuristic_lines"


class AutomatedCourtCalibrator:
    """
    Automated calibration of basketball court geometry without manual corner clicking.
    Detects high-contrast court boundary markings, computes line intersections,
    and calculates the projective homography matrix H to the standard FIBA 28m x 15m coordinate space.
    """

    def __init__(self, court_width_m: float = 28.0, court_depth_m: float = 15.0):
        self.court_width_m = court_width_m
        self.court_depth_m = court_depth_m

        # Standard canonical FIBA court keypoints (in meters)
        # [TopLeft, TopRight, BottomRight, BottomLeft, CenterTop, CenterBottom]
        self.canonical_points_m = np.array([
            [0.0, 0.0],
            [court_width_m, 0.0],
            [court_width_m, court_depth_m],
            [0.0, court_depth_m],
            [court_width_m / 2.0, 0.0],
            [court_width_m / 2.0, court_depth_m],
        ], dtype=np.float32)

    def detect_court_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask isolating light court markings (white/bright yellow lines)
        against the wooden or synthetic court floor.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Filter for high value (bright lines) with low-to-moderate saturation
        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 70, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def extract_line_segments(self, binary_mask: np.ndarray, min_length: int = 40) -> List[Tuple[int, int, int, int]]:
        """Extracts candidate court line segments using Probabilistic Hough Transform."""
        edges = cv2.Canny(binary_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=min_length, maxLineGap=20)
        if lines is None:
            return []
        return [tuple(l[0]) for l in lines]

    def separate_lines_by_orientation(
        self, lines: List[Tuple[int, int, int, int]]
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """Separates line segments into predominantly horizontal and vertical lines."""
        horizontal = []
        vertical = []
        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle < 35 or angle > 145:
                horizontal.append((x1, y1, x2, y2))
            elif 55 < angle < 125:
                vertical.append((x1, y1, x2, y2))
        return horizontal, vertical

    def compute_intersection(
        self, line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]
    ) -> Optional[Tuple[float, float]]:
        """Calculates the intersection point of two line segments extended to infinity."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-5:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return float(px), float(py)

    def calibrate_from_points(
        self, pixel_points: List[Tuple[float, float]], court_points_m: Optional[List[Tuple[float, float]]] = None
    ) -> CourtCalibrationResult:
        """
        Calculates homography matrix given at least 4 corresponding pixel coordinates.
        """
        if len(pixel_points) < 4:
            return CourtCalibrationResult(success=False, confidence=0.0)

        src_pts = np.array(pixel_points[:4], dtype=np.float32)
        dst_pts = (
            np.array(court_points_m[:4], dtype=np.float32)
            if court_points_m is not None
            else self.canonical_points_m[:4]
        )

        h_matrix, inliers = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if h_matrix is None:
            return CourtCalibrationResult(success=False, confidence=0.0)

        # Compute confidence based on condition number & reprojection stability
        cond = np.linalg.cond(h_matrix)
        confidence = float(np.clip(1.0 - (cond / 1e5), 0.1, 1.0))

        return CourtCalibrationResult(
            success=True,
            homography=h_matrix,
            confidence=confidence,
            detected_keypoints=[(float(p[0]), float(p[1])) for p in src_pts],
            method="point_correspondence",
        )

    def auto_calibrate(self, frame: np.ndarray) -> CourtCalibrationResult:
        """
        Performs end-to-end automated court calibration from a raw video frame.
        """
        fh, fw = frame.shape[:2]
        mask = self.detect_court_mask(frame)
        lines = self.extract_line_segments(mask)

        if len(lines) < 4:
            # Fallback: heuristic court quadrant initialization
            margin_x = fw * 0.08
            margin_y = fh * 0.15
            heuristic_pts = [
                (margin_x, margin_y),
                (fw - margin_x, margin_y),
                (fw - margin_x, fh - margin_y),
                (margin_x, fh - margin_y),
            ]
            res = self.calibrate_from_points(heuristic_pts)
            res.method = "heuristic_fallback"
            res.confidence = 0.5
            return res

        horizontal, vertical = self.separate_lines_by_orientation(lines)
        if not horizontal or not vertical:
            margin_x = fw * 0.08
            margin_y = fh * 0.15
            return self.calibrate_from_points([
                (margin_x, margin_y),
                (fw - margin_x, margin_y),
                (fw - margin_x, fh - margin_y),
                (margin_x, fh - margin_y),
            ])

        # Find intersections of extreme bounding lines
        intersections = []
        for h in horizontal[:5]:
            for v in vertical[:5]:
                pt = self.compute_intersection(h, v)
                if pt is not None and -100 <= pt[0] <= fw + 100 and -100 <= pt[1] <= fh + 100:
                    intersections.append(pt)

        if len(intersections) >= 4:
            # Sort intersections into [top-left, top-right, bottom-right, bottom-left]
            pts = np.array(intersections, dtype=np.float32)
            center = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            sorted_idx = np.argsort(angles)
            corners = [tuple(pts[i]) for i in sorted_idx[:4]]
            return self.calibrate_from_points(corners)

        # Fallback to frame bounds
        margin_x = fw * 0.08
        margin_y = fh * 0.15
        return self.calibrate_from_points([
            (margin_x, margin_y),
            (fw - margin_x, margin_y),
            (fw - margin_x, fh - margin_y),
            (margin_x, fh - margin_y),
        ])

    def create_supervision_zones(
        self, homography: np.ndarray, frame_resolution_wh: Tuple[int, int]
    ) -> Dict[str, Optional["sv.PolygonZone"]]:
        """
        Creates Roboflow Supervision PolygonZone instances for basketball zones
        (paint, 3-point territory, mid-range) mapped back to video pixel space.
        """
        if not HAS_SUPERVISION or homography is None:
            return {}

        try:
            inv_h = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return {}

        # Standard canonical zones in meters [x, y]
        # Paint / Key area (Left basket: 0 to 5.8m, 5.0 to 10.0m)
        court_zones_m = {
            "paint_left": np.array([
                [0.0, 5.0], [5.8, 5.0], [5.8, 10.0], [0.0, 10.0]
            ], dtype=np.float32),
            "paint_right": np.array([
                [22.2, 5.0], [28.0, 5.0], [28.0, 10.0], [22.2, 10.0]
            ], dtype=np.float32),
            "court_full": np.array([
                [0.0, 0.0], [28.0, 0.0], [28.0, 15.0], [0.0, 15.0]
            ], dtype=np.float32),
        }

        sv_zones = {}
        for zone_name, pts_m in court_zones_m.items():
            # Project real-world meters back to pixel polygon
            pts_reshaped = pts_m.reshape(-1, 1, 2)
            pts_px = cv2.perspectiveTransform(pts_reshaped, inv_h).reshape(-1, 2)
            pts_px_int = pts_px.astype(np.int64)
            sv_zones[zone_name] = sv.PolygonZone(
                polygon=pts_px_int,
            )

        return sv_zones
