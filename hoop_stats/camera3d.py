"""
camera3d.py
===========
Camera pose estimation and 2D → 3D lifting for the basketball court.

Uses cv2.solvePnP to compute the camera's position and orientation from
known court reference points, then provides methods to:
  - Project pixel coordinates onto the ground plane (Y=0) → Point3D
  - Estimate object heights from pixel span
  - Lift 2D primitives (Ball, PlayerCapsule) to 3D (Sphere3D, Cylinder3D)
  - Project 3D points back to any virtual camera view
"""

import math
import numpy as np
import cv2
from typing import List, Optional, Tuple

from .primitives import (
    Point2D, Point3D, Ball, PlayerCapsule, Rim,
    Sphere3D, Cylinder3D, Ring3D, Rect3D,
)

# Standard basketball dimensions (FIBA)
RIM_HEIGHT_M = 3.05          # rim height from floor
BACKBOARD_HEIGHT_M = 3.95    # top of backboard
BASKETBALL_RADIUS_M = 0.12   # ball radius
PLAYER_SHOULDER_HALF_M = 0.25  # average shoulder half-width
COURT_WIDTH_M = 28.0
COURT_DEPTH_M = 15.0


class Camera3D:
    """
    Camera model estimated from known court reference points.

    Coordinate system (right-handed):
      X = right  (along the court length)
      Y = up
      Z = forward (along the court width / depth)
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Camera intrinsics (estimated from frame size, refined during calibration)
        focal = max(frame_w, frame_h) * 1.0  # reasonable default
        cx, cy = frame_w / 2.0, frame_h / 2.0
        self.camera_matrix = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        # Camera extrinsics (set by calibrate())
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None  # rotation matrix 3×3
        self.camera_pos: Optional[Point3D] = None  # camera location in world

        self.calibrated = False

    # ── Calibration ──────────────────────────────────────────────────────

    def calibrate(
        self,
        pixel_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float, float]],
    ):
        """
        Solve camera pose from ≥4 known 2D↔3D correspondences.

        pixel_points: [(px, py), ...] in image coordinates
        world_points: [(X, Y, Z), ...] in meters (Y=0 for floor points)
        """
        pts_2d = np.array(pixel_points, dtype=np.float64)
        pts_3d = np.array(world_points, dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise RuntimeError("solvePnP failed — check your calibration points")

        self.rvec = rvec
        self.tvec = tvec
        self._R, _ = cv2.Rodrigues(rvec)

        # Camera position in world = -R^T · t
        cam_world = -self._R.T @ tvec
        self.camera_pos = Point3D(
            float(cam_world[0, 0]),
            float(cam_world[1, 0]),
            float(cam_world[2, 0]),
        )
        self.calibrated = True

    def calibrate_from_court_corners(
        self,
        pixel_corners: List[Tuple[float, float]],
        court_w: float = COURT_WIDTH_M,
        court_d: float = COURT_DEPTH_M,
    ):
        """
        Convenience: calibrate from 4 visible court floor corners.
        Order: top-left, top-right, bottom-right, bottom-left as seen in video.
        World coords: court floor at Y=0.
        """
        world_corners = [
            (0.0,     0.0, 0.0),       # top-left
            (court_w, 0.0, 0.0),       # top-right
            (court_w, 0.0, court_d),   # bottom-right
            (0.0,     0.0, court_d),   # bottom-left
        ]
        self.calibrate(pixel_corners, world_corners)

    # ── Projection: pixel → 3D ───────────────────────────────────────────

    def pixel_to_ground(self, px: float, py: float, ground_y: float = 0.0) -> Point3D:
        """
        Project a pixel (px, py) onto the ground plane (Y = ground_y).
        Returns the 3D world point where the camera ray hits the plane.
        """
        if not self.calibrated:
            # Fallback: simple normalized coords
            return Point3D(px / self.frame_w * COURT_WIDTH_M, ground_y,
                           py / self.frame_h * COURT_DEPTH_M)

        # Undo intrinsics: pixel → normalized camera ray
        inv_K = np.linalg.inv(self.camera_matrix)
        pixel_h = np.array([[px], [py], [1.0]], dtype=np.float64)
        ray_cam = inv_K @ pixel_h  # direction in camera frame

        # Rotate to world frame
        ray_world = self._R.T @ ray_cam  # direction in world
        cam_world = -self._R.T @ self.tvec  # camera origin in world

        # Intersect ray with plane Y = ground_y
        # P = cam + t * ray → P.y = ground_y → t = (ground_y - cam.y) / ray.y
        if abs(ray_world[1, 0]) < 1e-8:
            # Ray is parallel to ground — return projection directly below camera
            return Point3D(float(cam_world[0, 0]), ground_y, float(cam_world[2, 0]))

        t = (ground_y - cam_world[1, 0]) / ray_world[1, 0]
        hit = cam_world + t * ray_world

        return Point3D(float(hit[0, 0]), ground_y, float(hit[2, 0]))

    def estimate_height_m(
        self,
        foot_px: Tuple[float, float],
        head_px: Tuple[float, float],
    ) -> float:
        """
        Estimate the real-world height (meters) of an object given its foot
        and head positions in pixel coordinates.

        Uses the ground-plane distance of the foot to compute scale.
        """
        if not self.calibrated:
            # Rough heuristic: pixel height / frame_height * 2.0m
            pixel_h = math.hypot(head_px[0] - foot_px[0], head_px[1] - foot_px[1])
            return pixel_h / self.frame_h * 4.0

        foot_3d = self.pixel_to_ground(foot_px[0], foot_px[1])

        # Project a point 1m above the foot back to pixels, measure the pixel span
        test_3d = np.array([[foot_3d.x, 1.0, foot_3d.z]], dtype=np.float64)
        test_px, _ = cv2.projectPoints(test_3d, self.rvec, self.tvec,
                                       self.camera_matrix, self.dist_coeffs)
        base_3d = np.array([[foot_3d.x, 0.0, foot_3d.z]], dtype=np.float64)
        base_px, _ = cv2.projectPoints(base_3d, self.rvec, self.tvec,
                                       self.camera_matrix, self.dist_coeffs)

        px_per_meter = math.hypot(
            test_px[0, 0, 0] - base_px[0, 0, 0],
            test_px[0, 0, 1] - base_px[0, 0, 1],
        )
        if px_per_meter < 1e-3:
            return 1.8  # fallback

        actual_px = math.hypot(head_px[0] - foot_px[0], head_px[1] - foot_px[1])
        return actual_px / px_per_meter

    # ── Projection: 3D → pixel ───────────────────────────────────────────

    def project(self, pt: Point3D) -> Tuple[float, float]:
        """Project a 3D world point to pixel coordinates in this camera."""
        if not self.calibrated:
            # Fallback
            return (pt.x / COURT_WIDTH_M * self.frame_w,
                    pt.z / COURT_DEPTH_M * self.frame_h)

        pts_3d = np.array([[pt.x, pt.y, pt.z]], dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(
            pts_3d, self.rvec, self.tvec,
            self.camera_matrix, self.dist_coeffs,
        )
        return (float(pts_2d[0, 0, 0]), float(pts_2d[0, 0, 1]))

    def project_to_virtual(
        self,
        pt: Point3D,
        virtual_rvec: np.ndarray,
        virtual_tvec: np.ndarray,
        virtual_K: np.ndarray,
    ) -> Tuple[float, float]:
        """Project a 3D world point to a different (virtual) camera."""
        pts_3d = np.array([[pt.x, pt.y, pt.z]], dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(
            pts_3d, virtual_rvec, virtual_tvec,
            virtual_K, self.dist_coeffs,
        )
        return (float(pts_2d[0, 0, 0]), float(pts_2d[0, 0, 1]))

    # ── 2D → 3D lifting ─────────────────────────────────────────────────

    def lift_player(self, cap: PlayerCapsule) -> Cylinder3D:
        """Lift a 2D PlayerCapsule to a 3D Cylinder."""
        foot_3d = self.pixel_to_ground(cap.foot.x, cap.foot.y)
        height_m = self.estimate_height_m(
            (cap.foot.x, cap.foot.y),
            (cap.head.x, cap.head.y),
        )
        return Cylinder3D(
            base=foot_3d,
            height=height_m,
            radius=PLAYER_SHOULDER_HALF_M,
            obj_id=cap.obj_id,
        )

    def lift_ball(
        self, ball: Ball,
        nearby_players: Optional[List[Cylinder3D]] = None,
    ) -> Sphere3D:
        """
        Lift a 2D Ball to a 3D Sphere.

        If nearby_players are given, estimate the ball's Y from their heights.
        Otherwise, assume the ball is at rim height (~3.05m).
        """
        ground_pt = self.pixel_to_ground(ball.center.x, ball.center.y)

        # Estimate ball height
        ball_y = RIM_HEIGHT_M  # default assumption

        if nearby_players:
            # Find the closest player and use their position for context
            min_dist = float('inf')
            closest = None
            for cyl in nearby_players:
                d = math.hypot(ground_pt.x - cyl.base.x, ground_pt.z - cyl.base.z)
                if d < min_dist:
                    min_dist = d
                    closest = cyl
            if closest and min_dist < 3.0:
                # Ball is near a player — estimate relative height
                # Use the pixel Y ratio between ball and player
                ball_y = closest.height * 0.8  # rough: ball held at ~80% player height

        center = Point3D(ground_pt.x, ball_y, ground_pt.z)
        return Sphere3D(center=center, radius=BASKETBALL_RADIUS_M)

    def lift_rim(self, rim_2d: Rim, side: str = "left") -> Ring3D:
        """
        Create a 3D Ring for the basket rim at its known real-world position.
        side: 'left' (X=0 end) or 'right' (X=court_width end)
        """
        if side == "left":
            x = 1.575  # rim position from baseline (FIBA: 1.575m)
        else:
            x = COURT_WIDTH_M - 1.575
        z = COURT_DEPTH_M / 2.0  # centered on court width

        return Ring3D(
            center=Point3D(x, RIM_HEIGHT_M, z),
            radius=0.2286,  # standard rim inner radius (45.72cm / 2)
        )


# ── Virtual camera presets ───────────────────────────────────────────────────

def make_virtual_camera(
    preset: str,
    frame_w: int = 1280,
    frame_h: int = 720,
    court_w: float = COURT_WIDTH_M,
    court_d: float = COURT_DEPTH_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create virtual camera parameters (rvec, tvec, K) for a preset viewing angle.

    Returns (rvec, tvec, camera_matrix).
    """
    focal = max(frame_w, frame_h) * 0.9
    K = np.array([
        [focal, 0, frame_w / 2.0],
        [0, focal, frame_h / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)

    # Camera position and look-at target
    mid_x = court_w / 2.0
    mid_z = court_d / 2.0

    if preset == "birdseye":
        cam_pos = np.array([mid_x, 22.0, mid_z])  # directly above, 22m up
        look_at = np.array([mid_x, 0.0, mid_z])
        up = np.array([0.0, 0.0, -1.0])
    elif preset == "sideline":
        cam_pos = np.array([mid_x, 8.0, -12.0])  # sideline view
        look_at = np.array([mid_x, 1.5, mid_z])
        up = np.array([0.0, 1.0, 0.0])
    elif preset == "corner":
        cam_pos = np.array([-5.0, 10.0, -5.0])  # elevated corner
        look_at = np.array([mid_x, 1.5, mid_z])
        up = np.array([0.0, 1.0, 0.0])
    elif preset == "broadcast":
        cam_pos = np.array([mid_x, 6.0, -8.0])  # classic TV angle
        look_at = np.array([mid_x, 0.5, mid_z])
        up = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError(f"Unknown camera preset: {preset}")

    # Build rotation matrix from look-at
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # R = world-to-camera rotation
    R = np.array([right, -true_up, forward], dtype=np.float64)
    tvec = (-R @ cam_pos).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)

    return rvec, tvec, K
