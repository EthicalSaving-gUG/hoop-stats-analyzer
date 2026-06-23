"""
primitive_renderer.py
=====================
Renders simplified geometric views of basketball games.

Three modes:
  - wireframe: dark canvas with neon-colored 2D primitives only
  - overlay:   original video frame with semi-transparent 2D primitives on top
  - 3d:        virtual camera view of 3D primitives (spheres, cylinders, rings)
"""

import cv2
import math
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

from .primitives import (
    Ball, Rim, Backboard, PlayerCapsule, CourtLine, Point2D,
    Point3D, Sphere3D, Cylinder3D, Ring3D,
    bbox_to_ball, bbox_to_player,
)
from .camera3d import Camera3D, make_virtual_camera, COURT_WIDTH_M, COURT_DEPTH_M
from .detector import HoopDetector
from .geometry import CourtGeometry
from .rim_detector import RimDetector
from .shot_tracker import ShotTracker
from .video_utils import get_video_info


# ─── Color palette (BGR for OpenCV) ─────────────────────────────────────────

COLORS = {
    "bg":           (20,  17,  15),      # near-black canvas
    "court_lines":  (80,  80,  80),      # subtle gray
    "court_zones":  (40,  40,  35),      # very dark fill for zones
    "ball":         (40, 180, 255),       # orange
    "ball_glow":    (30, 130, 200),       # dimmer orange for glow
    "rim":          (50, 50,  255),       # red
    "rim_glow":     (30, 30,  180),       # dimmer red
    "backboard":    (200, 200, 200),      # light gray
    "team_a":       (255, 180, 50),       # cyan-ish
    "team_b":       (50,  120, 255),      # orange-red-ish
    "unknown_team": (160, 160, 160),      # gray
    "hud_bg":       (30,  27,  25),       # dark bar
    "hud_text":     (220, 220, 220),      # white text
    "hud_accent":   (40, 180, 255),       # orange accent
    "trail":        (100, 255, 100),      # green trail dots
}

TEAM_COLORS = [COLORS["team_a"], COLORS["team_b"], COLORS["unknown_team"]]


# ─── Renderer ────────────────────────────────────────────────────────────────

class PrimitiveRenderer:
    """
    Renders a simplified geometric view of a basketball game frame.
    """

    def __init__(self, width: int, height: int, mode: str = "wireframe"):
        """
        width/height: output frame dimensions.
        mode: "wireframe" or "overlay"
        """
        self.w = width
        self.h = height
        self.mode = mode

        # Build court lines from config
        self.court_lines: List[CourtLine] = []
        self._build_default_court()

        # Trail history for ball {frame_idx: Point2D}
        self.ball_trail: List[Point2D] = []
        self.max_trail = 20

    # ── Court geometry ───────────────────────────────────────────────────

    def _build_default_court(self):
        """Create court boundary and key lines scaled to pixel space."""
        w, h = self.w, self.h
        margin_x = int(w * 0.02)
        margin_y = int(h * 0.08)
        cw = w - 2 * margin_x  # court pixel width
        ch = h - 2 * margin_y  # court pixel height
        x0, y0 = margin_x, margin_y

        def p(rx, ry):
            return Point2D(x0 + rx * cw, y0 + ry * ch)

        # Outer boundary
        self.court_lines += [
            CourtLine(p(0, 0), p(1, 0)),
            CourtLine(p(1, 0), p(1, 1)),
            CourtLine(p(1, 1), p(0, 1)),
            CourtLine(p(0, 1), p(0, 0)),
        ]
        # Half-court line
        self.court_lines.append(CourtLine(p(0.5, 0), p(0.5, 1)))
        # Center circle (approximate with line segments)
        cx, cy = x0 + cw * 0.5, y0 + ch * 0.5
        r = min(cw, ch) * 0.08
        n_seg = 24
        for i in range(n_seg):
            a1 = 2 * np.pi * i / n_seg
            a2 = 2 * np.pi * (i + 1) / n_seg
            self.court_lines.append(CourtLine(
                Point2D(cx + r * np.cos(a1), cy + r * np.sin(a1)),
                Point2D(cx + r * np.cos(a2), cy + r * np.sin(a2)),
            ))
        # Paint / key (both sides)
        paint_w = 0.17  # relative width of paint area
        paint_h_top = 0.19
        paint_h_bot = 1.0 - paint_h_top
        # Left paint
        self.court_lines += [
            CourtLine(p(0, paint_h_top), p(paint_w, paint_h_top)),
            CourtLine(p(paint_w, paint_h_top), p(paint_w, paint_h_bot)),
            CourtLine(p(paint_w, paint_h_bot), p(0, paint_h_bot)),
        ]
        # Right paint
        self.court_lines += [
            CourtLine(p(1, paint_h_top), p(1 - paint_w, paint_h_top)),
            CourtLine(p(1 - paint_w, paint_h_top), p(1 - paint_w, paint_h_bot)),
            CourtLine(p(1 - paint_w, paint_h_bot), p(1, paint_h_bot)),
        ]
        # Three-point arcs (simplified as straight lines at the wings)
        arc_w = 0.23
        arc_top = 0.06
        arc_bot = 1.0 - arc_top
        # Left three-point
        self.court_lines += [
            CourtLine(p(0, arc_top), p(arc_w, arc_top)),
            CourtLine(p(arc_w, arc_top), p(arc_w, paint_h_top)),
        ]
        self.court_lines += [
            CourtLine(p(0, arc_bot), p(arc_w, arc_bot)),
            CourtLine(p(arc_w, arc_bot), p(arc_w, paint_h_bot)),
        ]
        # Right three-point
        self.court_lines += [
            CourtLine(p(1, arc_top), p(1 - arc_w, arc_top)),
            CourtLine(p(1 - arc_w, arc_top), p(1 - arc_w, paint_h_top)),
        ]
        self.court_lines += [
            CourtLine(p(1, arc_bot), p(1 - arc_w, arc_bot)),
            CourtLine(p(1 - arc_w, arc_bot), p(1 - arc_w, paint_h_bot)),
        ]

    # ── Drawing primitives ───────────────────────────────────────────────

    def _draw_court(self, canvas):
        """Draw all court lines."""
        for line in self.court_lines:
            pt1 = (int(line.start.x), int(line.start.y))
            pt2 = (int(line.end.x), int(line.end.y))
            cv2.line(canvas, pt1, pt2, COLORS["court_lines"], 1, cv2.LINE_AA)

    def _draw_ball(self, canvas, ball: Ball):
        """Draw the ball as a filled circle with a soft glow."""
        cx, cy = int(ball.center.x), int(ball.center.y)
        r = max(int(ball.radius), 6)

        # Glow layer (larger, semi-transparent)
        glow_overlay = canvas.copy()
        cv2.circle(glow_overlay, (cx, cy), r + 8, COLORS["ball_glow"], -1, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.3, canvas, 0.7, 0, canvas)

        # Solid ball
        cv2.circle(canvas, (cx, cy), r, COLORS["ball"], -1, cv2.LINE_AA)
        # Highlight dot
        cv2.circle(canvas, (cx - r // 3, cy - r // 3), max(r // 4, 2),
                   (180, 230, 255), -1, cv2.LINE_AA)

    def _draw_ball_trail(self, canvas):
        """Draw fading trail behind the ball."""
        n = len(self.ball_trail)
        for i, pt in enumerate(self.ball_trail):
            alpha = (i + 1) / n  # 0→1 fade
            radius = max(2, int(3 * alpha))
            color = tuple(int(c * alpha * 0.6) for c in COLORS["trail"])
            cv2.circle(canvas, (int(pt.x), int(pt.y)), radius, color, -1, cv2.LINE_AA)

    def _draw_player(self, canvas, cap: PlayerCapsule, team_idx: int = 2):
        """Draw a player capsule as a rounded rectangle (tube)."""
        color = TEAM_COLORS[min(team_idx, 2)]
        hx, hy = int(cap.head.x), int(cap.head.y)
        fx, fy = int(cap.foot.x), int(cap.foot.y)
        half_w = max(int(cap.width / 2), 6)

        # Draw the capsule body (thick line between head and foot)
        cv2.line(canvas, (hx, hy), (fx, fy), color, half_w, cv2.LINE_AA)

        # Head circle
        head_r = max(half_w // 2 + 2, 5)
        cv2.circle(canvas, (hx, hy), head_r, color, -1, cv2.LINE_AA)

        # Foot base
        cv2.circle(canvas, (fx, fy), max(half_w // 3, 3), color, -1, cv2.LINE_AA)

        # ID label
        label = str(cap.obj_id) if cap.obj_id is not None else "?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        lx = hx - tw // 2
        ly = hy - head_r - 6
        # Background pill
        cv2.rectangle(canvas, (lx - 4, ly - th - 2), (lx + tw + 4, ly + 4),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (lx, ly), font, 0.45, color, 1, cv2.LINE_AA)

    def _draw_rim(self, canvas, rim: Rim):
        """Draw the rim as a circle outline with glow."""
        cx, cy = int(rim.center.x), int(rim.center.y)
        r = max(int(rim.radius), 8)

        # Glow
        glow = canvas.copy()
        cv2.circle(glow, (cx, cy), r + 6, COLORS["rim_glow"], 2, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.4, canvas, 0.6, 0, canvas)

        # Rim circle
        cv2.circle(canvas, (cx, cy), r, COLORS["rim"], 2, cv2.LINE_AA)
        # Net indicator — small vertical lines
        for dx in range(-r // 2, r // 2 + 1, max(r // 4, 4)):
            cv2.line(canvas, (cx + dx, cy + r),
                     (cx + dx, cy + r + max(r // 2, 8)),
                     (80, 80, 100), 1, cv2.LINE_AA)

    def _draw_hud(self, canvas, scores: dict, frame_idx: int, fps: float):
        """Draw a heads-up display bar at the top."""
        bar_h = 48
        # Semi-transparent background bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, bar_h), COLORS["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        font = cv2.FONT_HERSHEY_SIMPLEX
        home_pts = scores.get("home_pts", scores.get("left_basket_pts", 0))
        away_pts = scores.get("away_pts", scores.get("right_basket_pts", 0))
        elapsed = frame_idx / max(fps, 1)
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        # Score (team-based)
        score_text = f"HOME {home_pts}  -  {away_pts} AWAY"
        (tw, _), _ = cv2.getTextSize(score_text, font, 0.7, 2)
        sx = (self.w - tw) // 2
        cv2.putText(canvas, score_text, (sx, 35), font, 0.7,
                    COLORS["hud_accent"], 2, cv2.LINE_AA)

        # Timer
        time_text = f"{minutes:02d}:{seconds:02d}"
        cv2.putText(canvas, time_text, (self.w - 120, 35), font, 0.65,
                    COLORS["hud_text"], 1, cv2.LINE_AA)

        # Mode label + halftime indicator
        mode_label = "WIREFRAME" if self.mode == "wireframe" else "OVERLAY"
        if scores.get("sides_swapped", False):
            mode_label += "  |  2ND HALF"
        cv2.putText(canvas, mode_label, (16, 33), font, 0.5,
                    COLORS["court_lines"], 1, cv2.LINE_AA)

        # Thin accent line at bottom of bar
        cv2.line(canvas, (0, bar_h), (self.w, bar_h), COLORS["hud_accent"], 1)

    # ── Composite frame ──────────────────────────────────────────────────

    def render_frame(
        self,
        original_frame: Optional[np.ndarray],
        balls: List[Ball],
        player_capsules: Dict[int, PlayerCapsule],
        rims: List[Rim],
        scores: dict,
        frame_idx: int = 0,
        fps: float = 30.0,
        team_assignments: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        Render one complete frame with all primitives.

        team_assignments: {player_id: 0 or 1} for team coloring
        """
        if self.mode == "wireframe":
            canvas = np.full((self.h, self.w, 3), COLORS["bg"], dtype=np.uint8)
        else:
            # Overlay mode: start with the original frame (dimmed)
            if original_frame is not None:
                h_orig, w_orig = original_frame.shape[:2]
                if w_orig != self.w or h_orig != self.h:
                    canvas = cv2.resize(original_frame, (self.w, self.h))
                else:
                    canvas = original_frame.copy()
                # Dim the original frame to make primitives pop
                canvas = (canvas * 0.35).astype(np.uint8)
            else:
                canvas = np.full((self.h, self.w, 3), COLORS["bg"], dtype=np.uint8)

        # 1. Court
        self._draw_court(canvas)

        # 2. Ball trail
        self._draw_ball_trail(canvas)

        # 3. Rims
        for rim in rims:
            self._draw_rim(canvas, rim)

        # 4. Players
        if team_assignments is None:
            team_assignments = {}
        for pid, cap in player_capsules.items():
            tidx = team_assignments.get(pid, 2)  # 2 = unknown
            self._draw_player(canvas, cap, tidx)

        # 5. Ball(s)
        for ball in balls:
            self._draw_ball(canvas, ball)
            # Update trail
            self.ball_trail.append(ball.center)
            if len(self.ball_trail) > self.max_trail:
                self.ball_trail.pop(0)

        # 6. HUD
        self._draw_hud(canvas, scores, frame_idx, fps)

        return canvas

    # ── 3D rendering methods ─────────────────────────────────────────────

    def _project(self, pt: Point3D, rvec, tvec, K) -> Optional[Tuple[int, int]]:
        """Project a 3D point to pixel. Returns None if behind camera."""
        pts_3d = np.array([[pt.x, pt.y, pt.z]], dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, np.zeros(5))
        px, py = int(pts_2d[0, 0, 0]), int(pts_2d[0, 0, 1])
        if 0 <= px < self.w and 0 <= py < self.h:
            return (px, py)
        return None

    def _draw_sphere_3d(self, canvas, sphere: Sphere3D, rvec, tvec, K):
        """Draw a 3D sphere as a projected circle with shading."""
        center_px = self._project(sphere.center, rvec, tvec, K)
        if center_px is None:
            return
        # Estimate projected radius: project a point offset by sphere.radius
        edge_pt = Point3D(sphere.center.x + sphere.radius, sphere.center.y, sphere.center.z)
        edge_px = self._project(edge_pt, rvec, tvec, K)
        if edge_px is None:
            r_px = 8
        else:
            r_px = max(6, int(math.hypot(edge_px[0] - center_px[0], edge_px[1] - center_px[1])))
        # Glow
        glow = canvas.copy()
        cv2.circle(glow, center_px, r_px + 6, COLORS["ball_glow"], -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.3, canvas, 0.7, 0, canvas)
        # Ball
        cv2.circle(canvas, center_px, r_px, COLORS["ball"], -1, cv2.LINE_AA)
        cv2.circle(canvas, (center_px[0] - r_px // 3, center_px[1] - r_px // 3),
                   max(r_px // 4, 2), (180, 230, 255), -1, cv2.LINE_AA)

    def _draw_cylinder_3d(self, canvas, cyl: Cylinder3D, rvec, tvec, K, team_idx=2):
        """Draw a 3D cylinder as a projected vertical shape with head circle."""
        color = TEAM_COLORS[min(team_idx, 2)]
        base_px = self._project(cyl.base, rvec, tvec, K)
        top_px = self._project(cyl.top, rvec, tvec, K)
        if base_px is None or top_px is None:
            return
        # Estimate width in pixels
        side_pt = Point3D(cyl.base.x + cyl.radius, cyl.base.y, cyl.base.z)
        side_px = self._project(side_pt, rvec, tvec, K)
        if side_px:
            half_w = max(abs(side_px[0] - base_px[0]), 4)
        else:
            half_w = 8
        # Draw body as thick line
        cv2.line(canvas, top_px, base_px, color, half_w, cv2.LINE_AA)
        # Head ellipse
        head_r = max(half_w // 2 + 2, 4)
        cv2.circle(canvas, top_px, head_r, color, -1, cv2.LINE_AA)
        # Foot base
        cv2.circle(canvas, base_px, max(half_w // 3, 3), color, -1, cv2.LINE_AA)
        # ID label
        label = str(cyl.obj_id) if cyl.obj_id is not None else "?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
        cv2.rectangle(canvas, (top_px[0] - tw // 2 - 3, top_px[1] - head_r - th - 5),
                      (top_px[0] + tw // 2 + 3, top_px[1] - head_r), (0, 0, 0), -1)
        cv2.putText(canvas, label, (top_px[0] - tw // 2, top_px[1] - head_r - 3),
                    font, 0.4, color, 1, cv2.LINE_AA)

    def _draw_ring_3d(self, canvas, ring: Ring3D, rvec, tvec, K):
        """Draw a 3D ring as a projected ellipse."""
        center_px = self._project(ring.center, rvec, tvec, K)
        if center_px is None:
            return
        # Sample points around the ring to form an ellipse
        n_pts = 24
        ring_pts = []
        for i in range(n_pts):
            angle = 2 * math.pi * i / n_pts
            pt = Point3D(
                ring.center.x + ring.radius * math.cos(angle),
                ring.center.y,
                ring.center.z + ring.radius * math.sin(angle),
            )
            px = self._project(pt, rvec, tvec, K)
            if px:
                ring_pts.append(px)
        if len(ring_pts) >= 4:
            pts_arr = np.array(ring_pts, dtype=np.int32)
            cv2.polylines(canvas, [pts_arr], True, COLORS["rim"], 2, cv2.LINE_AA)
            # Net lines
            for i in range(0, len(ring_pts), max(len(ring_pts) // 6, 1)):
                px = ring_pts[i]
                net_bottom = Point3D(
                    ring.center.x + ring.radius * 0.5 * math.cos(2 * math.pi * i / n_pts),
                    ring.center.y - 0.4,  # net hangs ~40cm
                    ring.center.z + ring.radius * 0.5 * math.sin(2 * math.pi * i / n_pts),
                )
                nb_px = self._project(net_bottom, rvec, tvec, K)
                if nb_px:
                    cv2.line(canvas, px, nb_px, (80, 80, 100), 1, cv2.LINE_AA)

    def _draw_court_3d(self, canvas, rvec, tvec, K):
        """Draw court floor as a 3D projected grid."""
        cw, cd = COURT_WIDTH_M, COURT_DEPTH_M
        # Boundary
        corners_3d = [
            Point3D(0, 0, 0), Point3D(cw, 0, 0),
            Point3D(cw, 0, cd), Point3D(0, 0, cd),
        ]
        corner_px = [self._project(c, rvec, tvec, K) for c in corners_3d]
        valid = [p for p in corner_px if p is not None]
        if len(valid) >= 2:
            for i in range(len(corner_px)):
                j = (i + 1) % 4
                if corner_px[i] and corner_px[j]:
                    cv2.line(canvas, corner_px[i], corner_px[j],
                             COLORS["court_lines"], 1, cv2.LINE_AA)
        # Half-court line
        hc_a = self._project(Point3D(cw / 2, 0, 0), rvec, tvec, K)
        hc_b = self._project(Point3D(cw / 2, 0, cd), rvec, tvec, K)
        if hc_a and hc_b:
            cv2.line(canvas, hc_a, hc_b, COLORS["court_lines"], 1, cv2.LINE_AA)
        # Center circle
        n_seg = 24
        r = min(cw, cd) * 0.06
        cx, cz = cw / 2, cd / 2
        prev = None
        for i in range(n_seg + 1):
            a = 2 * math.pi * i / n_seg
            pt = self._project(Point3D(cx + r * math.cos(a), 0, cz + r * math.sin(a)),
                               rvec, tvec, K)
            if prev and pt:
                cv2.line(canvas, prev, pt, COLORS["court_lines"], 1, cv2.LINE_AA)
            prev = pt
        # Paint areas
        paint_lines = [
            # Left paint
            (Point3D(0, 0, cd * 0.28), Point3D(5.8, 0, cd * 0.28)),
            (Point3D(5.8, 0, cd * 0.28), Point3D(5.8, 0, cd * 0.72)),
            (Point3D(5.8, 0, cd * 0.72), Point3D(0, 0, cd * 0.72)),
            # Right paint
            (Point3D(cw, 0, cd * 0.28), Point3D(cw - 5.8, 0, cd * 0.28)),
            (Point3D(cw - 5.8, 0, cd * 0.28), Point3D(cw - 5.8, 0, cd * 0.72)),
            (Point3D(cw - 5.8, 0, cd * 0.72), Point3D(cw, 0, cd * 0.72)),
        ]
        for a_3d, b_3d in paint_lines:
            a_px = self._project(a_3d, rvec, tvec, K)
            b_px = self._project(b_3d, rvec, tvec, K)
            if a_px and b_px:
                cv2.line(canvas, a_px, b_px, COLORS["court_lines"], 1, cv2.LINE_AA)

    def render_frame_3d(
        self,
        spheres: List[Sphere3D],
        cylinders: Dict[int, Cylinder3D],
        rings: List[Ring3D],
        scores: dict,
        frame_idx: int = 0,
        fps: float = 30.0,
        team_assignments: Optional[Dict[int, int]] = None,
        camera_preset: str = "broadcast",
    ) -> np.ndarray:
        """Render a 3D scene from a virtual camera angle."""
        canvas = np.full((self.h, self.w, 3), COLORS["bg"], dtype=np.uint8)
        rvec, tvec, K = make_virtual_camera(
            camera_preset, self.w, self.h,
        )
        # 1. Court floor
        self._draw_court_3d(canvas, rvec, tvec, K)
        # 2. Rings (rims)
        for ring in rings:
            self._draw_ring_3d(canvas, ring, rvec, tvec, K)
        # 3. Players
        if team_assignments is None:
            team_assignments = {}
        for pid, cyl in cylinders.items():
            tidx = team_assignments.get(pid, 2)
            self._draw_cylinder_3d(canvas, cyl, rvec, tvec, K, tidx)
        # 4. Ball
        for sphere in spheres:
            self._draw_sphere_3d(canvas, sphere, rvec, tvec, K)
        # 5. HUD
        self._draw_hud(canvas, scores, frame_idx, fps)
        return canvas

    # ── Full video rendering ─────────────────────────────────────────────

    def render_video(
        self,
        video_path: str,
        output_path: str,
        skip_frames: int = 2,
        max_frames: Optional[int] = None,
        progress_callback=None,
        device: Optional[str] = None,
    ):
        """
        Process an entire video file end-to-end and write the primitive view
        as an MP4 file.

        device: YOLO inference device — 'cuda', 'cpu', or None for auto-detect.
        """
        import sys
        import time

        info = get_video_info(video_path)
        fps = info["fps"]
        fw, fh = info["width"], info["height"]
        total = info["total_frames"]
        if max_frames:
            total = min(total, max_frames)
        frames_to_process = total // max(skip_frames, 1)

        # Adopt the source video's native resolution
        self.w = fw
        self.h = fh
        self.court_lines.clear()
        self._build_default_court()

        try:
            import torch
            if torch.cuda.is_available():
                device_label = f"GPU ({torch.cuda.get_device_name(0)})"
            else:
                device_label = "CPU"
        except Exception:
            device_label = "CPU"

        print(f"  ⚡ Inference device: {device_label}")
        print(f"  📹 {fw}x{fh} @ {fps:.1f}fps, {total} total frames")
        print(f"  🎯 Processing every {skip_frames}th frame → ~{frames_to_process} frames")
        sys.stdout.flush()

        # Init detector, shot tracker, and rim detector
        detector = HoopDetector()
        shot_tracker = ShotTracker(fw, fh, fps=fps)
        rim_det = RimDetector(fw, fh)

        # Output writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / max(skip_frames, 1)
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (self.w, self.h))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frame_idx = 0
        processed = 0
        t_start = time.time()
        t_warmup_end = None
        warmup_count = 10  # Time first N frames to get rate estimate
        sec_per_frame = None

        while cap.isOpened() and frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # ── Detect with YOLO tracking ────────────────────────────────
            results = detector.model.track(
                frame, persist=True, classes=[0], verbose=False,
                conf=0.35, device=device,
            )

            player_capsules = {}
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, obj_id in zip(boxes, ids):
                    cap_prim = bbox_to_player(box[0], box[1], box[2], box[3], obj_id=obj_id)
                    player_capsules[obj_id] = cap_prim

            # ── Ball detection ───────────────────────────────────────────
            balls = detector.detect_ball(frame)

            # ── Rim detection (dynamic) ──────────────────────────────────
            detected_rims = rim_det.detect(frame)
            shot_tracker.update_rims(detected_rims)

            # ── Shot tracking ────────────────────────────────────────────
            shot_tracker.update(balls, player_capsules, frame_idx)
            scores = shot_tracker.get_total_scores()

            # ── Render the frame ─────────────────────────────────────────
            if self.mode == "3d":
                # Lift 2D → 3D using camera model
                if not hasattr(self, '_cam3d'):
                    # One-time init: create a Camera3D with default court calibration
                    self._cam3d = Camera3D(fw, fh)
                    # Use approximate court corners from frame edges
                    pixel_corners = [
                        (fw * 0.05, fh * 0.15),   # top-left
                        (fw * 0.95, fh * 0.15),   # top-right
                        (fw * 0.95, fh * 0.85),   # bottom-right
                        (fw * 0.05, fh * 0.85),   # bottom-left
                    ]
                    self._cam3d.calibrate_from_court_corners(pixel_corners)
                    self._rings_3d = [
                        self._cam3d.lift_rim(shot_tracker.rims[0], "left"),
                        self._cam3d.lift_rim(shot_tracker.rims[1], "right"),
                    ]

                # Lift players
                cylinders_3d = {}
                players_3d_list = []
                for pid, cap in player_capsules.items():
                    cyl = self._cam3d.lift_player(cap)
                    cylinders_3d[pid] = cyl
                    players_3d_list.append(cyl)

                # Lift balls
                spheres_3d = []
                for ball in balls:
                    s = self._cam3d.lift_ball(ball, players_3d_list)
                    spheres_3d.append(s)

                rendered = self.render_frame_3d(
                    spheres=spheres_3d,
                    cylinders=cylinders_3d,
                    rings=self._rings_3d,
                    scores=scores,
                    frame_idx=frame_idx,
                    fps=fps,
                    camera_preset=getattr(self, 'camera_preset', 'broadcast'),
                )
            else:
                rendered = self.render_frame(
                    original_frame=frame,
                    balls=balls,
                    player_capsules=player_capsules,
                    rims=shot_tracker.rims,
                    scores=scores,
                    frame_idx=frame_idx,
                    fps=fps,
                )

            writer.write(rendered)
            processed += 1

            # ── Warmup timing ────────────────────────────────────────────
            if processed == warmup_count:
                t_warmup_end = time.time()
                sec_per_frame = (t_warmup_end - t_start) / warmup_count
                est_total_sec = sec_per_frame * frames_to_process
                est_min = est_total_sec / 60
                print(f"\n  ⏱️  Warmup: {sec_per_frame:.2f}s/frame → "
                      f"estimated total: {est_min:.0f} min")
                sys.stdout.flush()

            # ── Progress bar ─────────────────────────────────────────────
            if processed % 100 == 0 or processed == frames_to_process:
                pct = min(100, processed * 100 // frames_to_process)
                elapsed = time.time() - t_start
                current_rate = elapsed / max(processed, 1)
                remaining = current_rate * (frames_to_process - processed)
                rem_min = int(remaining // 60)
                rem_sec = int(remaining % 60)
                fps_render = processed / max(elapsed, 0.01)

                # Terminal progress bar
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                line = (f"\r  [{bar}] {pct:3d}% | "
                        f"{processed}/{frames_to_process} frames | "
                        f"{fps_render:.1f} fps | "
                        f"ETA {rem_min:02d}:{rem_sec:02d}")
                sys.stdout.write(line)
                sys.stdout.flush()

                # Callback for server/API
                if progress_callback:
                    progress_callback(
                        min(99, pct),
                        f"Frame {processed}/{frames_to_process} | "
                        f"ETA {rem_min:02d}:{rem_sec:02d}"
                    )

            frame_idx += 1

        cap.release()
        writer.release()

        elapsed = time.time() - t_start
        print(f"\n\n  ✅ Done! {processed} frames in {elapsed/60:.1f} min "
              f"({processed/max(elapsed,0.01):.1f} fps)")
        print(f"  📁 Output: {output_path}")
        sys.stdout.flush()

        if progress_callback:
            progress_callback(100, "Render complete")

        return output_path

    # ── 3D vector data export ────────────────────────────────────────────

    def export_3d_data(
        self,
        video_path: str,
        output_path: str,
        skip_frames: int = 2,
        max_frames: Optional[int] = None,
        progress_callback=None,
        device: Optional[str] = None,
    ) -> str:
        """
        Process a video and export per-frame 3D primitive vectors as JSON.
        No video rendering — just the raw 3D coordinate data.

        Output JSON structure:
        {
          "video": "...",
          "resolution": [w, h],
          "fps": 30.0,
          "total_source_frames": 203098,
          "processed_frames": 67699,
          "camera": { "position": [x,y,z], "rvec": [...], "tvec": [...] },
          "rims": [ { "center": [x,y,z], "radius": 0.23 }, ... ],
          "frames": [
            {
              "idx": 0,
              "time_s": 0.0,
              "players": {
                "1": { "base": [x,y,z], "top": [x,y,z], "height": 1.9, "radius": 0.25 },
                ...
              },
              "ball": { "center": [x,y,z], "radius": 0.12 } | null,
              "scores": { "home_pts": 0, "away_pts": 0, ... },
              "events": [ ... ]
            },
            ...
          ]
        }
        """
        import json
        import sys
        import time

        info = get_video_info(video_path)
        fps = info["fps"]
        fw, fh = info["width"], info["height"]
        total = info["total_frames"]
        if max_frames:
            total = min(total, max_frames)
        frames_to_process = total // max(skip_frames, 1)

        # Init detector, shot tracker, and camera FIRST
        detector = HoopDetector()
        shot_tracker = ShotTracker(fw, fh, fps=fps)
        rim_det = RimDetector(fw, fh)
        cam = Camera3D(fw, fh)

        print(f"  ⚡ Inference device: {detector.device_name}")
        print(f"  📹 {fw}x{fh} @ {fps:.1f}fps, {total} total frames")
        print(f"  🎯 Exporting 3D vectors every {skip_frames}th frame → ~{frames_to_process} frames")
        sys.stdout.flush()

        # Calibrate camera from approximate court corners
        pixel_corners = [
            (fw * 0.05, fh * 0.15),
            (fw * 0.95, fh * 0.15),
            (fw * 0.95, fh * 0.85),
            (fw * 0.05, fh * 0.85),
        ]
        cam.calibrate_from_court_corners(pixel_corners)

        # Lift rims to 3D (fixed positions)
        rims_3d = [
            cam.lift_rim(shot_tracker.rims[0], "left"),
            cam.lift_rim(shot_tracker.rims[1], "right"),
        ]

        # Build output structure
        result = {
            "video": os.path.basename(video_path),
            "resolution": [fw, fh],
            "fps": fps,
            "total_source_frames": total,
            "frames_exported": 0,
            "skip_frames": skip_frames,
            "camera": {
                "position": list(cam.camera_pos.as_tuple()) if cam.camera_pos else None,
                "calibrated": cam.calibrated,
            },
            "rims": [
                {"center": list(r.center.as_tuple()), "radius": r.radius}
                for r in rims_3d
            ],
            "frames": [],
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frame_idx = 0
        processed = 0
        t_start = time.time()

        while cap.isOpened() and frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # ── Detect ────────────────────────────────────────────────────
            # Do not pass 'device' here. The detector handles its own engine
            # (PyTorch, ONNX+DirectML, or OpenVINO)
            results = detector.model.track(
                frame, persist=True, classes=[0], verbose=False,
                conf=0.35,
            )

            player_capsules = {}
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, obj_id in zip(boxes, ids):
                    cap_prim = bbox_to_player(box[0], box[1], box[2], box[3], obj_id=obj_id)
                    player_capsules[obj_id] = cap_prim

            # ── Ball detection ───────────────────────────────────────────
            balls = detector.detect_ball(frame)

            # ── Rim detection (dynamic) ──────────────────────────────────
            detected_rims = rim_det.detect(frame)
            shot_tracker.update_rims(detected_rims)

            # ── Shot tracking ─────────────────────────────────────────────
            events = shot_tracker.update(balls, player_capsules, frame_idx)
            scores = shot_tracker.get_total_scores()

            # ── Lift to 3D ────────────────────────────────────────────────
            players_3d = {}
            players_3d_list = []
            for pid, cap_prim in player_capsules.items():
                cyl = cam.lift_player(cap_prim)
                players_3d[str(pid)] = {
                    "base": list(cyl.base.as_tuple()),
                    "top": list(cyl.top.as_tuple()),
                    "height": round(cyl.height, 3),
                    "radius": round(cyl.radius, 3),
                }
                players_3d_list.append(cyl)

            ball_data = None
            if balls:
                s = cam.lift_ball(balls[0], players_3d_list)
                ball_data = {
                    "center": list(s.center.as_tuple()),
                    "radius": round(s.radius, 4),
                }

            # ── Build frame record ────────────────────────────────────────
            # Re-lift rims every frame since they are dynamic now
            frame_rims_3d = [
                cam.lift_rim(shot_tracker.rims[0], "left"),
                cam.lift_rim(shot_tracker.rims[1], "right"),
            ]
            
            frame_record = {
                "idx": frame_idx,
                "time_s": round(frame_idx / fps, 3),
                "rims": [
                    {"center": list(r.center.as_tuple()), "radius": r.radius}
                    for r in frame_rims_3d
                ],
                "players": players_3d,
                "ball": ball_data,
                "scores": scores,
            }
            if events:
                frame_record["events"] = [
                    {"basket": e[0], "pts": e[1], "team": e[2], "team_total": e[3]}
                    for e in events
                ]
            result["frames"].append(frame_record)
            processed += 1

            # ── Progress bar ──────────────────────────────────────────────
            if processed % 100 == 0 or processed == frames_to_process:
                pct = min(100, processed * 100 // frames_to_process)
                elapsed = time.time() - t_start
                current_rate = elapsed / max(processed, 1)
                remaining = current_rate * (frames_to_process - processed)
                rem_min = int(remaining // 60)
                rem_sec = int(remaining % 60)
                fps_rate = processed / max(elapsed, 0.01)

                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                line = (f"\r  [{bar}] {pct:3d}% | "
                        f"{processed}/{frames_to_process} frames | "
                        f"{fps_rate:.1f} fps | "
                        f"ETA {rem_min:02d}:{rem_sec:02d}")
                sys.stdout.write(line)
                sys.stdout.flush()

                if progress_callback:
                    progress_callback(
                        min(99, pct),
                        f"Frame {processed}/{frames_to_process} | "
                        f"ETA {rem_min:02d}:{rem_sec:02d}"
                    )

            frame_idx += 1

        cap.release()

        result["frames_exported"] = processed
        result["shot_summary"] = {
            "total_scores": scores,
            "completed_shots": len(shot_tracker.completed_shots),
        }

        # Write JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t_start
        print(f"\n\n  ✅ Done! {processed} frames in {elapsed/60:.1f} min "
              f"({processed/max(elapsed,0.01):.1f} fps)")
        print(f"  📁 Output: {output_path}")
        sys.stdout.flush()

        if progress_callback:
            progress_callback(100, "Export complete")

        return output_path
