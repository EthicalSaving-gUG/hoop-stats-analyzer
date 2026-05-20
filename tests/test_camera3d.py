"""Tests for camera3d module — 3D camera calibration and primitive lifting."""
import pytest
import math
import numpy as np

from hoop_stats.primitives import (
    Point2D, Point3D, Ball, PlayerCapsule, Rim,
    Sphere3D, Cylinder3D, Ring3D,
)
from hoop_stats.camera3d import Camera3D, make_virtual_camera, COURT_WIDTH_M, COURT_DEPTH_M


@pytest.fixture
def calibrated_camera():
    """Camera calibrated with synthetic court corners."""
    cam = Camera3D(frame_w=1920, frame_h=1080)
    # Approximate pixel positions of court corners as seen in a broadcast view
    pixel_corners = [
        (96, 162),     # top-left  (5% × 15%)
        (1824, 162),   # top-right (95% × 15%)
        (1824, 918),   # bottom-right (95% × 85%)
        (96, 918),     # bottom-left (5% × 85%)
    ]
    cam.calibrate_from_court_corners(pixel_corners)
    return cam


class TestPoint3D:
    def test_distance(self):
        a = Point3D(0, 0, 0)
        b = Point3D(3, 4, 0)
        assert abs(a.distance_to(b) - 5.0) < 0.01

    def test_distance_3d(self):
        a = Point3D(1, 2, 3)
        b = Point3D(4, 6, 3)
        assert abs(a.distance_to(b) - 5.0) < 0.01

    def test_as_tuple(self):
        p = Point3D(1.0, 2.0, 3.0)
        assert p.as_tuple() == (1.0, 2.0, 3.0)

    def test_xz(self):
        p = Point3D(5.0, 10.0, 7.0)
        assert p.xz() == (5.0, 7.0)


class TestSphere3D:
    def test_create(self):
        s = Sphere3D(center=Point3D(1, 2, 3), radius=0.12)
        assert s.radius == 0.12
        assert s.center.y == 2


class TestCylinder3D:
    def test_top(self):
        c = Cylinder3D(base=Point3D(5, 0, 7), height=1.9, radius=0.25)
        top = c.top
        assert abs(top.y - 1.9) < 0.01
        assert top.x == 5

    def test_center(self):
        c = Cylinder3D(base=Point3D(5, 0, 7), height=2.0, radius=0.25)
        assert abs(c.center.y - 1.0) < 0.01


class TestRing3D:
    def test_create(self):
        r = Ring3D(center=Point3D(1.575, 3.05, 7.5), radius=0.2286)
        assert abs(r.center.y - 3.05) < 0.01
        assert r.normal == (0.0, 1.0, 0.0)


class TestCamera3DCalibration:
    def test_calibration_succeeds(self, calibrated_camera):
        assert calibrated_camera.calibrated is True
        assert calibrated_camera.camera_pos is not None

    def test_camera_position_reasonable(self, calibrated_camera):
        """Camera should be above the court (positive Y) and behind/in-front of it."""
        pos = calibrated_camera.camera_pos
        assert pos.y > 0, "Camera should be above ground"

    def test_uncalibrated_fallback(self):
        cam = Camera3D(1920, 1080)
        pt = cam.pixel_to_ground(960, 540)
        # Should return some position without crashing
        assert isinstance(pt, Point3D)


class TestPixelToGround:
    def test_center_pixel_maps_to_court_center(self, calibrated_camera):
        """Middle of the frame should map roughly to the court center."""
        pt = calibrated_camera.pixel_to_ground(960, 540)
        assert 5 < pt.x < 23, f"X={pt.x} should be near court center"
        assert 2 < pt.z < 13, f"Z={pt.z} should be near court center"
        assert abs(pt.y) < 0.01, "Y should be 0 (ground plane)"


class TestHeightEstimation:
    def test_positive_height(self, calibrated_camera):
        h = calibrated_camera.estimate_height_m(
            foot_px=(960, 700),
            head_px=(960, 400),
        )
        assert h > 0, f"Height {h}m should be positive"


class TestLiftPlayer:
    def test_lift_returns_cylinder(self, calibrated_camera):
        cap = PlayerCapsule(
            head=Point2D(500, 300),
            foot=Point2D(500, 600),
            width=60,
            obj_id=7,
        )
        cyl = calibrated_camera.lift_player(cap)
        assert isinstance(cyl, Cylinder3D)
        assert cyl.obj_id == 7
        assert cyl.height > 0
        assert abs(cyl.base.y) < 0.01  # on the ground


class TestLiftBall:
    def test_lift_returns_sphere(self, calibrated_camera):
        ball = Ball(center=Point2D(960, 400), radius=15)
        s = calibrated_camera.lift_ball(ball)
        assert isinstance(s, Sphere3D)
        assert abs(s.radius - 0.12) < 0.01

    def test_lift_with_nearby_players(self, calibrated_camera):
        ball = Ball(center=Point2D(500, 350), radius=12)
        players = [Cylinder3D(base=Point3D(10, 0, 7), height=1.9, radius=0.25)]
        s = calibrated_camera.lift_ball(ball, players)
        assert isinstance(s, Sphere3D)


class TestLiftRim:
    def test_left_rim(self, calibrated_camera):
        rim_2d = Rim(center=Point2D(200, 400), radius=50)
        ring = calibrated_camera.lift_rim(rim_2d, side="left")
        assert isinstance(ring, Ring3D)
        assert abs(ring.center.y - 3.05) < 0.01
        assert ring.center.x < COURT_WIDTH_M / 2

    def test_right_rim(self, calibrated_camera):
        rim_2d = Rim(center=Point2D(1700, 400), radius=50)
        ring = calibrated_camera.lift_rim(rim_2d, side="right")
        assert ring.center.x > COURT_WIDTH_M / 2


class TestProjection:
    def test_project_ground_point(self, calibrated_camera):
        """A 3D point on the court should project to a valid pixel."""
        pt = Point3D(14.0, 0.0, 7.5)  # court center on ground
        px, py = calibrated_camera.project(pt)
        assert 0 < px < 1920
        assert 0 < py < 1080


class TestVirtualCamera:
    @pytest.mark.parametrize("preset", ["birdseye", "sideline", "corner", "broadcast"])
    def test_preset_creates_valid_params(self, preset):
        rvec, tvec, K = make_virtual_camera(preset, 1280, 720)
        assert rvec.shape == (3, 1)
        assert tvec.shape == (3, 1)
        assert K.shape == (3, 3)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            make_virtual_camera("nonexistent")
