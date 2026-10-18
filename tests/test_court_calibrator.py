import pytest
import numpy as np
import cv2
from hoop_stats.court_calibrator import AutomatedCourtCalibrator, CourtCalibrationResult, HAS_SUPERVISION


def test_court_calibrator_init():
    calibrator = AutomatedCourtCalibrator(court_width_m=28.0, court_depth_m=15.0)
    assert calibrator.court_width_m == 28.0
    assert calibrator.court_depth_m == 15.0
    assert len(calibrator.canonical_points_m) == 6


def test_detect_court_mask():
    calibrator = AutomatedCourtCalibrator()
    # Create dark frame with white lines
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(frame, (50, 50), (590, 50), (255, 255, 255), 4)

    mask = calibrator.detect_court_mask(frame)
    assert mask.shape == (480, 640)
    assert np.any(mask > 0)


def test_calibrate_from_points():
    calibrator = AutomatedCourtCalibrator()
    pixel_points = [
        (50.0, 50.0),
        (590.0, 50.0),
        (590.0, 430.0),
        (50.0, 430.0),
    ]
    res = calibrator.calibrate_from_points(pixel_points)
    assert res.success is True
    assert res.homography is not None
    assert res.homography.shape == (3, 3)
    assert res.confidence > 0.0
    assert len(res.detected_keypoints) == 4


def test_auto_calibrate_fallback():
    calibrator = AutomatedCourtCalibrator()
    # Completely black frame should trigger heuristic fallback safely
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = calibrator.auto_calibrate(blank_frame)
    assert res.success is True
    assert res.homography is not None
    assert res.homography.shape == (3, 3)


def test_create_supervision_zones():
    if not HAS_SUPERVISION:
        pytest.skip("supervision library not installed")

    calibrator = AutomatedCourtCalibrator()
    pixel_points = [
        (100.0, 80.0),
        (1180.0, 80.0),
        (1180.0, 640.0),
        (100.0, 640.0),
    ]
    res = calibrator.calibrate_from_points(pixel_points)
    assert res.success is True

    zones = calibrator.create_supervision_zones(res.homography, frame_resolution_wh=(1280, 720))
    assert "paint_left" in zones
    assert "paint_right" in zones
    assert "court_full" in zones
    assert zones["paint_left"] is not None
