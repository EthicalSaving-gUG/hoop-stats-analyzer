"""Tests for the RimDetector class."""
import pytest
import numpy as np
import cv2

from hoop_stats.rim_detector import RimDetector
from hoop_stats.primitives import Rim


@pytest.fixture
def blank_frame():
    """Returns a blank 1920x1080 BGR image."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_detector_init():
    det = RimDetector(1920, 1080, smoothing=0.5)
    assert det.frame_w == 1920
    assert det.frame_h == 1080
    assert det.smoothing == 0.5
    assert not det.detected
    assert len(det.get_rims()) == 0


def test_detect_empty_frame(blank_frame):
    det = RimDetector(1920, 1080)
    rims = det.detect(blank_frame)
    assert len(rims) == 0
    assert not det.detected


def draw_synthetic_rim(frame, cx, cy, r):
    """Draw a bright orange rim-like circle on the frame."""
    # BGR for orange is (0, 165, 255), let's use pure bright orange
    orange = (0, 128, 255)
    cv2.circle(frame, (cx, cy), r, orange, 4)
    # Fill context so contour area is valid
    # Actually, contourArea computes the area *inside* the outer boundary,
    # so a thick hollow circle will have a large area.
    # To be safe for minEnclosingCircle, let's just draw a filled circle for the test.
    cv2.circle(frame, (cx, cy), r, orange, -1)


def test_detect_synthetic_rim_left(blank_frame):
    det = RimDetector(1920, 1080, smoothing=1.0)
    draw_synthetic_rim(blank_frame, 200, 400, 25)
    
    rims = det.detect(blank_frame)
    assert len(rims) == 1
    assert det.detected
    
    rim = rims[0]
    assert abs(rim.center.x - 200) < 10
    assert abs(rim.center.y - 400) < 10
    assert abs(rim.radius - 25) < 10


def test_detect_synthetic_rims_both(blank_frame):
    det = RimDetector(1920, 1080, smoothing=1.0)
    draw_synthetic_rim(blank_frame, 200, 400, 25)  # left
    draw_synthetic_rim(blank_frame, 1700, 400, 25) # right
    
    rims = det.detect(blank_frame)
    assert len(rims) == 2
    assert det.detected
    
    # Rims should be sorted internally by left/right assigning
    left = next(r for r in rims if r.center.x < 960)
    right = next(r for r in rims if r.center.x > 960)
    
    assert abs(left.center.x - 200) < 10
    assert abs(right.center.x - 1700) < 10


def test_smoothing(blank_frame):
    det = RimDetector(1920, 1080, smoothing=0.5)
    
    # First frame: left rim at 200
    frame1 = blank_frame.copy()
    draw_synthetic_rim(frame1, 200, 400, 25)
    det.detect(frame1)
    rims = det.get_rims()
    assert abs(rims[0].center.x - 200) < 5
    
    # Second frame: rim moved to 220
    # With smoothing=0.5, new position should be 200*0.5 + 220*0.5 = 210
    frame2 = blank_frame.copy()
    draw_synthetic_rim(frame2, 220, 400, 25)
    det.detect(frame2)
    rims = det.get_rims()
    
    assert 205 < rims[0].center.x < 215


def test_confidence_decay(blank_frame):
    det = RimDetector(1920, 1080)
    
    # Detect rim -> confidence goes up
    frame_with_rim = blank_frame.copy()
    draw_synthetic_rim(frame_with_rim, 200, 400, 25)
    det.detect(frame_with_rim)
    
    conf_before = det.confidence[0]
    
    # Empty frame -> confidence decays
    det.detect(blank_frame)
    conf_after = det.confidence[0]
    
    assert conf_after < conf_before
