import pytest
from hoop_stats.shot_tracker import ShotTracker
from hoop_stats.primitives import Ball, Point2D, PlayerCapsule


def test_get_rim_focus_boxes():
    tracker = ShotTracker(1280, 720)
    boxes = tracker.get_rim_focus_boxes()
    assert len(boxes) == 2
    # Check that each box is valid within frame bounds
    for (x1, y1, x2, y2) in boxes:
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720
        assert (x2 - x1) > 100
        assert (y2 - y1) > 100


def test_detect_free_throw_formation():
    tracker = ShotTracker(1280, 720)
    right_rim = tracker.rims[1]

    # Create shooter positioned at free-throw distance in front of rim
    shooter = PlayerCapsule(
        head=Point2D(right_rim.center.x - 200, right_rim.center.y - 50),
        foot=Point2D(right_rim.center.x - 200, right_rim.center.y + 50),
        width=50,
        obj_id=20
    )

    # Rebounders along key lanes
    reb1 = PlayerCapsule(
        head=Point2D(right_rim.center.x - 100, right_rim.center.y - 120),
        foot=Point2D(right_rim.center.x - 100, right_rim.center.y - 20),
        width=50,
        obj_id=14
    )
    reb2 = PlayerCapsule(
        head=Point2D(right_rim.center.x - 100, right_rim.center.y + 120),
        foot=Point2D(right_rim.center.x - 100, right_rim.center.y + 220),
        width=50,
        obj_id=11
    )

    capsules = {20: shooter, 14: reb1, 11: reb2}
    detected = tracker.detect_free_throw_formation(capsules, frame_idx=100)
    assert detected is True
    assert tracker.is_free_throw_mode is True
