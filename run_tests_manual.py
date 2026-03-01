import pytest
from tests.test_shot_tracker import TestShotTracker

t = TestShotTracker()
from hoop_stats.shot_tracker import ShotTracker
det = ShotTracker(1280, 720)

print("Running test_initial_score_is_zero")
t.test_initial_score_is_zero(det)

print("Running test_update_no_ball_returns_empty")
t.test_update_no_ball_returns_empty(det)

print("Running test_shot_arc_detected")
t.test_shot_arc_detected()

print("Running test_missed_shot")
t.test_missed_shot()

print("ALL PASSED!")
