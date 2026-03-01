"""test_scoring.py — tests for hoop_stats.scoring (ScoringDetector)"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from hoop_stats.scoring import ScoringDetector


@pytest.fixture
def detector():
    return ScoringDetector(frame_w=1280, frame_h=720)


class TestScoringDetector:
    def test_initial_score_is_zero(self, detector):
        totals = detector.get_total_scores()
        assert totals["left_basket_pts"] == 0
        assert totals["right_basket_pts"] == 0
        assert totals["total_baskets"] == 0

    def test_update_no_ball_returns_empty(self, detector):
        events = detector.update([], 0)
        assert events == []

    def test_score_increments_on_ball_drop(self):
        """
        Simulate a ball dropping through the left hoop zone:
        1. Ball appears above zone → sets last_ball_y
        2. Ball appears inside zone → triggers a score.
        """
        fw, fh = 1280, 720

        # Build a model that returns a ball positioned above-then-inside zone
        left_zone = (
            int(fw * 0.03), int(fh * 0.25),
            int(fw * 0.20), int(fh * 0.55),
        )
        zx_mid = (left_zone[0] + left_zone[2]) // 2

        detector = ScoringDetector(fw, fh)

        # Frame 0: ball is above zone
        above_y = left_zone[1] - 20
        detector.update([(zx_mid, above_y)], 0)

        # Frame 1: ball enters zone → score
        inside_y = (left_zone[1] + left_zone[3]) // 2
        events = detector.update([(zx_mid, inside_y)], 1)

        assert len(events) == 1
        assert detector.scores[0] == 2  # 2 pts

    def test_cooldown_prevents_double_counting(self):
        """After a score, the same zone should not score again immediately."""
        fw, fh = 1280, 720
        left_zone = (
            int(fw * 0.03), int(fh * 0.25),
            int(fw * 0.20), int(fh * 0.55),
        )
        zx_mid = (left_zone[0] + left_zone[2]) // 2

        detector = ScoringDetector(fw, fh, cooldown_frames=45)

        above_y  = left_zone[1] - 20
        inside_y = (left_zone[1] + left_zone[3]) // 2

        detector.update([(zx_mid, above_y)], 0)
        detector.update([(zx_mid, inside_y)], 1)

        first_score = detector.scores[0]

        # Immediately repeat — no score increase expected (cooldown active)
        detector.update([(zx_mid, above_y)], 2)
        detector.update([(zx_mid, inside_y)], 3)

        assert detector.scores[0] == first_score  # unchanged
