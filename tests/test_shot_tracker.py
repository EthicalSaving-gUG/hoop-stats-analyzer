import pytest
from hoop_stats.shot_tracker import ShotTracker, ShotTrackerConfig
from hoop_stats.primitives import Ball, Point2D, PlayerCapsule

@pytest.fixture
def detector():
    return ShotTracker(frame_w=1280, frame_h=720)


def _make_ball(x, y, r=10):
    """Helper to create a Ball primitive."""
    return Ball(center=Point2D(x, y), radius=r)


def _make_capsule(px, py, mx, my, obj_id=None, width=50):
    """Helper to create a PlayerCapsule with meter_pos set."""
    cap = PlayerCapsule(
        head=Point2D(px, py - 100),
        foot=Point2D(px, py),
        width=width,
        obj_id=obj_id,
        meter_pos=Point2D(mx, my),
    )
    return cap


class TestShotTracker:
    def test_initial_score_is_zero(self, detector):
        totals = detector.get_total_scores()
        assert totals["home_pts"] == 0
        assert totals["away_pts"] == 0
        assert totals["total_baskets"] == 0

    def test_update_no_ball_returns_empty(self, detector):
        events = detector.update([], {}, 0)
        assert events == []

    def test_shot_arc_detected(self):
        """Simulate a shot taken from the wing that rises and enters the hoop."""
        fw, fh = 1280, 720
        detector = ShotTracker(fw, fh)

        player_capsules = {1: _make_capsule(500, 500, 5.0, 5.0, obj_id=1)}
        
        # Frame 0-4: Ball rises (y decreases)
        y_vals = [400, 380, 360, 340, 320]
        x_vals = [500, 450, 400, 350, 300]
        for i in range(5):
            detector.update([_make_ball(x_vals[i], y_vals[i])], player_capsules, i)
            
        assert len(detector.active_shots) == 1
        shot = detector.active_shots[0]
        assert shot["shooter_id"] == 1
        assert shot["status"] == "rising"
        
        # Frame 5: Ball reaches apex
        detector.update([_make_ball(250, 300)], player_capsules, 5)
        
        # Frame 6-7: Ball starts falling
        detector.update([_make_ball(200, 305)], player_capsules, 6)
        detector.update([_make_ball(150, 315)], player_capsules, 7)
        assert detector.active_shots[0]["status"] == "falling"
        
        # Frame 8: Ball enters target rim zone
        # Ball starts at x=500 < frame_w/2=640, so target_hoop=1 (right rim)
        rim = detector.rims[1]
        events = detector.update(
            [_make_ball(rim.center.x, rim.center.y)], player_capsules, 8
        )
        
        assert len(events) == 1
        assert len(detector.completed_shots) == 1
        assert detector.completed_shots[0]["status"] == "made"
        assert detector.completed_shots[0]["team"] == "home"  # right basket = home before halftime
        assert detector.scores[1] == 2
        assert detector.team_scores["home"] == 2
        
    def test_missed_shot(self):
        """Simulate a shot that rises but falls below the hoop zone without entering it."""
        fw, fh = 1280, 720
        detector = ShotTracker(fw, fh)
        player_capsules = {2: _make_capsule(900, 500, 10.0, 5.0, obj_id=2)}
        
        # Rise
        y_vals = [400, 380, 360, 340, 320]
        x_vals = [900, 850, 800, 750, 700]
        for i in range(5):
            detector.update([_make_ball(x_vals[i], y_vals[i])], player_capsules, i)
            
        assert len(detector.active_shots) == 1
        
        # Fall quickly OUTSIDE the hoop boundaries
        rim_bottom = detector._rim_bottom_y[1]
        for i in range(5, 25):
            detector.update(
                [_make_ball(400, rim_bottom + (i - 5) * 10)],
                player_capsules, i
            )
            
        assert len(detector.active_shots) == 0
        # completed_shots may include the missed shot
        missed = [s for s in detector.completed_shots if s["status"] == "missed"]
        assert len(missed) == 1
        assert detector.team_scores["home"] == 0
        assert detector.team_scores["away"] == 0


class TestHalftimeSwap:
    def test_swap_sides(self):
        detector = ShotTracker(frame_w=1280, frame_h=720)
        assert detector._home_basket == 1
        assert detector._away_basket == 0
        
        detector.swap_sides()
        
        assert detector._home_basket == 0
        assert detector._away_basket == 1
        assert detector._sides_swapped is True

    def test_basket_to_team_before_swap(self):
        detector = ShotTracker(frame_w=1280, frame_h=720)
        # Before halftime: basket 0 = away, basket 1 = home
        assert detector._basket_to_team(0) == "away"
        assert detector._basket_to_team(1) == "home"

    def test_basket_to_team_after_swap(self):
        detector = ShotTracker(frame_w=1280, frame_h=720)
        detector.swap_sides()
        # After halftime: basket 0 = home, basket 1 = away
        assert detector._basket_to_team(0) == "home"
        assert detector._basket_to_team(1) == "away"

    def test_auto_halftime_trigger(self):
        """Halftime should auto-trigger at the configured frame."""
        detector = ShotTracker(frame_w=1280, frame_h=720, config=ShotTrackerConfig(halftime_frame=100))
        assert detector._halftime_triggered is False
        
        # Process frames up to halftime
        for i in range(101):
            detector.update([], {}, i)
        
        assert detector._halftime_triggered is True
        assert detector._sides_swapped is True
        # Should have a HALFTIME event in completed_shots
        halftime_events = [s for s in detector.completed_shots if s.get("event") == "HALFTIME"]
        assert len(halftime_events) == 1

    def test_scores_correct_across_halftime(self):
        """
        Score into right basket before and after halftime.
        Before halftime: right basket = home team.
        After halftime:  right basket = away team.
        Total should be 2 home + 2 away.
        """
        fw, fh = 1280, 720
        detector = ShotTracker(fw, fh, config=ShotTrackerConfig(halftime_frame=50))
        
        # --- 1st half: score into right basket (home scores) ---
        player_capsules = {1: _make_capsule(500, 500, 5.0, 5.0, obj_id=1)}
        # Rise
        for i in range(5):
            detector.update([_make_ball(500 - i*50, 400 - i*20)], player_capsules, i)
        # Apex + fall
        detector.update([_make_ball(250, 300)], player_capsules, 5)
        detector.update([_make_ball(200, 305)], player_capsules, 6)
        detector.update([_make_ball(150, 315)], player_capsules, 7)
        # Hit the right rim
        rim = detector.rims[1]
        detector.update([_make_ball(rim.center.x, rim.center.y)], player_capsules, 8)
        
        assert detector.team_scores["home"] == 2
        assert detector.team_scores["away"] == 0
        
        # --- Halftime passes at frame 50 ---
        for i in range(9, 55):
            detector.update([], {}, i)
        
        assert detector._sides_swapped is True
        
        # --- 2nd half: score into right basket AGAIN (now away scores) ---
        # Clear ball history for a clean arc
        detector.ball_history.clear()
        detector.ball_frame_history.clear()
        detector.active_shots.clear()
        detector.cooldown = {0: 0, 1: 0}
        
        for i in range(5):
            detector.update([_make_ball(500 - i*50, 400 - i*20)], player_capsules, 60 + i)
        detector.update([_make_ball(250, 300)], player_capsules, 65)
        detector.update([_make_ball(200, 305)], player_capsules, 66)
        detector.update([_make_ball(150, 315)], player_capsules, 67)
        detector.update([_make_ball(rim.center.x, rim.center.y)], player_capsules, 68)
        
        # Now the right basket belongs to away team
        assert detector.team_scores["home"] == 2
        assert detector.team_scores["away"] == 2

    def test_get_total_scores_includes_team_data(self):
        detector = ShotTracker(frame_w=1280, frame_h=720)
        totals = detector.get_total_scores()
        assert "home_pts" in totals
        assert "away_pts" in totals
        assert "sides_swapped" in totals
        assert totals["sides_swapped"] is False
