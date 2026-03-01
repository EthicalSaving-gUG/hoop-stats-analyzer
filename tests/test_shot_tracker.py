import pytest
from hoop_stats.shot_tracker import ShotTracker

@pytest.fixture
def detector():
    return ShotTracker(frame_w=1280, frame_h=720)

class TestShotTracker:
    def test_initial_score_is_zero(self, detector):
        totals = detector.get_total_scores()
        assert totals["left_basket_pts"] == 0
        assert totals["total_baskets"] == 0

    def test_update_no_ball_returns_empty(self, detector):
        events = detector.update([], {}, 0)
        assert events == []

    def test_shot_arc_detected(self):
        """Simulate a shot taken from the wing that rises and enters the hoop."""
        fw, fh = 1280, 720
        # left hoop is ~ x=38–256, y=180–396
        detector = ShotTracker(fw, fh)

        # Shooter is at x=500, y=500
        player_positions = {1: (500, 500, 5.0, 5.0)}
        
        # Frame 0-4: Ball rises (y decreases)
        y_vals = [400, 380, 360, 340, 320]
        x_vals = [500, 450, 400, 350, 300]
        for i in range(5):
            detector.update([(x_vals[i], y_vals[i])], player_positions, i)
            
        # At frame 4, a release should have been detected (5 monotonic rising frames)
        assert len(detector.active_shots) == 1
        shot = detector.active_shots[0]
        assert shot["shooter_id"] == 1
        assert shot["status"] == "rising"
        
        # Frame 5-6: Ball reaches apex
        detector.update([(250, 310)], player_positions, 5) # Apex!
        detector.update([(200, 310)], player_positions, 6)
        
        # Frame 7-8: Ball falls
        detector.update([(150, 320)], player_positions, 7)
        assert detector.active_shots[0]["status"] == "falling"
        
        # Frame 9: Ball enters left hoop (x=100, y=300 -> which fits into left zone x: 38-256, y: 180-396)
        events = detector.update([(100, 300)], player_positions, 8)
        
        # It should trigger a MADE shot
        assert len(events) == 1
        assert len(detector.completed_shots) == 1
        assert detector.completed_shots[0]["status"] == "made"
        assert detector.scores[0] == 2
        
    def test_missed_shot(self):
        """Simulate a shot that rises but falls below the hoop zone without entering it."""
        fw, fh = 1280, 720
        detector = ShotTracker(fw, fh)
        player_positions = {2: (900, 500, 10.0, 5.0)}
        
        # Rise
        y_vals = [400, 380, 360, 340, 320]
        x_vals = [900, 850, 800, 750, 700]
        for i in range(5):
            detector.update([(x_vals[i], y_vals[i])], player_positions, i)
            
        assert len(detector.active_shots) == 1
        
        # Fall quickly OUTSIDE the hoop boundaries
        # Left hoop ends at x=256. This falls at x=400, below y=396.
        for i in range(5, 25): # Loop longer to satisfy the "start_frame > 15" miss condition
            detector.update([(400, 300 + i*10)], player_positions, i)
            
        assert len(detector.active_shots) == 0
        assert len(detector.completed_shots) == 1
        assert detector.completed_shots[0]["status"] == "missed"
        assert detector.scores[0] == 0
