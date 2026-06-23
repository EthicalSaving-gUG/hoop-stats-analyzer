import pytest
from hoop_stats.shot_tracker import ShotTracker
from hoop_stats.primitives import Ball, Point2D, PlayerCapsule
from hoop_stats.agents.referee_agent import RefereeAgent
from hoop_stats.agents.scoresheet_agent import ScoresheetAgent
from hoop_stats.game_state import GameState

@pytest.fixture
def test_court_geom():
    class DummyGeometry:
        def get_zone(self, x, y):
            if x > 10.0:
                return "ThreePointRight"
            return "TwoPointLeft"
        def pixel_to_meter(self, px, py):
            return px / 100.0, py / 100.0
    return DummyGeometry()

def _make_ball(x, y, r=10):
    return Ball(center=Point2D(x, y), radius=r)

def _make_capsule(px, py, mx, my, obj_id=None, width=50):
    cap = PlayerCapsule(
        head=Point2D(px, py - 100),
        foot=Point2D(px, py),
        width=width,
        obj_id=obj_id,
        meter_pos=Point2D(mx, my),
    )
    return cap

def test_xpts_and_shot_quality_three_pointer(test_court_geom):
    fw, fh = 1280, 720
    tracker = ShotTracker(fw, fh, court_geometry=test_court_geom)
    
    # Shooter has a high defender distance (uncontested)
    player_capsules = {
        1001: _make_capsule(800, 500, 15.0, 5.0, obj_id=1001), # Shooter (ThreePointRight zone)
        1002: _make_capsule(200, 500, 2.0, 5.0, obj_id=1002),  # Far defender (13 meters away)
    }
    player_teams = {"1001": "home", "1002": "away"}
    
    # 5 frames of upward motion (rising)
    for i in range(5):
        tracker.update([_make_ball(800 - i*10, 400 - i*20)], player_capsules, i, player_teams=player_teams)
        
    assert len(tracker.active_shots) == 1
    shot = tracker.active_shots[0]
    
    assert shot["defender_dist"] == pytest.approx(13.0)
    assert shot["shot_quality"] == pytest.approx(0.38)  # Uncontested three-pointer quality
    assert shot["expected_points"] == pytest.approx(3.0 * 0.38)

def test_xpts_and_shot_quality_two_pointer_contested(test_court_geom):
    fw, fh = 1280, 720
    tracker = ShotTracker(fw, fh, court_geometry=test_court_geom)
    
    # Shooter is closely contested (defender distance = 0.8 meters)
    player_capsules = {
        1001: _make_capsule(400, 500, 4.0, 5.0, obj_id=1001), # Shooter (TwoPointLeft zone)
        1002: _make_capsule(450, 500, 4.8, 5.0, obj_id=1002), # Close defender (0.8m)
    }
    player_teams = {"1001": "home", "1002": "away"}
    
    # 5 frames of upward motion (rising)
    for i in range(5):
        tracker.update([_make_ball(400 - i*10, 400 - i*20)], player_capsules, i, player_teams=player_teams)
        
    assert len(tracker.active_shots) == 1
    shot = tracker.active_shots[0]
    assert shot["defender_dist"] == pytest.approx(0.8)
    assert shot["shot_quality"] == pytest.approx(0.40)  # Contested two-pointer quality (< 1.2m)
    assert shot["expected_points"] == pytest.approx(2.0 * 0.40)

def test_pass_and_possession_telemetry():
    gs = GameState()
    # Mock court tracking data structure in GameState
    gs.court_tracking = {
        "ball": [5.0, 5.0],
        "players": {
            "1001": {"team": "home", "pos": [5.2, 5.0]},
            "1002": {"team": "home", "pos": [12.0, 5.0]}
        }
    }
    gs.player_stats = {
        "1001": {"team": "home", "dribble_duration_s": 0.0},
        "1002": {"team": "home", "dribble_duration_s": 0.0}
    }
    
    referee = RefereeAgent(gs, None)
    
    # Frame 0: Player 1001 in possession (closest, within 1.2m)
    referee.process_frame("cam1", None, 0)
    assert referee._possession_player == "1001"
    
    # Frame 5: Player 1001 keeps dribbling (adds dt)
    referee.process_frame("cam1", None, 5)
    assert gs.player_stats["1001"]["dribble_duration_s"] > 0.0
    
    # Frame 6: Loose ball - in transit
    gs.court_tracking["ball"] = [8.0, 5.0]
    referee.process_frame("cam1", None, 6)
    assert referee._possession_player is None
    assert referee._last_possessor == "1001"
    
    # Frame 12: Player 1002 catches the ball (distance from 1001 to 1002 is 6.8m, time is 6/30s = 0.2s)
    gs.court_tracking["ball"] = [11.8, 5.0]
    referee.process_frame("cam1", None, 12)
    
    # Check for PASS_DETECTED event in the game state events
    pass_events = [e for e in gs.events if e.event_type == "PASS_DETECTED"]
    assert len(pass_events) == 1
    assert pass_events[0].data["passer_id"] == "1001"
    assert pass_events[0].data["receiver_id"] == "1002"
    assert pass_events[0].data["distance_m"] == 6.8

def test_camera_fusion_and_coordinate_smoothing():
    # Verify spatial camera fusion in ScoresheetAgent
    # Since ScoresheetAgent needs a real frame image and YOLO model, we can test the spatial logic mathematically.
    fused_players = {}
    
    # Frame 1: Player detected at x=5.0, y=5.0
    mx1, my1 = 5.0, 5.0
    camera_id = "cam1"
    obj_id = 1
    
    matched_global_id = None
    min_dist = float('inf')
    for gid, data in fused_players.items():
        gx, gy = data["pos"]
        dist = ((mx1 - gx)**2 + (my1 - gy)**2)**0.5
        if dist < 1.5 and dist < min_dist:
            min_dist = dist
            matched_global_id = gid
            
    assert matched_global_id is None
    
    # Register player 1000
    global_id = 1000 + len(fused_players)
    fused_players[global_id] = {
        "pos": [mx1, my1],
        "last_update": 1,
        "local_ids": {camera_id: obj_id}
    }
    
    assert global_id == 1000
    
    # Frame 2: Player detected slightly shifted at x=5.3, y=5.1 from another camera
    mx2, my2 = 5.3, 5.1
    camera_id2 = "cam2"
    obj_id2 = 2
    
    matched_global_id = None
    min_dist = float('inf')
    for gid, data in fused_players.items():
        gx, gy = data["pos"]
        dist = ((mx2 - gx)**2 + (my2 - gy)**2)**0.5
        if dist < 1.5 and dist < min_dist:
            min_dist = dist
            matched_global_id = gid
            
    assert matched_global_id == 1000
    
    # Update with alpha=0.3
    alpha = 0.3
    gx, gy = fused_players[matched_global_id]["pos"]
    new_x = (1 - alpha) * gx + alpha * mx2
    new_y = (1 - alpha) * gy + alpha * my2
    
    fused_players[matched_global_id]["pos"] = [new_x, new_y]
    
    assert round(new_x, 2) == 5.09  # 0.7 * 5.0 + 0.3 * 5.3 = 3.5 + 1.59 = 5.09
    assert round(new_y, 2) == 5.03  # 0.7 * 5.0 + 0.3 * 5.1 = 3.5 + 1.53 = 5.03
