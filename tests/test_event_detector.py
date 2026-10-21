import pytest
from hoop_stats.event_detector import EventDetector
from hoop_stats.player_registry import PlayerRegistry
from hoop_stats.primitives import Ball, PlayerCapsule, Point2D


def create_player(pid: int, x: float, y: float, w: float = 30.0, h: float = 80.0):
    return PlayerCapsule(
        head=Point2D(x, y - h / 2),
        foot=Point2D(x, y + h / 2),
        width=w,
        obj_id=pid
    )


def test_traveling_detection_without_dribble():
    """Player moving > 1.8m while holding ball with 0 dribbles triggers Traveling."""
    registry = PlayerRegistry(roster_home=["20"])
    detector = EventDetector(
        frame_w=1280,
        frame_h=720,
        player_registry=registry,
        traveling_dist_m=1.8,
        possession_radius_px=60.0
    )
    teams = {"5": "home"}
    registry.register_track(5, team="home")
    registry.bind_jersey(5, "20", team="home")

    # Frame 1 to 5: Player 5 catches ball at (500, 300)
    for f in range(1, 6):
        ball = Ball(center=Point2D(500, 300), radius=10)
        p5 = create_player(5, 500, 300, h=80.0)
        evts = detector.update([ball], {5: p5}, frame_idx=f, player_teams=teams)
        assert len(evts) == 0

    # Frames 6 to 25: Player 5 walks to (620, 300) (displacement 120px / 80px * 1.85m ≈ 2.77m) without bouncing
    detected_travel = False
    for f in range(6, 26):
        # Move 6px per frame
        px = 500 + (f - 5) * 6
        ball = Ball(center=Point2D(px, 300), radius=10)
        p5 = create_player(5, px, 300, h=80.0)
        evts = detector.update([ball], {5: p5}, frame_idx=f, player_teams=teams)
        if any(e["type"] == "TRAVELING" for e in evts):
            detected_travel = True

    assert detected_travel is True
    player = registry.canonical_players["20"]
    assert player.turnovers == 1
    assert player.turnovers_traveling == 1


def test_dribble_bounces_reset_traveling():
    """When the player dribbles (vertical ball oscillation), traveling distance resets."""
    registry = PlayerRegistry(roster_home=["14"])
    detector = EventDetector(
        frame_w=1280,
        frame_h=720,
        player_registry=registry,
        traveling_dist_m=1.8,
        possession_radius_px=60.0
    )
    teams = {"3": "home"}
    registry.register_track(3, team="home")
    registry.bind_jersey(3, "14", team="home")

    # Initial frame
    ball = Ball(center=Point2D(400, 280), radius=10)
    p3 = create_player(3, 400, 300, h=80.0)
    detector.update([ball], {3: p3}, frame_idx=1, player_teams=teams)

    # Simulate dribble: ball moves down to floor (y=340) then bounces back up to (y=290)
    # Downward
    detector.update([Ball(center=Point2D(405, 310), radius=10)], {3: create_player(3, 405, 300, h=80.0)}, frame_idx=2, player_teams=teams)
    detector.update([Ball(center=Point2D(410, 340), radius=10)], {3: create_player(3, 410, 300, h=80.0)}, frame_idx=3, player_teams=teams)
    # Upward bounce
    detector.update([Ball(center=Point2D(415, 310), radius=10)], {3: create_player(3, 415, 300, h=80.0)}, frame_idx=4, player_teams=teams)
    
    assert detector.dribble_bounces >= 1
    assert detector.handler_accumulated_dist_m == 0.0


def test_bad_pass_interception_and_steal():
    """A pass intercepted by opposing team records Turnover for passer and Steal for interceptor."""
    registry = PlayerRegistry(roster_home=["20"], roster_away=["8"])
    detector = EventDetector(frame_w=1280, frame_h=720, player_registry=registry)
    teams = {"1": "home", "2": "away"}
    registry.register_track(1, team="home")
    registry.bind_jersey(1, "20", team="home")
    registry.register_track(2, team="away")
    registry.bind_jersey(2, "8", team="away")

    # Player 1 has ball at (200, 300)
    p1 = create_player(1, 200, 300)
    p2 = create_player(2, 600, 300)
    capsules = {1: p1, 2: p2}

    detector.update([Ball(center=Point2D(200, 300), radius=10)], capsules, frame_idx=10, player_teams=teams)
    assert detector.current_handler_id == 1

    # Ball leaves Player 1 towards Player 2
    detector.update([Ball(center=Point2D(350, 300), radius=10)], capsules, frame_idx=15, player_teams=teams)
    assert detector.active_pass is not None
    assert detector.active_pass["passer_id"] == 1

    # Ball arrives at Defender 2 (Steal / Fehlpass abgefangen)
    evts = detector.update([Ball(center=Point2D(600, 300), radius=10)], capsules, frame_idx=20, player_teams=teams)
    
    assert len(evts) == 1
    assert evts[0]["type"] == "FEHLPASS_STEAL"
    assert evts[0]["passer_id"] == 1
    assert evts[0]["stealer_id"] == 2

    # Check player stat sheets
    passer = registry.canonical_players["20"]
    assert passer.turnovers == 1
    assert passer.turnovers_bad_pass == 1

    defender = registry.canonical_players["8"]
    assert defender.steals == 1


def test_bad_pass_out_of_bounds():
    """Ball passed out of bounds triggers Turnover for passer."""
    registry = PlayerRegistry(roster_home=["11"])
    detector = EventDetector(frame_w=1280, frame_h=720, player_registry=registry)
    teams = {"4": "home"}
    registry.register_track(4, team="home")
    registry.bind_jersey(4, "11", team="home")

    p4 = create_player(4, 300, 300)
    # Player 4 has ball
    detector.update([Ball(center=Point2D(300, 300), radius=10)], {4: p4}, frame_idx=50, player_teams=teams)
    
    # Ball leaves player towards sideline
    detector.update([Ball(center=Point2D(150, 150), radius=10)], {4: p4}, frame_idx=55, player_teams=teams)
    assert detector.active_pass is not None

    # Ball crosses out of bounds (margin < 40px)
    evts = detector.update([Ball(center=Point2D(20, 20), radius=10)], {4: p4}, frame_idx=62, player_teams=teams)
    assert len(evts) == 1
    assert evts[0]["type"] == "FEHLPASS_AUS"
    assert evts[0]["player_id"] == 4

    player = registry.canonical_players["11"]
    assert player.turnovers == 1


def test_assist_on_made_shot():
    """Teammate pass leading to made shot awards Assist."""
    registry = PlayerRegistry(roster_home=["14", "20"])
    detector = EventDetector(frame_w=1280, frame_h=720, player_registry=registry)
    teams = {"1": "home", "2": "home"}
    registry.register_track(1, team="home")
    registry.bind_jersey(1, "14", team="home")
    registry.register_track(2, team="home")
    registry.bind_jersey(2, "20", team="home")

    p1 = create_player(1, 300, 300)
    p2 = create_player(2, 500, 300)
    capsules = {1: p1, 2: p2}

    # Player 1 has ball
    detector.update([Ball(center=Point2D(300, 300), radius=10)], capsules, frame_idx=100, player_teams=teams)
    # Ball in flight towards teammate 2
    detector.update([Ball(center=Point2D(400, 300), radius=10)], capsules, frame_idx=105, player_teams=teams)
    # Teammate 2 catches ball
    detector.update([Ball(center=Point2D(500, 300), radius=10)], capsules, frame_idx=110, player_teams=teams)

    assert detector.last_completed_pass is not None
    assert detector.last_completed_pass["passer_id"] == 1
    assert detector.last_completed_pass["receiver_id"] == 2

    # Player 2 makes a shot within 2 seconds (frame 140)
    assist_evt = detector.check_assist_on_made_shot(shooter_id=2, frame_idx=140)
    assert assist_evt is not None
    assert assist_evt["type"] == "ASSIST"
    assert assist_evt["passer_id"] == 1

    player14 = registry.canonical_players["14"]
    assert player14.assists == 1


def test_player_out_of_bounds_floor_contact():
    """Ballhandler stepping on boundary line triggers PLAYER_OUT_OF_BOUNDS turnover (FIBA Art. 23)."""
    from hoop_stats.geometry import CourtGeometry
    geo = CourtGeometry(config_path="non_existent.json")
    registry = PlayerRegistry(roster_home=["20"])
    detector = EventDetector(frame_w=1280, frame_h=720, court_geometry=geo, player_registry=registry, fps=30.0)
    teams = {"5": "home"}
    registry.register_track(5, team="home")
    registry.bind_jersey(5, "20", team="home")

    # Player 5 with ball steps out of bounds at baseline (x=0.02m <= 0.05m line)
    # Using 1:1 meter coordinates for unit test
    p5 = create_player(5, 0.02, 7.5, w=0.5, h=1.8)
    ball = Ball(center=Point2D(0.02, 7.5), radius=0.15)

    all_events = []
    for f in range(1, 10):
        all_events.extend(detector.update([ball], {5: p5}, frame_idx=f, player_teams=teams))

    assert any(e["type"] == "PLAYER_OUT_OF_BOUNDS" for e in all_events)
    player = registry.canonical_players["20"]
    assert player.turnovers == 1
    assert player.turnovers_out_of_bounds == 1


def test_backcourt_violation_over_and_back():
    """Ballhandler established in frontcourt stepping back onto centerline triggers BACKCOURT_VIOLATION (FIBA Art. 30)."""
    from hoop_stats.geometry import CourtGeometry
    geo = CourtGeometry(config_path="non_existent.json")
    registry = PlayerRegistry(roster_home=["14"])
    detector = EventDetector(frame_w=1280, frame_h=720, court_geometry=geo, player_registry=registry, fps=30.0)
    teams = {"3": "home"}
    registry.register_track(3, team="home")
    registry.bind_jersey(3, "14", team="home")

    # Attacking basket 1 (Right basket, frontcourt is x > 14.0m)
    # 1. Establish frontcourt at x = 16.0m (both feet in frontcourt)
    p3_fc = create_player(3, 16.0, 7.5, w=0.6, h=1.8)
    ball_fc = Ball(center=Point2D(16.0, 7.5), radius=0.15)
    for f in range(1, 8):
        detector.update([ball_fc], {3: p3_fc}, frame_idx=f, player_teams=teams)

    assert detector.team_frontcourt_established["home"] is True

    # 2. Player retreats back and touches the centerline (x=14.0m)
    p3_cl = create_player(3, 14.0, 7.5, w=0.6, h=1.8)
    ball_cl = Ball(center=Point2D(14.0, 7.5), radius=0.15)
    evts = detector.update([ball_cl], {3: p3_cl}, frame_idx=15, player_teams=teams)

    assert any(e["type"] == "BACKCOURT_VIOLATION" for e in evts)
    player = registry.canonical_players["14"]
    assert player.turnovers == 1
    assert player.turnovers_backcourt == 1


def test_three_seconds_violation_in_restricted_area():
    """Offensive player remaining > 3.0s in opponent key triggers THREE_SECONDS_VIOLATION (FIBA Art. 26)."""
    from hoop_stats.geometry import CourtGeometry
    geo = CourtGeometry(config_path="non_existent.json")
    registry = PlayerRegistry(roster_home=["66", "14"])
    detector = EventDetector(frame_w=1280, frame_h=720, court_geometry=geo, player_registry=registry, fps=30.0)
    teams = {"1": "home", "2": "home"}
    registry.register_track(1, team="home")
    registry.bind_jersey(1, "14", team="home")
    registry.register_track(2, team="home")
    registry.bind_jersey(2, "66", team="home")

    # Establish frontcourt possession by Player 1 (x=18.0m, ballhandler)
    p1 = create_player(1, 18.0, 7.5, w=0.6, h=1.8)
    ball = Ball(center=Point2D(18.0, 7.5), radius=0.15)

    # Teammate Player 2 is camping in the opponent's paint (x=24.0m, y=7.5m)
    p2_paint = create_player(2, 24.0, 7.5, w=0.6, h=1.8)
    capsules = {1: p1, 2: p2_paint}

    # Frame 1 to 7: establish frontcourt
    for f in range(1, 8):
        detector.update([ball], capsules, frame_idx=f, player_teams=teams)

    assert detector.team_frontcourt_established["home"] is True

    # Continue for > 3.0 seconds (95 frames at 30fps)
    detected_3sec = False
    for f in range(8, 105):
        evts = detector.update([ball], capsules, frame_idx=f, player_teams=teams)
        if any(e["type"] == "THREE_SECONDS_VIOLATION" and e["player_id"] == 2 for e in evts):
            detected_3sec = True
            break

    assert detected_3sec is True
    player66 = registry.canonical_players["66"]
    assert player66.turnovers == 1
    assert player66.turnovers_three_seconds == 1


def test_step_based_traveling_discrete_three_steps():
    """Taking >= 3 discrete steps after ball gathering without dribbling triggers Traveling (FIBA Art. 25)."""
    from hoop_stats.geometry import CourtGeometry
    geo = CourtGeometry(config_path="non_existent.json")
    registry = PlayerRegistry(roster_home=["20"])
    detector = EventDetector(frame_w=1280, frame_h=720, court_geometry=geo, player_registry=registry, fps=30.0)
    teams = {"5": "home"}
    registry.register_track(5, team="home")
    registry.bind_jersey(5, "20", team="home")

    # Start: Player 5 catches ball at (15.0m, 7.5m)
    p5 = create_player(5, 15.0, 7.5, w=0.6, h=1.8)
    p5.meter_pos = Point2D(15.0, 7.5)
    ball = Ball(center=Point2D(15.0, 7.5), radius=0.15)
    detector.update([ball], {5: p5}, frame_idx=1, player_teams=teams)

    # Step 1 (Pivot Foot): advances by 0.70m -> x = 15.70m
    p5_step1 = create_player(5, 15.70, 7.5, w=0.6, h=1.8)
    p5_step1.meter_pos = Point2D(15.70, 7.5)
    detector.update([Ball(center=Point2D(15.70, 7.5), radius=0.15)], {5: p5_step1}, frame_idx=6, player_teams=teams)
    assert detector.steps_since_gather == 1

    # Step 2: advances by another 0.70m -> x = 16.40m
    p5_step2 = create_player(5, 16.40, 7.5, w=0.6, h=1.8)
    p5_step2.meter_pos = Point2D(16.40, 7.5)
    evts_step2 = detector.update([Ball(center=Point2D(16.40, 7.5), radius=0.15)], {5: p5_step2}, frame_idx=12, player_teams=teams)
    assert detector.steps_since_gather == 2
    # 2 steps are legal: no traveling yet!
    assert not any(e["type"] == "TRAVELING" for e in evts_step2)

    # Step 3: illegal 3rd step without dribble/pass/shot -> x = 17.15m
    p5_step3 = create_player(5, 17.15, 7.5, w=0.6, h=1.8)
    p5_step3.meter_pos = Point2D(17.15, 7.5)
    evts_step3 = detector.update([Ball(center=Point2D(17.15, 7.5), radius=0.15)], {5: p5_step3}, frame_idx=18, player_teams=teams)

    assert any(e["type"] == "TRAVELING" for e in evts_step3)
    player = registry.canonical_players["20"]
    assert player.turnovers == 1
    assert player.turnovers_traveling == 1


def test_double_dribble_violation():
    """Dribbling, gathering the ball, and then bouncing again triggers Double Dribble (FIBA Art. 24)."""
    registry = PlayerRegistry(roster_home=["14"])
    detector = EventDetector(frame_w=1280, frame_h=720, player_registry=registry, fps=30.0)
    teams = {"1": "home"}
    registry.register_track(1, team="home")
    registry.bind_jersey(1, "14", team="home")

    p1 = create_player(1, 500, 300)
    ball = Ball(center=Point2D(500, 300), radius=10)

    # 1. Start possession
    detector.update([ball], {1: p1}, frame_idx=1, player_teams=teams)

    # 2. First Dribble: ball down then up
    detector.update([Ball(center=Point2D(500, 330), radius=10)], {1: p1}, frame_idx=2, player_teams=teams)
    detector.update([Ball(center=Point2D(500, 300), radius=10)], {1: p1}, frame_idx=3, player_teams=teams)
    assert detector.dribble_bounces >= 1

    # 3. Ball is gathered (no bounces for > 15 frames)
    for f in range(4, 22):
        detector.update([Ball(center=Point2D(500, 300), radius=10)], {1: p1}, frame_idx=f, player_teams=teams)

    assert detector.dribble_state == "gathered"

    # 4. Player tries to dribble a SECOND time (downward then upward bounce)
    detector.update([Ball(center=Point2D(500, 335), radius=10)], {1: p1}, frame_idx=23, player_teams=teams)
    evts = detector.update([Ball(center=Point2D(500, 305), radius=10)], {1: p1}, frame_idx=24, player_teams=teams)

    assert any(e["type"] == "DOUBLE_DRIBBLE" for e in evts)
    player = registry.canonical_players["14"]
    assert player.turnovers == 1
    assert player.turnovers_double_dribble == 1


def test_eight_seconds_violation_in_backcourt():
    """Failing to advance ball past centerline within 8.0s triggers EIGHT_SECONDS_VIOLATION (FIBA Art. 28)."""
    from hoop_stats.geometry import CourtGeometry
    geo = CourtGeometry(config_path="non_existent.json")
    registry = PlayerRegistry(roster_home=["11"])
    detector = EventDetector(frame_w=1280, frame_h=720, court_geometry=geo, player_registry=registry, fps=30.0)
    teams = {"4": "home"}
    registry.register_track(4, team="home")
    registry.bind_jersey(4, "11", team="home")

    # Team attacking right rim (frontcourt is x > 14.0m, backcourt is x <= 14.0m)
    # Player 4 has ball in backcourt at x = 8.0m
    p4 = create_player(4, 8.0, 7.5, w=0.6, h=1.8)
    ball = Ball(center=Point2D(8.0, 7.5), radius=0.15)

    # Frame 1: start possession in backcourt
    detector.update([ball], {4: p4}, frame_idx=1, player_teams=teams)

    # Remains in backcourt for > 8.0 seconds (245 frames at 30 fps)
    detected_8sec = False
    for f in range(2, 255):
        evts = detector.update([ball], {4: p4}, frame_idx=f, player_teams=teams)
        if any(e["type"] == "EIGHT_SECONDS_VIOLATION" for e in evts):
            detected_8sec = True
            break

    assert detected_8sec is True
    player = registry.canonical_players["11"]
    assert player.turnovers == 1
    assert player.turnovers_eight_seconds == 1


def test_five_seconds_closely_guarded():
    """Ballhandler closely guarded (<= 1.0m) holding ball for >= 5.0s triggers FIVE_SECONDS_VIOLATION (FIBA Art. 27)."""
    registry = PlayerRegistry(roster_home=["14"], roster_away=["8"])
    detector = EventDetector(frame_w=1280, frame_h=720, player_registry=registry, fps=30.0)
    teams = {"1": "home", "2": "away"}
    registry.register_track(1, team="home")
    registry.bind_jersey(1, "14", team="home")
    registry.register_track(2, team="away")
    registry.bind_jersey(2, "8", team="away")

    # Player 1 holds ball at (500, 300)
    # Defender 2 is closely guarding at (525, 300) -> distance 25px (~0.58m <= 1.0m)
    p1 = create_player(1, 500, 300, h=80.0)
    p2 = create_player(2, 525, 300, h=80.0)
    ball = Ball(center=Point2D(500, 300), radius=10)
    capsules = {1: p1, 2: p2}

    # Start possession
    detector.update([ball], capsules, frame_idx=1, player_teams=teams)

    # Held under pressure for >= 5.0 seconds (155 frames at 30fps)
    detected_5sec = False
    for f in range(2, 165):
        evts = detector.update([ball], capsules, frame_idx=f, player_teams=teams)
        if any(e["type"] == "FIVE_SECONDS_VIOLATION" for e in evts):
            detected_5sec = True
            break

    assert detected_5sec is True
    player = registry.canonical_players["14"]
    assert player.turnovers == 1
    assert player.turnovers_five_seconds == 1


