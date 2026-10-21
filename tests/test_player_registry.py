import pytest
from hoop_stats.player_registry import PlayerRegistry, PlayerStatSheet


def test_player_registry_initialization():
    registry = PlayerRegistry(roster_home=["11", "14", "20", "26", "35", "66"])
    assert "20" in registry.roster_home
    assert len(registry.canonical_players) == 0


def test_late_binding_retroactive_flush():
    """Actions performed before identity is known must be retroactively credited upon identification."""
    registry = PlayerRegistry(roster_home=["11", "14", "20", "26", "35", "66"])
    track_id = 7
    registry.register_track(track_id, team="home")

    # Step 1: Track #7 scores a 2-point shot and an offensive rebound before OCR succeeds
    registry.record_action(track_id, "SHOT_MAKE", points=2, details={"zone": "paint"})
    registry.record_action(track_id, "OFFENSIVE_REBOUND")

    # Verify no canonical player yet, but team totals include pending actions
    assert "20" not in registry.canonical_players
    totals = registry.get_team_totals("home")
    assert totals["points"] == 2
    assert totals["oreb"] == 1

    # Step 2: High-confidence OCR observation identifies track 7 as #20
    resolved = registry.record_ocr_observation(track_id, "20", confidence=0.88)
    assert resolved == "20"
    assert registry.track_to_jersey[track_id] == "20"

    # Step 3: Canonical player #20 must now have the retroactively credited points & rebound
    player = registry.canonical_players["20"]
    assert player.points == 2
    assert player.fg2m == 1
    assert player.oreb == 1
    assert track_id in player.associated_tracks


def test_substitution_merge_across_stints():
    """When a player is subbed out and returns with a new track ID, stats must merge into one player sheet."""
    registry = PlayerRegistry(roster_home=["11", "14", "20", "26", "35", "66"])

    # Stint 1: Player #14 plays as track 4
    registry.register_track(4, team="home")
    registry.record_ocr_observation(4, "14", confidence=0.90)
    registry.record_action(4, "SHOT_MAKE", points=3, details={"zone": "above_the_break_three"})

    # Player is subbed out
    registry.set_substitution_state(4, is_on_court=False)

    # Stint 2: Later in the game, player #14 enters with track 19
    registry.register_track(19, team="home")
    registry.record_action(19, "SHOT_MAKE", points=2, details={"zone": "paint"})
    registry.record_ocr_observation(19, "14", confidence=0.85)

    player = registry.canonical_players["14"]
    assert player.points == 5  # 3 + 2
    assert player.fg3m == 1
    assert player.fg2m == 1
    assert 4 in player.associated_tracks
    assert 19 in player.associated_tracks


def test_roster_constraint_filters_noise():
    """OCR noise numbers that do not exist on the roster must be discarded."""
    registry = PlayerRegistry(roster_home=["11", "14", "20", "26", "35", "66"])
    track_id = 9
    registry.register_track(track_id, team="home")

    # Noise detection "84" with high confidence
    res = registry.record_ocr_observation(track_id, "84", confidence=0.95)
    assert res is None
    assert track_id not in registry.track_to_jersey

    # Valid roster detection "26"
    res2 = registry.record_ocr_observation(track_id, "26", confidence=0.91)
    assert res2 == "26"
    assert registry.track_to_jersey[track_id] == "26"


def test_cumulative_confidence_voting():
    """Multiple medium confidence readings accumulate to trigger binding."""
    registry = PlayerRegistry(
        roster_home=["11", "14", "20", "26", "35", "66"],
        voting_threshold=1.2,
        single_hit_conf=0.85
    )
    track_id = 12
    registry.register_track(track_id, team="home")

    # Reading 1: 0.50 (below threshold)
    assert registry.record_ocr_observation(track_id, "35", confidence=0.50) is None
    # Reading 2: 0.45 (total 0.95 < 1.2)
    assert registry.record_ocr_observation(track_id, "35", confidence=0.45) is None
    # Reading 3: 0.40 (total 1.35 >= 1.2 -> triggers binding)
    assert registry.record_ocr_observation(track_id, "35", confidence=0.40) == "35"
    assert registry.track_to_jersey[track_id] == "35"
