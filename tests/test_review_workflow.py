from hoop_stats.game_state import GameState


def test_game_events_include_review_contract():
    state = GameState()

    event = state.append_event(
        "scoresheet",
        "SHOT_MAKE",
        {"shooter_id": "12"},
        frame_ref=42,
        confidence=0.87,
        evidence={"bbox": [1, 2, 3, 4]},
        source_camera="cam1",
    )
    snapshot = state.snapshot()
    recent = snapshot["recent_events"][0]

    assert event.event_id
    assert recent["id"] == event.event_id
    assert recent["confidence"] == 0.87
    assert recent["review_status"] == "pending"
    assert recent["evidence"] == {"bbox": [1, 2, 3, 4]}
    assert recent["source_camera"] == "cam1"
