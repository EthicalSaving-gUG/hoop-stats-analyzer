"""
test_video_clipper.py
=====================
Unit tests for Stat-Linked Video Clipper and Clip Registry.
"""

import os
import json
import tempfile
import pytest
from hoop_stats.video_clipper import StatClip, StatClipRegistry


class DummyPlayerRegistry:
    def __init__(self):
        self.track_to_jersey = {101: "12", 102: "7"}


def test_stat_clip_creation_and_dict():
    clip = StatClip(
        clip_id="clip_0001_shot_make",
        event_type="SHOT_MAKE",
        player_id=101,
        jersey="12",
        team="home",
        event_frame=300,
        start_frame=225,
        end_frame=360,
        start_time_s=7.5,
        end_time_s=12.0,
        duration_s=4.5,
        description="3PT Made by #12",
        points=3,
        zone="three_right",
        metadata={"shot_type": "3P"}
    )
    d = clip.to_dict()
    assert d["clip_id"] == "clip_0001_shot_make"
    assert d["jersey"] == "12"
    assert d["points"] == 3
    assert d["duration_s"] == 4.5


def test_stat_clip_registry_add_events_and_filtering():
    registry = StatClipRegistry(fps=30.0, pre_pad_s=2.0, post_pad_s=1.5)
    dummy_reg = DummyPlayerRegistry()

    # Event 1: Made 3-pointer
    evt1 = {
        "type": "SHOT_MAKE",
        "shooter_id": 101,
        "team": "home",
        "frame": 300,
        "points": 3,
        "zone": "three_top"
    }
    clip1 = registry.add_event(evt1, player_registry=dummy_reg, total_frames=1000)
    assert clip1.event_type == "SHOT_MAKE"
    assert clip1.jersey == "12"
    assert clip1.start_frame == 300 - int(30 * 2.0)  # 240
    assert clip1.end_frame == 300 + int(30 * 1.5)    # 345
    assert clip1.points == 3

    # Event 2: Traveling Turnover
    evt2 = {
        "type": "TRAVELING",
        "player_id": 102,
        "team": "home",
        "frame": 600,
        "description": "Traveling violation #7"
    }
    clip2 = registry.add_event(evt2, player_registry=dummy_reg, total_frames=1000)
    assert clip2.event_type == "TRAVELING"
    assert clip2.jersey == "7"

    # Event 3: Steal by away team
    evt3 = {
        "type": "FEHLPASS_STEAL",
        "stealer_id": 999,
        "stealer_team": "away",
        "frame": 900
    }
    clip3 = registry.add_event(evt3, player_registry=dummy_reg, total_frames=1000)
    assert clip3.jersey == "Unknown"

    assert len(registry.clips) == 3

    # Test filtering by player
    clips_12 = registry.get_clips_for_player("12")
    assert len(clips_12) == 1
    assert clips_12[0].clip_id == clip1.clip_id

    # Test filtering by event type
    travel_clips = registry.get_clips_by_event("TRAVELING")
    assert len(travel_clips) == 1
    assert travel_clips[0].clip_id == clip2.clip_id


def test_stat_clip_registry_export_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StatClipRegistry(fps=30.0)
        registry.add_event({
            "type": "SHOT_MAKE",
            "shooter_id": 5,
            "team": "home",
            "frame": 120,
            "points": 2
        })
        manifest_file = os.path.join(tmpdir, "clips_manifest.json")
        registry.export_manifest(manifest_file)

        assert os.path.exists(manifest_file)
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_clips"] == 1
        assert len(data["clips"]) == 1
        assert data["clips"][0]["event_type"] == "SHOT_MAKE"
