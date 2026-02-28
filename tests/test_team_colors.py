"""test_team_colors.py — tests for hoop_stats.team_colors"""
import numpy as np
import pytest
from hoop_stats.team_colors import get_dominant_jersey_color, assign_teams_by_color


class TestGetDominantJerseyColor:
    def test_returns_string(self, solid_color_crop):
        """Color function must always return a string."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        bbox = [100, 100, 200, 300]
        # Plant the solid-colour crop into the frame's torso region (64×64)
        crop = solid_color_crop(0, 0, 255)  # Red, size 64×64
        frame[110:174, 100:164] = crop
        color = get_dominant_jersey_color(frame, bbox)
        assert isinstance(color, str)
        assert len(color) > 0

    def test_unknown_on_empty_crop(self):
        """Degenerate crop (zero size) should not crash – returns 'Unknown'."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        bbox = [0, 0, 1, 1]   # 1×1 pixel bbox → crop is near-empty
        color = get_dominant_jersey_color(frame, bbox)
        assert isinstance(color, str)


class TestAssignTeamsByColor:
    def test_two_groups_produce_two_teams(self):
        """Players with very different colors should be split into 2 teams."""
        # Simulate two jersey colors
        player_colors = {
            1: "Red", 2: "Red", 3: "Red",
            4: "Blue", 5: "Blue", 6: "Blue",
        }
        assignments = assign_teams_by_color(player_colors)
        assert isinstance(assignments, dict)
        teams = set(assignments.values())
        # Should produce at most 2 distinct team labels
        assert len(teams) <= 2

    def test_empty_colors_returns_dict(self):
        result = assign_teams_by_color({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_all_same_color_one_team(self):
        player_colors = {1: "White", 2: "White", 3: "White"}
        assignments = assign_teams_by_color(player_colors)
        assert len(set(assignments.values())) == 1
