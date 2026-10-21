import pytest
import os
import json
from hoop_stats.geometry import CourtGeometry
from hoop_stats.primitives import PlayerCapsule, Point2D

@pytest.fixture
def mock_config(tmp_path):
    config = {
        "zones": {
            "paint": {"type": "rectangle", "p1": [0, 5], "p2": [4, 0]},
            "three_point": {"type": "polygon", "points": [[0, 7], [8, 7], [8, 15], [0, 15]]}
        }
    }
    path = tmp_path / "mock_court.json"
    with open(path, "w") as f:
        json.dump(config, f)
    return str(path)

class TestCourtGeometry:
    def test_get_zone_inside_paint(self, mock_config):
        geo = CourtGeometry(config_path=mock_config)
        assert geo.get_zone(2, 2) == "paint"
        
    def test_get_zone_inside_three_point(self, mock_config):
        geo = CourtGeometry(config_path=mock_config)
        assert geo.get_zone(4, 10) == "three_point"
        assert geo.get_zone(2, 8) == "three_point"
        
    def test_get_zone_out_of_bounds(self, mock_config):
        geo = CourtGeometry(config_path=mock_config)
        # Outside paint but not in 3pt
        assert geo.get_zone(6, 4) == "out_of_bounds"
        # Completely out of court
        assert geo.get_zone(20, 20) == "out_of_bounds"
        
    def test_edge_cases(self, mock_config):
        geo = CourtGeometry(config_path=mock_config)
        # On the line of paint (x=4, y=2 should be inside/edge)
        assert geo.get_zone(4, 2) == "paint"


class TestFibaFloorContact:
    def test_three_point_arc_vs_two_point(self):
        """FIBA 6.75m line: floor contact determines 2PM vs 3PM."""
        geo = CourtGeometry(config_path="non_existent.json")  # Default FIBA geometry
        # Right basket rim is at (26.425, 7.5)
        # Point at 7.0m straight out from rim: x = 26.425 - 7.0 = 19.425, y = 7.5
        assert geo.is_three_pointer(19.425, 7.5, basket_idx=1) is True

        # Point at 6.0m straight out from rim (mid-range): x = 26.425 - 6.0 = 20.425, y = 7.5
        assert geo.is_three_pointer(20.425, 7.5, basket_idx=1) is False

        # Touching the 6.75m line (exactly 6.75m) belongs to 2-point zone in FIBA rules!
        assert geo.is_three_pointer(26.425 - 6.75, 7.5, basket_idx=1) is False

        # Corner 3-pointer (y=0.5m < 0.90m sideline distance, x=24.0m < 25.01m)
        assert geo.is_three_pointer(24.0, 0.5, basket_idx=1) is True
        # Corner 2-pointer (y=0.5m, x=26.0m > 25.01m)
        assert geo.is_three_pointer(26.0, 0.5, basket_idx=1) is False

    def test_paint_restricted_area_floor_contact(self):
        """FIBA Art. 2.4.3: Restricted area (paint) dimensions and floor contact."""
        geo = CourtGeometry(config_path="non_existent.json")
        # Right basket paint: x in [22.20, 28.0], y in [5.05, 9.95]
        assert geo.is_in_paint(24.0, 7.5, basket_idx=1) is True
        assert geo.is_in_paint(20.0, 7.5, basket_idx=1) is False
        # Edge of paint (y=5.05 is inside the key)
        assert geo.is_in_paint(25.0, 5.05, basket_idx=1) is True
        # Outside paint width (y=4.5)
        assert geo.is_in_paint(25.0, 4.5, basket_idx=1) is False

    def test_out_of_bounds_floor_contact(self):
        """FIBA Art. 23: Boundary lines belong to out-of-bounds."""
        geo = CourtGeometry(config_path="non_existent.json")
        # In-bounds position
        assert geo.is_out_of_bounds(14.0, 7.5) is False
        # Stepping on baseline (x=0.02 <= 0.05)
        assert geo.is_out_of_bounds(0.02, 7.5) is True
        # Stepping beyond baseline (x=-0.5)
        assert geo.is_out_of_bounds(-0.5, 7.5) is True
        # Stepping on sideline (y=14.98 >= 14.95)
        assert geo.is_out_of_bounds(14.0, 14.98) is True

    def test_centerline_and_backcourt_rules(self):
        """FIBA Art. 30: Centerline (x=14.0) belongs to the backcourt."""
        geo = CourtGeometry(config_path="non_existent.json")
        # Attacking Right basket (basket 1): frontcourt is x > 14.0, backcourt is x <= 14.0
        assert geo.is_frontcourt(16.0, 7.5, attacking_basket=1) is True
        assert geo.is_backcourt(16.0, 7.5, attacking_basket=1) is False

        # In backcourt
        assert geo.is_backcourt(10.0, 7.5, attacking_basket=1) is True
        assert geo.is_frontcourt(10.0, 7.5, attacking_basket=1) is False

        # Stepping on the centerline (x=14.0m): belongs to backcourt!
        assert geo.is_on_centerline(14.0, 7.5) is True
        assert geo.is_backcourt(14.0, 7.5, attacking_basket=1) is True
        assert geo.is_frontcourt(14.0, 7.5, attacking_basket=1) is False

    def test_evaluate_player_floor_status_two_feet_rule(self):
        """
        Tests two-feet ground contact rules:
        - 3-second rule: 1 foot in paint => in key; both feet out => cleared!
        - 3-point rule: both feet behind line => 3-point.
        """
        geo = CourtGeometry(config_path="non_existent.json")
        # Assume 1:1 pixel to meter mapping for unit test simplicity
        # Player right at the edge of the paint (x=22.2, y=7.5)
        # width = 2.0 (meter scale for test), so foot_left is at x=21.7 (outside paint), foot_right is at x=22.7 (inside paint)
        cap_straddling = PlayerCapsule(
            head=Point2D(22.2, 5.0),
            foot=Point2D(22.2, 7.5),
            width=2.0
        )
        status = geo.evaluate_player_floor_status(cap_straddling, attacking_basket=1)
        # One foot is in the paint => player is in_paint!
        assert status["in_paint"] is True
        assert status["paint_cleared"] is False

        # Player moved completely outside the paint (x=20.0, width=2.0 => left=19.5, right=20.5)
        cap_outside = PlayerCapsule(
            head=Point2D(20.0, 5.0),
            foot=Point2D(20.0, 7.5),
            width=2.0
        )
        status_outside = geo.evaluate_player_floor_status(cap_outside, attacking_basket=1)
        assert status_outside["in_paint"] is False
        assert status_outside["paint_cleared"] is True

