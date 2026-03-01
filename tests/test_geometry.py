import pytest
import os
import json
from hoop_stats.geometry import CourtGeometry

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
