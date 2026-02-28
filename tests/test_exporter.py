"""test_exporter.py — tests for hoop_stats.exporter (DataExporter)"""
import os
import json
import csv
import tempfile
import pytest
from hoop_stats.exporter import DataExporter


SAMPLE_PLAYERS = {
    "1": {
        "jersey": "23",
        "team": "Team A",
        "jersey_color": "Red",
        "max_speed_ms": 7.5,
        "max_accel_ms2": 3.2,
        "trajectory_pts": 120,
        "trail_sample": [[0.0, 0.0], [1.0, 1.0]],
    }
}

SAMPLE_SCORES = {
    "left_basket_pts": 4,
    "right_basket_pts": 6,
    "total_baskets": 5,
}

SAMPLE_OUTPUT = {
    "players": SAMPLE_PLAYERS,
    "scoring": SAMPLE_SCORES,
    "team_summary": {
        "Team A": {
            "player_count": 1,
            "avg_max_speed": 7.5,
            "avg_max_accel": 3.2,
        }
    },
}


@pytest.fixture
def exporter(tmp_path):
    name = str(tmp_path / "test_output")
    return DataExporter(name)


class TestDataExporter:
    def test_export_json_creates_file(self, exporter, tmp_path):
        exporter.export_json(SAMPLE_OUTPUT)
        json_file = str(tmp_path / "test_output.json")
        assert os.path.isfile(json_file)

    def test_export_json_valid_content(self, exporter, tmp_path):
        exporter.export_json(SAMPLE_OUTPUT)
        json_file = str(tmp_path / "test_output.json")
        with open(json_file) as f:
            data = json.load(f)
        assert "players" in data
        assert "scoring" in data
        assert "team_summary" in data

    def test_export_json_player_fields(self, exporter, tmp_path):
        exporter.export_json(SAMPLE_OUTPUT)
        json_file = str(tmp_path / "test_output.json")
        with open(json_file) as f:
            data = json.load(f)
        player = data["players"]["1"]
        assert player["jersey"] == "23"
        assert player["max_speed_ms"] == 7.5

    def test_export_csv_creates_file(self, exporter, tmp_path):
        exporter.export_csv_players(SAMPLE_PLAYERS, SAMPLE_SCORES)
        # Exporter names the file as output_name + ".csv" (no "_players" suffix)
        csv_file = str(tmp_path / "test_output.csv")
        assert os.path.isfile(csv_file)

    def test_export_csv_has_rows(self, exporter, tmp_path):
        exporter.export_csv_players(SAMPLE_PLAYERS, SAMPLE_SCORES)
        csv_file = str(tmp_path / "test_output.csv")
        with open(csv_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
