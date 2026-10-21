"""
test_coach_report.py
====================
Unit tests for AI Coach Report and Four Factors Analytics Generator.
"""

import os
import json
import tempfile
import pytest
from hoop_stats.coach_report import CoachReportGenerator


def test_coach_report_four_factors_calculation():
    generator = CoachReportGenerator(team_name="SB Uffenheim")

    tracked_events = {
        "points": 55,
        "fgm": 22,
        "fga": 50,
        "fg2m": 18,
        "fg2a": 38,
        "fg3m": 4,
        "fg3a": 12,
        "ftm": 7,
        "fta": 14,
        "reb": 30,
        "oreb": 8,
        "dreb": 22,
        "assists": 10,
        "turnovers": 15,
        "turnovers_traveling": 4,
        "turnovers_bad_pass": 6,
        "turnovers_three_seconds": 1,
        "turnovers_backcourt": 1,
        "turnovers_out_of_bounds": 1,
        "turnovers_double_dribble": 1,
        "turnovers_eight_seconds": 1,
        "turnovers_five_seconds": 0,
        "steals": 7,
        "blocks": 3,
    }

    canonical_players = {
        "12": {"name": "L. Schmidt", "points": 18, "fgm": 7, "fga": 14},
        "7": {"name": "M. Weber", "points": 12, "fgm": 5, "fga": 10},
    }

    report = generator.generate_report(tracked_events, canonical_players, game_duration_min=40.0)

    # 1. Possessions: FGA (50) + 0.44 * FTA (14 = 6.16) - OREB (8) + TO (15) = 63.16
    expected_poss = 50 + 0.44 * 14 - 8 + 15
    assert abs(report["summary"]["possessions"] - round(expected_poss, 1)) < 0.2

    # 2. Four Factors
    # eFG% = (22 + 0.5 * 4) / 50 * 100 = 24 / 50 * 100 = 48.0%
    assert report["four_factors"]["efg_pct"] == 48.0

    # TOV% = TO / Possessions * 100 = 15 / 63.16 * 100 ~ 23.7%
    assert 23.0 <= report["four_factors"]["tov_pct"] <= 24.5

    # FT Rate = FTA / FGA = 14 / 50 = 0.280
    assert report["four_factors"]["ft_rate"] == 0.280

    # 3. Turnover Diagnosis check
    to_diag = report["turnover_diagnosis"]
    assert to_diag["traveling"] == 4
    assert to_diag["double_dribble"] == 1
    assert to_diag["eight_seconds"] == 1
    assert to_diag["three_seconds"] == 1
    assert to_diag["total_turnovers"] == 15

    # 4. Top Scorers check
    top = report["top_scorers"]
    assert len(top) == 2
    assert top[0]["jersey"] == "12"
    assert top[0]["pts"] == 18


def test_coach_report_export_markdown():
    generator = CoachReportGenerator(team_name="SB Uffenheim")

    tracked_events = {
        "points": 50,
        "fgm": 20,
        "fga": 45,
        "fg2m": 18,
        "fg2a": 35,
        "fg3m": 2,
        "fg3a": 10,
        "ftm": 8,
        "fta": 12,
        "reb": 25,
        "oreb": 6,
        "dreb": 19,
        "assists": 8,
        "turnovers": 12,
        "turnovers_traveling": 3,
        "turnovers_bad_pass": 5,
        "turnovers_three_seconds": 1,
        "turnovers_backcourt": 1,
        "turnovers_out_of_bounds": 1,
        "turnovers_double_dribble": 1,
        "turnovers_eight_seconds": 0,
        "turnovers_five_seconds": 0,
        "steals": 5,
        "blocks": 2,
    }

    report = generator.generate_report(tracked_events, {})

    with tempfile.TemporaryDirectory() as tmpdir:
        md_file = os.path.join(tmpdir, "coach_report.md")
        generator.export_markdown_report(report, md_file)

        assert os.path.exists(md_file)
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# 🏀 AI Coach Game Report: SB Uffenheim" in content
        assert "Dean Oliver's Four Factors" in content
        assert "Schrittfehler (Traveling, Art. 25)" in content
        assert "Zweimaliges Dribbeln (Art. 24)" in content
        assert "8 Sekunden im Rückfeld (Art. 28)" in content
