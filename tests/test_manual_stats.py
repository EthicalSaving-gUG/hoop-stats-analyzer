"""
test_manual_stats.py
====================
Unit tests for manual stats entry, Boxscore calculation, and protocol aggregation.
"""

import pytest
from hoop_stats.manual_stats import (
    BoxscoreCalculator,
    OFFICIAL_ROSTER_UFFENHEIM_26_27,
    ACTION_CODES,
    parse_time_to_seconds,
    format_seconds_to_time
)


def test_time_parsing_and_formatting():
    assert parse_time_to_seconds("36:55") == 2215.0
    assert parse_time_to_seconds("01:16:55") == 4615.0
    assert parse_time_to_seconds("00:00") == 0.0
    assert parse_time_to_seconds("") == 0.0
    assert format_seconds_to_time(2215.0) == "36:55"
    assert format_seconds_to_time(0.0) == "00:00"


def test_player_line_calculation():
    inputs = {
        "played": "x",
        "2T": 6,
        "2V": 4,
        "3T": 5,
        "3V": 8,
        "FT": 4,
        "FV": 4,
        "OR": 1,
        "DR": 6,
        "AS": 0,
        "ST": 5,
        "BL": 0,
        "TO": 9,
        "FZ": 5,
        "OF": 0,
        "min": "36:55",
        "PF": 3,
        "plus_minus": 8
    }

    line = BoxscoreCalculator.calculate_player_line("20", "Roman Blank", inputs)

    # Points: 6*2 + 5*3 + 4*1 = 12 + 15 + 4 = 31
    assert line["pts"] == 31
    assert line["2P_T"] == 6
    assert line["2P_V"] == 10
    assert line["2P_PCT"] == 0.600
    assert line["3P_T"] == 5
    assert line["3P_V"] == 13
    assert line["FG_T"] == 11
    assert line["FG_V"] == 23
    assert line["FW_T"] == 4
    assert line["FW_V"] == 8
    assert line["FW_PCT"] == 0.500
    assert line["Reb"] == 7
    assert line["fouls_drawn"] == 5
    assert line["played"] == 1

    # eFG% = (11 + 0.5 * 5) / 23 = 13.5 / 23 ~ 0.587
    assert abs(line["efg_pct"] - 0.587) < 0.005

    # Pro 40 scaling
    pro_40 = line["pro_40"]
    assert "pts_40" in pro_40
    # 31 / (36.916 min) * 40 ~ 33.58
    assert abs(pro_40["pts_40"] - 33.6) < 0.2


def test_boxscore_team_totals_and_control_ok():
    roster = [{"jersey": "20", "name": "Roman Blank"}, {"jersey": "14", "name": "Ralf Dubiel"}]
    data = {
        "20": {"2T": 5, "2V": 5, "3T": 2, "3V": 2, "FT": 2, "FV": 0, "TO": 3},  # 5*2 + 2*3 + 2 = 18 pts
        "14": {"2T": 3, "2V": 2, "3T": 1, "3V": 1, "FT": 3, "FV": 1, "TO": 2},  # 3*2 + 1*3 + 3 = 12 pts
    }

    box = BoxscoreCalculator.calculate_boxscore(roster, data, target_sbu_score=30)
    totals = box["team_totals"]
    ctrl = box["control"]

    assert totals["pts"] == 30
    assert totals["2P_T"] == 8
    assert totals["3P_T"] == 3
    assert totals["FW_T"] == 5
    assert totals["TO"] == 5
    assert ctrl["status"] == "OK"
    assert "✓ Boxscore-Punkte = Endergebnis" in ctrl["message"]


def test_boxscore_control_error():
    roster = [{"jersey": "20", "name": "Roman Blank"}]
    data = {"20": {"2T": 5, "2V": 0, "3T": 0, "3V": 0, "FT": 0, "FV": 0}}  # 10 pts
    box = BoxscoreCalculator.calculate_boxscore(roster, data, target_sbu_score=15)
    ctrl = box["control"]

    assert ctrl["status"] == "ERROR"
    assert ctrl["diff"] == -5
    assert "Differenz" in ctrl["message"]


def test_aggregate_protocol_variante_b():
    protocol = [
        {"jersey": "91", "code": "3V"},
        {"jersey": "20", "code": "OR"},
        {"jersey": "20", "code": "2T"},
        {"jersey": "21", "code": "AS"},
        {"jersey": "20", "code": "TO"},
        {"jersey": "14", "code": "FZ"},
        {"jersey": "14", "code": "OF"},
    ]

    agg = BoxscoreCalculator.aggregate_protocol(protocol)

    assert agg["91"]["3V"] == 1
    assert agg["20"]["OR"] == 1
    assert agg["20"]["2T"] == 1
    assert agg["20"]["TO"] == 1
    assert agg["21"]["AS"] == 1
    assert agg["14"]["FZ"] == 1
    assert agg["14"]["OF"] == 1

    # Verify Boxscore generation from aggregated protocol
    roster = [{"jersey": "20", "name": "Roman Blank"}, {"jersey": "91", "name": "Arthur Apinyan"}]
    box = BoxscoreCalculator.calculate_boxscore(roster, agg)
    assert box["team_totals"]["pts"] == 2
    assert box["team_totals"]["OR"] == 1
    assert box["team_totals"]["TO"] == 1
