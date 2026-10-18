import pytest
from hoop_stats.benchmark_evaluator import BenchmarkEvaluator, OFFICIAL_GROUND_TRUTH_UFFENHEIM


def test_benchmark_evaluator_init():
    evaluator = BenchmarkEvaluator()
    assert evaluator.gt["team"] == "SB Uffenheim"
    assert evaluator.gt["final_score"]["uffenheim"] == 55
    assert len(evaluator.gt["players"]) == 6


def test_evaluate_team_totals_exact_match():
    evaluator = BenchmarkEvaluator()
    perfect_detected = {
        "points": 55,
        "fgm": 17,
        "fga": 56,
        "fg2m": 10,
        "fg2a": 37,
        "fg3m": 7,
        "fg3a": 19,
        "efg_pct": 0.366,
        "ts_pct": 0.408,
    }
    res = evaluator.evaluate_team_totals(perfect_detected)
    assert res["points"]["delta"] == 0
    assert res["points"]["accuracy_pct"] == 100.0
    assert res["fgm"]["accuracy_pct"] == 100.0


def test_evaluate_player_recognition():
    evaluator = BenchmarkEvaluator()
    detected = ["11", "26", "20", "99", "Unknown"]
    res = evaluator.evaluate_player_recognition(detected)
    assert "11" in res["matched_jerseys"]
    assert "26" in res["matched_jerseys"]
    assert "20" in res["matched_jerseys"]
    assert res["recall_pct"] == 50.0  # 3 of 6
    assert res["precision_pct"] == 75.0  # 3 of 4 valid detected
