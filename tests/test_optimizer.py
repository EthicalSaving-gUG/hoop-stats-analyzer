import pytest
import os
import json
from pathlib import Path
from hoop_stats.optimizer import PipelineOptimizer, DEFAULT_BASELINE_PARAMS


def test_optimizer_initialization():
    optimizer = PipelineOptimizer()
    assert len(optimizer.ground_truth_datasets) >= 2
    assert "rim_confidence" in optimizer.baseline_params


def test_evaluate_params_fitness():
    optimizer = PipelineOptimizer()
    score, metrics = optimizer.evaluate_params_fitness(DEFAULT_BASELINE_PARAMS)
    assert 50.0 <= score <= 100.0
    assert "points_acc" in metrics
    assert "fgm_acc" in metrics
    assert "player_recall" in metrics
    assert "traveling_acc" in metrics


def test_run_quick_optimization(tmp_path):
    temp_config = tmp_path / "test_pipeline_params.json"
    temp_report = tmp_path / "test_report.md"

    optimizer = PipelineOptimizer(config_path=str(temp_config))
    res = optimizer.run_optimization(quick=True, max_trials=10)

    assert "best_score" in res
    assert "optimized_params" in res
    assert temp_config.exists()

    optimizer.export_markdown_report(res, path=str(temp_report))
    assert temp_report.exists()
    content = temp_report.read_text(encoding="utf-8")
    assert "Hoop-Stats Pipeline Optimization Report" in content
