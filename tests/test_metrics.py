"""test_metrics.py — pure unit tests for hoop_stats.metrics (no video needed)"""
import math
import pytest
from hoop_stats.metrics import PlayerMetrics


@pytest.fixture
def metrics():
    return PlayerMetrics(fps=30)


class TestUpdatePosition:
    def test_registers_new_player(self, metrics):
        metrics.update_position(1, 0.0, 0.0, 0)
        assert 1 in metrics.trajectories
        assert 1 in metrics.metrics

    def test_trajectory_grows(self, metrics):
        for i in range(5):
            metrics.update_position(1, float(i), 0.0, i)
        assert len(metrics.trajectories[1]) == 5


class TestSpeedCalculation:
    def test_constant_speed(self, metrics):
        """Player moves 1 m per frame at 30 fps → speed should ≈ 30 m/s."""
        fps = 30
        dt = 1.0 / fps
        for i in range(10):
            metrics.update_position(1, float(i), 0.0, i)
        speed = metrics.metrics[1]["max_speed"]
        assert abs(speed - fps) < 0.01   # 1 m / (1/30 s) = 30 m/s

    def test_zero_speed_stationary_player(self, metrics):
        for i in range(5):
            metrics.update_position(2, 0.0, 0.0, i)
        assert metrics.metrics[2]["max_speed"] == 0.0

    def test_max_speed_not_decreasing(self, metrics):
        # First two frames: fast movement
        metrics.update_position(3, 0.0, 0.0, 0)
        metrics.update_position(3, 5.0, 0.0, 1)
        fast = metrics.metrics[3]["max_speed"]
        # Then a stationary frame
        metrics.update_position(3, 5.0, 0.0, 2)
        assert metrics.metrics[3]["max_speed"] >= fast


class TestGetTrajectoryLines:
    def test_returns_list_of_tuples(self, metrics):
        metrics.update_position(4, 1.0, 2.0, 0)
        metrics.update_position(4, 3.0, 4.0, 1)
        traj = metrics.get_trajectory_lines(4)
        assert isinstance(traj, list)
        assert traj[0] == (1.0, 2.0)

    def test_unknown_player_returns_empty(self, metrics):
        traj = metrics.get_trajectory_lines(9999)
        assert traj == []
