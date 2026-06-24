import time
import numpy as np
from hoop_stats.game_state import GameState
from hoop_stats.agents.referee_agent import RefereeAgent
from hoop_stats.basketball_rules import NBA_RULES

class MockDetector:
    class Model:
        def track(self, frame, **kwargs):
            class Box:
                def __init__(self):
                    # x1, y1, x2, y2 -> width=10, height=20, centered at (20, 20) -> bottom center = (20, 30)
                    self.xyxy = type('t', (), {'cpu': lambda self_cpu: type('t2', (), {'numpy': lambda self_np: np.array([[15.0, 10.0, 25.0, 30.0]])})()})()
                    self.id = type('t', (), {'cpu': lambda self_cpu: type('t2', (), {'numpy': lambda self_np: np.array([1])})()})()
            class Result:
                def __init__(self):
                    self.boxes = Box()
            return [Result()]
    def __init__(self):
        self.model = self.Model()

class MockGeometry:
    def pixel_to_meter(self, x, y):
        # return dummy values that simulate a paint coordinate
        return 2.0, 2.0
    def get_zone(self, mx, my):
        return "paint"

def test_referee_defensive_3s():
    gs = GameState(rules=NBA_RULES)
    gs.game_clock_running = True

    # Init agent
    agent = RefereeAgent(gs, None)
    agent._detector = MockDetector()
    agent._geometry = MockGeometry()

    # Simulate first frame
    dummy_frame = np.zeros((100, 100, 3))
    agent._check_defensive_three_seconds(dummy_frame, frame_idx=0)

    # Paint time should be recorded
    assert "1" in agent._paint_times

    # Frame index should increase such that frame_idx * tick_rate > 3.0s
    # Tick rate is 5.0, so 1 frame is 0.2s. For 3.1s we need 16 frames.

    # Simulate next frame
    agent._check_defensive_three_seconds(dummy_frame, frame_idx=16)

    # Violation should be emitted
    events = [e for e in gs.events if e.event_type == "DEFENSIVE_3_SECONDS"]
    assert len(events) == 1
    assert events[0].data["player_id"] == "1"
    assert events[0].data["duration"] > 3.0

    print("Defensive 3 Seconds Test Passed!")

if __name__ == "__main__":
    test_referee_defensive_3s()
