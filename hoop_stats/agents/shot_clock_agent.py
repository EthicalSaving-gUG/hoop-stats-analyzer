"""
24-Sekunden Shot Clock Agent.
Tracks ball possession and manages the 24-second shot clock.
Detects possession changes by monitoring which half of the court the ball is in.
"""
import time
from .base_agent import BaseAgent
from ..detector import HoopDetector


class ShotClockAgent(BaseAgent):
    AGENT_NAME = "shot_clock"

    def __init__(self, game_state, stream_manager, camera_ids=None, **kwargs):
        super().__init__(game_state, stream_manager, camera_ids, tick_rate=10, **kwargs)
        self._detector = None
        self._last_tick = None
        self._last_ball_x = None
        self._consecutive_no_ball = 0

    def on_start(self):
        self._detector = HoopDetector()
        self._last_tick = time.time()

    def process_frame(self, camera_id, frame, frame_idx):
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now

        # Advance shot clock
        self.game_state.tick(dt)

        # Detect ball position
        ball_boxes = self._detector.detect_ball(frame)
        h, w = frame.shape[:2]
        mid_x = w / 2

        if ball_boxes:
            bx = (ball_boxes[0][0] + ball_boxes[0][2]) / 2
            self._consecutive_no_ball = 0

            # Determine possession side (simplified: left half = home, right = away)
            new_possession = "home" if bx < mid_x else "away"
            old_possession = self.game_state.possession

            if old_possession is None:
                # First possession
                self.game_state.set_possession(new_possession)
                self.game_state.reset_shot_clock(short=False)
                self.game_state.append_event(
                    self.AGENT_NAME, "POSSESSION_START",
                    {"team": new_possession}, frame_idx
                )
            elif new_possession != old_possession:
                # Possession change → full 24s reset
                self.game_state.set_possession(new_possession)
                self.game_state.reset_shot_clock(short=False)
                self.game_state.append_event(
                    self.AGENT_NAME, "POSSESSION_CHANGE",
                    {"from": old_possession, "to": new_possession}, frame_idx
                )

            self._last_ball_x = bx
        else:
            self._consecutive_no_ball += 1
            # If ball not seen for a while, assume dead ball → stop clock
            if self._consecutive_no_ball > 30:  # ~3 seconds at 10fps
                if self.game_state.shot_clock_running:
                    self.game_state.shot_clock_running = False

        # Check shot clock violation
        if self.game_state.shot_clock_s <= 0.0 and self.game_state.shot_clock_running:
            self.game_state.shot_clock_running = False
            self.game_state.append_event(
                self.AGENT_NAME, "SHOT_CLOCK_VIOLATION",
                {"team": self.game_state.possession}, frame_idx
            )
