"""
Game Clock Agent — Spielzeiterfassung.
Manages the game clock (4×10min quarters FIBA), detects timeouts,
and publishes quarter/game end events.
"""
import time
from .base_agent import BaseAgent
from ..detector import HoopDetector


class GameClockAgent(BaseAgent):
    AGENT_NAME = "game_clock"

    def __init__(self, game_state, stream_manager, camera_ids=None, **kwargs):
        super().__init__(game_state, stream_manager, camera_ids, tick_rate=5, **kwargs)
        self._detector = None
        self._last_tick = None
        self._no_movement_frames = 0
        self._prev_frame_gray = None

    def on_start(self):
        import cv2
        self._detector = HoopDetector()
        self._last_tick = time.time()
        # Start game clock on agent start
        self.game_state.game_clock_running = True

    def process_frame(self, camera_id, frame, frame_idx):
        import cv2
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now

        # Detect activity level via simple frame differencing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))  # low-res for speed

        if self._prev_frame_gray is not None:
            diff = cv2.absdiff(self._prev_frame_gray, gray)
            movement = diff.mean()

            if movement < 3.0:
                # Very low movement → possible dead ball / timeout
                self._no_movement_frames += 1
            else:
                if self._no_movement_frames > 25:  # ~5 seconds at 5fps
                    # Resuming play after a stoppage
                    if not self.game_state.game_clock_running:
                        self.game_state.game_clock_running = True
                        self.game_state.shot_clock_running = True
                        self.game_state.append_event(
                            self.AGENT_NAME, "PLAY_RESUMED", {}, frame_idx
                        )
                self._no_movement_frames = 0

            # Detect timeout (>5s of no movement while clock was running)
            if self._no_movement_frames == 25 and self.game_state.game_clock_running:
                self.game_state.game_clock_running = False
                self.game_state.shot_clock_running = False
                self.game_state.append_event(
                    self.AGENT_NAME, "DEAD_BALL",
                    {"reason": "no_movement"}, frame_idx
                )

        self._prev_frame_gray = gray

        # Check quarter end
        if self.game_state.game_clock_s <= 0.0 and not self.game_state.game_over:
            self.game_state.game_clock_running = False
            self.game_state.shot_clock_running = False

            if self.game_state.quarter < 4:
                self.game_state.append_event(
                    self.AGENT_NAME, "QUARTER_END",
                    {"quarter": self.game_state.quarter}, frame_idx
                )
                self.game_state.quarter += 1
                self.game_state.game_clock_s = self.game_state.quarter_length_s
                # Reset quarter fouls
                self.game_state.team_fouls_home[self.game_state.quarter] = 0
                self.game_state.team_fouls_away[self.game_state.quarter] = 0
            else:
                # Game over (or overtime logic could go here)
                self.game_state.game_over = True
                self.game_state.append_event(
                    self.AGENT_NAME, "GAME_END",
                    {"final_score_home": self.game_state.score_home,
                     "final_score_away": self.game_state.score_away},
                    frame_idx
                )
