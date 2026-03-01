"""
Abstract base class for all Kampfgericht agents.
Each agent runs in its own thread and processes frames from the StreamManager.
"""
import threading
import time
from abc import ABC, abstractmethod
from ..game_state import GameState
from ..stream_manager import StreamManager


class BaseAgent(threading.Thread, ABC):
    """
    Abstract agent. Subclasses implement `process_frame()`.
    The run-loop grabs the latest frame from the assigned cameras
    and calls process_frame() at a configurable rate.
    """

    AGENT_NAME = "base"

    def __init__(self, game_state: GameState, stream_manager: StreamManager,
                 camera_ids: list = None, tick_rate: float = 30.0):
        super().__init__(daemon=True)
        self.game_state = game_state
        self.stream_manager = stream_manager
        self.camera_ids = camera_ids or []
        self.tick_rate = tick_rate         # target FPS for processing
        self._running = False
        self._active = True               # can be toggled from UI

    @abstractmethod
    def process_frame(self, camera_id: str, frame, frame_idx: int):
        """Process a single frame from one camera. Implement in subclass."""
        ...

    def on_start(self):
        """Optional hook called once when the agent starts."""
        pass

    def on_stop(self):
        """Optional hook called once when the agent stops."""
        pass

    def run(self):
        self._running = True
        self.on_start()
        print(f"[{self.AGENT_NAME}] Agent started (cameras: {self.camera_ids})")

        dt = 1.0 / self.tick_rate
        while self._running:
            if not self._active:
                time.sleep(0.1)
                continue

            for cam_id in self.camera_ids:
                frame, idx = self.stream_manager.get_frame(cam_id)
                if frame is not None:
                    try:
                        self.process_frame(cam_id, frame, idx)
                    except Exception as e:
                        print(f"[{self.AGENT_NAME}] Error processing {cam_id}: {e}")

            time.sleep(dt)

        self.on_stop()
        print(f"[{self.AGENT_NAME}] Agent stopped")

    def stop(self):
        self._running = False

    def set_active(self, active: bool):
        self._active = active

    @property
    def is_active(self):
        return self._active and self._running
