"""
Multi-camera stream manager.
Opens each camera source in a dedicated reader thread and
provides the latest frame to any agent that requests it.
"""
import cv2
import threading
import time
from typing import List, Optional, Union
import numpy as np


class CameraReader(threading.Thread):
    """Continuously reads frames from a single video source."""

    def __init__(self, source: Union[str, int], camera_id: str):
        super().__init__(daemon=True)
        self.source = source
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_idx: int = 0
        self._running = False
        self._fps: float = 30.0
        self._width: int = 0
        self._height: int = 0

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[CameraReader:{self.camera_id}] ERROR: Cannot open {self.source}")
            return

        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._running = True

        print(f"[CameraReader:{self.camera_id}] Opened {self.source} "
              f"({self._width}x{self._height} @ {self._fps:.1f}fps)")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                # For files: loop or stop
                if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # For streams: try reconnect
                    time.sleep(1.0)
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
                    continue

            with self._lock:
                self._frame = frame
                self._frame_idx += 1

        cap.release()

    def stop(self):
        self._running = False

    def get_frame(self):
        """Returns (frame, frame_idx) or (None, 0)."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy(), self._frame_idx
            return None, 0

    @property
    def fps(self):
        return self._fps

    @property
    def resolution(self):
        return (self._width, self._height)


class StreamManager:
    """
    Manages multiple camera sources, each in its own thread.

    Usage:
        mgr = StreamManager()
        mgr.add_camera("cam_overview", "rtsp://192.168.1.10/stream")
        mgr.add_camera("cam_left", "uploads/left_basket.mp4")
        mgr.add_camera("cam_right", 0)  # webcam
        mgr.start_all()

        frame, idx = mgr.get_frame("cam_overview")
    """

    def __init__(self):
        self._cameras: dict[str, CameraReader] = {}

    def add_camera(self, camera_id: str, source: Union[str, int]):
        reader = CameraReader(source, camera_id)
        self._cameras[camera_id] = reader

    def start_all(self):
        for reader in self._cameras.values():
            reader.start()

    def stop_all(self):
        for reader in self._cameras.values():
            reader.stop()
        for reader in self._cameras.values():
            reader.join(timeout=3.0)

    def get_frame(self, camera_id: str):
        """Returns (frame, frame_idx) from the named camera."""
        reader = self._cameras.get(camera_id)
        if reader is None:
            return None, 0
        return reader.get_frame()

    def list_cameras(self) -> List[str]:
        return list(self._cameras.keys())

    def get_info(self, camera_id: str) -> dict:
        reader = self._cameras.get(camera_id)
        if reader is None:
            return {}
        return {
            "camera_id": camera_id,
            "source": str(reader.source),
            "fps": reader.fps,
            "resolution": reader.resolution,
        }
