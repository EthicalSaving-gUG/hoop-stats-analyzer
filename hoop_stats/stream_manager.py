"""
Multi-camera stream manager with audio support.
Opens each camera source in a dedicated reader thread and
provides the latest frame to any agent that requests it.
Also supports audio capture for whistle detection.
"""
import cv2
import subprocess
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
                if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
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


class AudioReader(threading.Thread):
    """
    Extracts audio from a video file using ffmpeg or records from a microphone.
    Produces raw PCM float32 chunks for the WhistleDetector.
    """

    def __init__(self, source: str, audio_id: str = "audio",
                 sample_rate: int = 44100, chunk_duration: float = 0.1):
        super().__init__(daemon=True)
        self.source = source
        self.audio_id = audio_id
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self._lock = threading.Lock()
        self._chunk: Optional[np.ndarray] = None
        self._running = False

    def run(self):
        self._running = True

        # Try to use ffmpeg to extract audio from the video source
        if isinstance(self.source, str) and not self.source.startswith("mic:"):
            self._run_ffmpeg()
        else:
            self._run_microphone()

    def _run_ffmpeg(self):
        """Extract audio from a video file using ffmpeg subprocess."""
        cmd = [
            "ffmpeg", "-i", self.source,
            "-f", "f32le",           # raw float32 PCM
            "-acodec", "pcm_f32le",
            "-ac", "1",              # mono
            "-ar", str(self.sample_rate),
            "-v", "quiet",
            "pipe:1"
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            print(f"[AudioReader:{self.audio_id}] Extracting audio from {self.source}")
        except FileNotFoundError:
            print(f"[AudioReader:{self.audio_id}] WARNING: ffmpeg not found, audio disabled")
            return

        bytes_per_chunk = self.chunk_size * 4  # 4 bytes per float32
        while self._running:
            raw = proc.stdout.read(bytes_per_chunk)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.float32)
            with self._lock:
                self._chunk = chunk

        proc.terminate()
        print(f"[AudioReader:{self.audio_id}] Audio extraction finished")

    def _run_microphone(self):
        """Record from system microphone using sounddevice."""
        try:
            import sounddevice as sd
            print(f"[AudioReader:{self.audio_id}] Recording from microphone")
            while self._running:
                chunk = sd.rec(self.chunk_size, samplerate=self.sample_rate,
                               channels=1, dtype='float32', blocking=True)
                with self._lock:
                    self._chunk = chunk.flatten()
        except ImportError:
            print(f"[AudioReader:{self.audio_id}] WARNING: sounddevice not installed, mic disabled")
        except Exception as e:
            print(f"[AudioReader:{self.audio_id}] Mic error: {e}")

    def stop(self):
        self._running = False

    def get_chunk(self) -> Optional[np.ndarray]:
        """Returns the latest audio chunk or None."""
        with self._lock:
            chunk = self._chunk
            self._chunk = None  # consume it
            return chunk


class StreamManager:
    """
    Manages multiple camera + audio sources.

    Usage:
        mgr = StreamManager()
        mgr.add_camera("cam_overview", "rtsp://192.168.1.10/stream")
        mgr.add_audio("audio_main", "uploads/game.mp4")
        mgr.start_all()

        frame, idx = mgr.get_frame("cam_overview")
        chunk = mgr.get_audio_chunk("audio_main")
    """

    def __init__(self):
        self._cameras: dict[str, CameraReader] = {}
        self._audio: dict[str, AudioReader] = {}

    def add_camera(self, camera_id: str, source: Union[str, int]):
        reader = CameraReader(source, camera_id)
        self._cameras[camera_id] = reader

    def add_audio(self, audio_id: str, source: str, sample_rate: int = 44100):
        reader = AudioReader(source, audio_id, sample_rate)
        self._audio[audio_id] = reader

    def start_all(self):
        for reader in self._cameras.values():
            reader.start()
        for reader in self._audio.values():
            reader.start()

    def stop_all(self):
        for reader in self._cameras.values():
            reader.stop()
        for reader in self._audio.values():
            reader.stop()
        for reader in self._cameras.values():
            reader.join(timeout=3.0)
        for reader in self._audio.values():
            reader.join(timeout=3.0)

    def get_frame(self, camera_id: str):
        reader = self._cameras.get(camera_id)
        if reader is None:
            return None, 0
        return reader.get_frame()

    def get_audio_chunk(self, audio_id: str) -> Optional[np.ndarray]:
        reader = self._audio.get(audio_id)
        if reader is None:
            return None
        return reader.get_chunk()

    def list_cameras(self) -> List[str]:
        return list(self._cameras.keys())

    def list_audio(self) -> List[str]:
        return list(self._audio.keys())

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

