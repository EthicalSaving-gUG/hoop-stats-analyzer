"""
conftest.py — shared pytest fixtures for hoop-stats-analyzer tests.
"""
import os
import numpy as np
import pytest
import cv2

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_VIDEO = os.path.join(
    ROOT_DIR, "uploads", "Veo Bringe dein Spiel auf das nachste Level.mp4"
)


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def upload_video_path():
    """Return path to the upload video, or skip the test if not present."""
    if not os.path.isfile(UPLOAD_VIDEO):
        pytest.skip(f"Upload video not found: {UPLOAD_VIDEO}")
    return UPLOAD_VIDEO


@pytest.fixture(scope="session")
def first_frame(upload_video_path):
    """Extract and return the very first frame of the upload video."""
    cap = cv2.VideoCapture(upload_video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Could not read first frame from upload video"
    return frame


@pytest.fixture
def black_frame():
    """A completely black 720×1280 BGR frame (no motion, very blurry)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """A completely white 720×1280 BGR frame."""
    return np.full((720, 1280, 3), 255, dtype=np.uint8)


@pytest.fixture
def noisy_frame():
    """Random noise frame – high motion signal and reasonable sharpness."""
    return np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def solid_color_crop():
    """
    Return a factory function that builds a BGR crop filled with a given
    BGR color. Useful for testing jersey color functions.
    """
    def _make(b, g, r, size=(64, 64)):
        crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        crop[:, :] = (b, g, r)
        return crop
    return _make
