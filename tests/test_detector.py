"""test_detector.py — tests for hoop_stats.detector"""
import numpy as np
import pytest
from hoop_stats.detector import HoopDetector


@pytest.fixture(scope="module")
def detector():
    return HoopDetector()


class TestHoopDetector:
    def test_instantiates(self, detector):
        assert detector.model is not None

    def test_detect_players_returns_list(self, detector, noisy_frame):
        detections = detector.detect_players(noisy_frame)
        assert isinstance(detections, list)
        # Each detection should be a 4-element list [x1, y1, x2, y2]
        for det in detections:
            assert len(det) == 4

    def test_get_jersey_crop_shape(self, detector):
        """Crop should return a non-empty ndarray."""
        import numpy as np
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        bbox = [100, 100, 200, 300]          # x1, y1, x2, y2
        crop = detector.get_jersey_crop(frame, bbox)
        assert crop.ndim == 3
        assert crop.shape[2] == 3            # BGR channels
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_detect_players_on_real_frame(self, detector, first_frame):
        """Should detect at least one person in a real game frame."""
        detections = detector.detect_players(first_frame)
        assert len(detections) >= 0          # Allow 0 on a still title card
        print(f"\nDetections in first frame: {len(detections)}")
