"""test_prescreener.py — tests for hoop_stats.prescreener"""
import numpy as np
import pytest
from hoop_stats.prescreener import FramePrescreener


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestBlurFilter:
    def test_blurry_black_frame_is_skipped(self, black_frame):
        screener = FramePrescreener(blur_threshold=60.0)
        result = screener.process(0, black_frame)
        assert result.skip is True
        assert result.skip_reason == "blur"

    def test_sharp_noisy_frame_passes(self, noisy_frame):
        screener = FramePrescreener(blur_threshold=60.0)
        result = screener.process(0, noisy_frame)
        assert result.skip is False or result.skip_reason == "motion"
        # blur_score should be high for noisy data
        assert result.blur_score > 60.0


class TestMotionFilter:
    def test_static_scene_skipped_after_first(self, noisy_frame):
        """Two identical noise frames → second one has zero motion → skipped."""
        screener = FramePrescreener(motion_threshold=0.02, blur_threshold=10.0)
        screener.process(0, noisy_frame)                  # prime
        identical = noisy_frame.copy()
        result = screener.process(1, identical)
        assert result.skip is True
        assert result.skip_reason == "motion"

    def test_first_frame_never_skipped_for_motion(self, noisy_frame):
        """The very first frame has no previous reference so motion skip is N/A."""
        screener = FramePrescreener(motion_threshold=0.99, blur_threshold=10.0)
        result = screener.process(0, noisy_frame)
        assert result.skip is False


class TestSceneCutDetection:
    def test_hard_cut_flagged(self, noisy_frame, black_frame):
        """Black → noise transition is a scene cut (large fraction change)."""
        import numpy as np
        screener = FramePrescreener(
            scene_cut_threshold=0.30,
            blur_threshold=10.0,
            motion_threshold=0.01,
        )
        # Prime with a pure black frame (then override blur threshold)
        screener._prev_gray = np.zeros((180, 320), dtype=np.uint8)  # pre-set prev
        result = screener.process(1, noisy_frame)
        assert result.scene_cut is True

    def test_no_cut_on_gradual_change(self, noisy_frame):
        """Slightly dimmed version of same frame → not a cut."""
        screener = FramePrescreener(scene_cut_threshold=0.40, blur_threshold=10.0)
        screener.process(0, noisy_frame)
        dim = (noisy_frame * 0.95).astype("uint8")
        result = screener.process(1, dim)
        assert result.scene_cut is False


class TestReset:
    def test_reset_clears_prev_gray(self, noisy_frame):
        screener = FramePrescreener(blur_threshold=10.0)
        screener.process(0, noisy_frame)
        assert screener._prev_gray is not None
        screener.reset()
        assert screener._prev_gray is None


class TestStatsSummary:
    def test_summary_structure(self, noisy_frame, black_frame):
        screener = FramePrescreener(blur_threshold=60.0)
        results = [
            screener.process(i, black_frame if i % 2 == 0 else noisy_frame)
            for i in range(6)
        ]
        summary = screener.stats_summary(results)
        assert "total_frames" in summary
        assert "skip_rate_pct" in summary
        assert summary["total_frames"] == 6


# ── Integration test — processes a clip of the real video ────────────────────

def test_prescreener_on_real_video(upload_video_path):
    import cv2
    screener = FramePrescreener()
    results = []
    cap = cv2.VideoCapture(upload_video_path)
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        results.append(screener.process(i, frame))
    cap.release()

    assert len(results) > 0
    summary = screener.stats_summary(results)
    # At least some frames should survive the pre-screener
    assert summary["processed"] > 0
    print("\nPrescreener summary (100 frames):", summary)
