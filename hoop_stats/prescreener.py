"""
prescreener.py
==============
Lightweight pre-screening pass applied to every raw frame BEFORE the
expensive YOLO model is called.

Three independent filters:
  1. Motion filter  – skip frames with too little movement (dead time).
  2. Blur filter    – skip frames that are too blurry to yield good detections.
  3. Scene-cut detector – flag hard cuts so the tracker can be reset.

Usage
-----
    screener = FramePrescreener(
        motion_threshold=0.02,   # fraction of changed pixels
        blur_threshold=80.0,     # Laplacian variance minimum
        scene_cut_threshold=0.40 # fraction of changed pixels for a cut
    )

    for idx, frame in video_frames:
        result = screener.process(idx, frame)
        if result.skip:
            continue
        if result.scene_cut:
            tracker.reset()
        run_yolo(frame)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScreenResult:
    frame_idx: int
    skip: bool = False               # True → do not send to YOLO
    skip_reason: str = ""            # "motion" | "blur" | ""
    scene_cut: bool = False          # True → hard camera cut detected
    motion_score: float = 0.0        # fraction of changed pixels
    blur_score: float = 0.0          # Laplacian variance (higher = sharper)


class FramePrescreener:
    """
    Fast per-frame quality and activity filter.

    Parameters
    ----------
    motion_threshold : float
        Minimum fraction of pixels that must change between consecutive frames
        for the frame to be considered 'active'. Default 0.02 (2 %).
    blur_threshold : float
        Minimum Laplacian variance. Frames below this are considered blurry.
        Typical values: outdoor sports ≥ 60, indoor ≥ 40.
    scene_cut_threshold : float
        Fraction of changed pixels that indicates a hard camera cut.
        Default 0.40 (40 %).
    downsample : int
        Factor by which frames are downsampled before analysis to
        reduce the cost of the pre-screener itself.
    """

    def __init__(
        self,
        motion_threshold: float = 0.02,
        blur_threshold: float = 60.0,
        scene_cut_threshold: float = 0.40,
        downsample: int = 4,
    ):
        self.motion_threshold = motion_threshold
        self.blur_threshold = blur_threshold
        self.scene_cut_threshold = scene_cut_threshold
        self.downsample = downsample

        self._prev_gray: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def process(self, frame_idx: int, frame: np.ndarray) -> ScreenResult:
        result = ScreenResult(frame_idx=frame_idx)

        # --- Downsample for speed ----------------------------------------
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame,
            (w // self.downsample, h // self.downsample),
            interpolation=cv2.INTER_NEAREST,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # --- Blur filter -------------------------------------------------
        result.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if result.blur_score < self.blur_threshold:
            result.skip = True
            result.skip_reason = "blur"
            self._prev_gray = gray
            return result

        # --- Motion / scene-cut filter -----------------------------------
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, thresh = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
            motion_frac = float(thresh.mean())
            result.motion_score = motion_frac

            if motion_frac >= self.scene_cut_threshold:
                result.scene_cut = True  # hard cut – do NOT skip
            elif motion_frac < self.motion_threshold:
                result.skip = True
                result.skip_reason = "motion"
                self._prev_gray = gray
                return result
        else:
            result.motion_score = 1.0  # first frame always passes

        self._prev_gray = gray
        return result

    # ------------------------------------------------------------------ #
    def reset(self):
        """Reset internal state (call after a scene cut or seek)."""
        self._prev_gray = None

    # ------------------------------------------------------------------ #
    def stats_summary(self, results: list) -> dict:
        """Summarise a list of ScreenResults for logging / benchmarking."""
        total = len(results)
        skipped_motion = sum(1 for r in results if r.skip_reason == "motion")
        skipped_blur   = sum(1 for r in results if r.skip_reason == "blur")
        scene_cuts     = sum(1 for r in results if r.scene_cut)
        processed      = total - skipped_motion - skipped_blur
        return {
            "total_frames":    total,
            "processed":       processed,
            "skipped_motion":  skipped_motion,
            "skipped_blur":    skipped_blur,
            "scene_cuts":      scene_cuts,
            "skip_rate_pct":   round(100 * (1 - processed / max(total, 1)), 1),
        }
