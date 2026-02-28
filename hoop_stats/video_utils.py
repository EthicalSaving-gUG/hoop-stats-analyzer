"""
video_utils.py
==============
Helpers for splitting a video into segments so each segment can be
processed independently (in a separate process) by the fast analyzer.
"""

import cv2
from typing import Generator, List, Tuple


def get_video_info(video_path: str) -> dict:
    """Return basic metadata for the video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def get_segment_ranges(
    video_path: str,
    n_segments: int,
    max_frames: int = None,
) -> List[Tuple[int, int]]:
    """
    Divide the video into *n_segments* non-overlapping frame ranges.

    Parameters
    ----------
    video_path : str
    n_segments : int
        Number of segments (usually = cpu_count()).
    max_frames : int, optional
        If provided, only the first *max_frames* frames are considered.

    Returns
    -------
    List of (start_frame, end_frame) tuples (end is exclusive).
    """
    info = get_video_info(video_path)
    total = min(info["total_frames"], max_frames) if max_frames else info["total_frames"]
    n_segments = min(n_segments, total)

    chunk = total // n_segments
    ranges = []
    for i in range(n_segments):
        start = i * chunk
        end   = start + chunk if i < n_segments - 1 else total
        ranges.append((start, end))
    return ranges


def read_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    skip_frames: int = 1,
) -> Generator[Tuple[int, object], None, None]:
    """
    Yield (frame_idx, frame) for a given frame range.

    Parameters
    ----------
    video_path : str
    start_frame : int  – inclusive
    end_frame   : int  – exclusive
    skip_frames : int  – yield every Nth frame within the range
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame

    while idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_frame) % skip_frames == 0:
            yield idx, frame
        idx += 1

    cap.release()
