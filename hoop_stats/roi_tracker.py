"""
roi_tracker.py
==============
Multi-object optical tracker with OpenCV tracker support and a lightweight
template fallback.

Instead of running heavy YOLO on every frame, we run YOLO periodically and then
track regions of interest (ROI) frame-by-frame. CSRT is preferred when available,
but many OpenCV wheels ship without contrib trackers; the fallback keeps tests
and basic tracking usable on those installations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _create_opencv_tracker():
    """Return the best available OpenCV tracker, or None if none are installed."""
    namespaces = []
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        namespaces.append(legacy)
    namespaces.append(cv2)

    for namespace in namespaces:
        for factory_name in (
            "TrackerCSRT_create",
            "TrackerKCF_create",
            "TrackerMIL_create",
            "TrackerMOSSE_create",
        ):
            factory = getattr(namespace, factory_name, None)
            if factory is not None:
                return factory()
    return None


@dataclass
class _TemplateBackedTracker:
    opencv_tracker: object
    bbox_xywh: Tuple[int, int, int, int]
    template: np.ndarray
    misses: int = 0
    max_misses: int = 2
    min_confidence: float = 0.35

    @classmethod
    def create(cls, frame, bbox_xywh: Tuple[int, int, int, int]):
        tracker = _create_opencv_tracker()
        if tracker is not None:
            init_result = tracker.init(frame, bbox_xywh)
            if init_result is False:
                tracker = None

        x, y, w, h = bbox_xywh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = gray[y:y + h, x:x + w].copy()
        return cls(tracker, bbox_xywh, template)

    def update(self, frame) -> Tuple[bool, Tuple[float, float, float, float]]:
        candidate = None
        if self.opencv_tracker is not None:
            success, bbox = self.opencv_tracker.update(frame)
            if success:
                candidate = tuple(int(round(v)) for v in bbox)

        matched_bbox, confidence = self._match_template(frame, candidate)
        if matched_bbox is not None and confidence >= self.min_confidence:
            self.bbox_xywh = matched_bbox
            self.misses = 0
            return True, tuple(float(v) for v in matched_bbox)

        self.misses += 1
        if self.misses > self.max_misses:
            return False, tuple(float(v) for v in self.bbox_xywh)

        if candidate is not None:
            self.bbox_xywh = candidate
        return True, tuple(float(v) for v in self.bbox_xywh)

    def _match_template(self, frame, candidate):
        if self.template.size == 0:
            return None, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = candidate if candidate is not None else self.bbox_xywh
        fh, fw = gray.shape[:2]
        margin = max(w, h, 20)

        sx1 = max(0, x - margin)
        sy1 = max(0, y - margin)
        sx2 = min(fw, x + w + margin)
        sy2 = min(fh, y + h + margin)
        search = gray[sy1:sy2, sx1:sx2]

        if search.shape[0] < h or search.shape[1] < w:
            return None, 0.0

        if float(self.template.std()) < 1.0:
            loc, confidence = self._match_uniform_template(search, w, h)
        else:
            result = cv2.matchTemplate(search, self.template, cv2.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv2.minMaxLoc(result)
            loc = min_loc
            confidence = 1.0 - float(min_val)

        mx, my = loc
        matched = (sx1 + mx, sy1 + my, w, h)
        return matched, confidence

    def _match_uniform_template(self, search, w: int, h: int):
        target_mean = float(self.template.mean())
        ones = np.ones((h, w), dtype=np.float32)
        sums = cv2.matchTemplate(search.astype(np.float32), ones, cv2.TM_CCORR)

        if target_mean >= 127.5:
            _, best_sum, _, loc = cv2.minMaxLoc(sums)
        else:
            best_sum, _, loc, _ = cv2.minMaxLoc(sums)

        patch_mean = float(best_sum) / float(w * h)
        confidence = 1.0 - min(1.0, abs(patch_mean - target_mean) / 255.0)
        return loc, confidence

class MultiRoiTracker:
    def __init__(self):
        # {obj_id: tracker wrapper}
        self.trackers = {}
        
    def add_tracker(self, frame, bbox_xyxy: List[float], obj_id: int):
        """
        Initializes a CSRT tracker for a given bounding box.
        bbox_xyxy: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        w, h = x2 - x1, y2 - y1
        
        # Ensure box is valid and inside frame bounds
        fh, fw = frame.shape[:2]
        if w <= 0 or h <= 0 or x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
            return False
            
        bbox_xywh = (x1, y1, w, h)
        
        self.trackers[obj_id] = _TemplateBackedTracker.create(frame, bbox_xywh)
        return True

    def update(self, frame) -> Dict[int, List[float]]:
        """
        Updates all tracked objects with the new frame.
        Returns a dict of successful updates: {obj_id: [x1, y1, x2, y2]}
        """
        updated_bboxes = {}
        lost_ids = []
        
        for obj_id, tracker in self.trackers.items():
            success, bbox_xywh = tracker.update(frame)
            if success:
                x, y, w, h = bbox_xywh
                updated_bboxes[obj_id] = [x, y, x + w, y + h]
            else:
                lost_ids.append(obj_id)
                
        # Remove trackers that lost their target
        for obj_id in lost_ids:
            self.remove_tracker(obj_id)
            
        return updated_bboxes

    def remove_tracker(self, obj_id: int):
        if obj_id in self.trackers:
            del self.trackers[obj_id]

    def clear(self):
        """Remove all trackers (e.g., before refreshing from YOLO)."""
        self.trackers.clear()
        
    @property
    def active_count(self) -> int:
        return len(self.trackers)
