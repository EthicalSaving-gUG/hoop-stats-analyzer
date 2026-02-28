"""
roi_tracker.py
==============
Multi-object optical tracker wrapping OpenCV's CSRT algorithm.
Instead of running heavy YOLO on every frame, we run YOLO periodically
and then use CSRT to track the regions of interest (ROI) frame-by-frame.
"""

import cv2
from typing import Dict, List, Tuple

class MultiRoiTracker:
    def __init__(self):
        # {obj_id: cv2.TrackerCSRT}
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
        
        # Create a new CSRT Tracker (highly accurate for translation/scale)
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            tracker = cv2.TrackerCSRT_create()  # Fallback for some OpenCV versions
            
        # init() in python bindings may return None or True depending on version
        res = tracker.init(frame, bbox_xywh)
        success = res if res is not None else True
        
        if success:
            self.trackers[obj_id] = tracker
            
        return success

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
