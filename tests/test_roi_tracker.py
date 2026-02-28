import pytest
import numpy as np
import cv2
from hoop_stats.roi_tracker import MultiRoiTracker

def test_add_tracker_success():
    tracker = MultiRoiTracker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Valid box inside frame
    bbox = [100, 100, 200, 200]  
    success = tracker.add_tracker(frame, bbox, obj_id=1)
    
    assert success is True
    assert tracker.active_count == 1
    assert 1 in tracker.trackers

def test_add_tracker_invalid_box():
    tracker = MultiRoiTracker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Negative width
    bbox1 = [200, 100, 100, 200]
    success1 = tracker.add_tracker(frame, bbox1, obj_id=1)
    
    # Outside frame
    bbox2 = [-10, -10, 50, 50]
    success2 = tracker.add_tracker(frame, bbox2, obj_id=2)
    
    assert success1 is False
    assert success2 is False
    assert tracker.active_count == 0

def test_tracker_update_movement():
    tracker = MultiRoiTracker()
    
    # Create a frame with a white square on a black background
    frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (50, 50), (100, 100), (255, 255, 255), -1)
    
    bbox = [50, 50, 100, 100]
    tracker.add_tracker(frame1, bbox, obj_id=42)
    
    # Move the square to the right
    frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (60, 50), (110, 100), (255, 255, 255), -1)
    
    updated = tracker.update(frame2)
    
    assert 42 in updated
    new_box = updated[42]
    # Check if the tracker followed the x-translation
    assert new_box[0] > 50
    assert new_box[2] > 100

def test_tracker_lost_target():
    tracker = MultiRoiTracker()
    
    frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (50, 50), (100, 100), (255, 255, 255), -1)
    
    tracker.add_tracker(frame1, [50, 50, 100, 100], obj_id=99)
    
    # Completely black frame = severe occlusion / target lost
    frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Note: CSRT might still guess the location if movement is small,
    # but over a few frames of total blackness it should fail.
    for _ in range(5):
        updated = tracker.update(frame2)
        
    # Assuming it lost tracking, it should remove the ID and return empty dict
    assert 99 not in updated
    assert tracker.active_count == 0

def test_clear_trackers():
    tracker = MultiRoiTracker()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    
    tracker.add_tracker(frame, [10, 10, 50, 50], obj_id=1)
    tracker.add_tracker(frame, [60, 60, 100, 100], obj_id=2)
    
    assert tracker.active_count == 2
    tracker.clear()
    assert tracker.active_count == 0
