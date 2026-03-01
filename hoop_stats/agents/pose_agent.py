"""
MediaPipe Pose Analysis Agent (Tasks API v0.10+).

Tracks 33 3D body landmarks for the active player using the modern
PoseLandmarker Tasks API. Calculates shooting biomechanics (elbow angle,
jump height) and verifies 3-point shot footing.
"""
import time
import cv2
import math
import numpy as np
import threading
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base_agent import BaseAgent

def calculate_angle(a, b, c):
    """
    Calculate the 2D angle between three landmarks: a, b, c.
    Returns the angle in degrees (0 - 180).
    """
    a = np.array([a.x, a.y]) # First point
    b = np.array([b.x, b.y]) # Mid point (vertex)
    c = np.array([c.x, c.y]) # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class PoseAnalysisAgent(BaseAgent):
    AGENT_NAME = "pose"

    def __init__(self, game_state, stream_manager, camera_ids=None, audio_ids=None, **kwargs):
        super().__init__(game_state, stream_manager, camera_ids, audio_ids, tick_rate=15, **kwargs)
        self._landmarker = None
        self._shooting_motion = False
        
        # Keep track of active shooter for the dashboard
        self._active_shooter_stats = {
            "elbow_angle": 0.0,
            "knee_angle": 0.0,
            "status": "Idle"
        }

    def on_start(self):
        # Initialize MediaPipe PoseLandmarker (Tasks API)
        model_path = os.path.join(os.getcwd(), 'pose_landmarker.task')
        if not os.path.exists(model_path):
            print(f"[{self.AGENT_NAME}] WARNING: Model 'pose_landmarker.task' not found. Agent disabled.")
            return

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def on_stop(self):
        if self._landmarker is not None:
            self._landmarker.close()

    def process_frame(self, camera_id, frame, frame_idx):
        gs = self.game_state
        
        if self._landmarker is None:
            return

        # Only do heavy pose processing if the game is active 
        if not gs.game_clock_running and self._active_shooter_stats["status"] == "Idle":
            return

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Wrap the NumPy array in a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose landmarks
        detection_result = self._landmarker.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            if self._shooting_motion:
                self._shooting_motion = False
                self._active_shooter_stats["status"] = "Idle"
            return
            
        landmarks = detection_result.pose_landmarks[0] # Use first tracked person

        # MP Pose Landmarks definition for Tasks API
        # 11: left_shoulder, 13: left_elbow, 15: left_wrist
        # 23: left_hip, 25: left_knee, 27: left_ankle
        l_shoulder = landmarks[11]
        l_elbow = landmarks[13]
        l_wrist = landmarks[15]
        
        # Calculate elbow angle (Release angle)
        elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        self._active_shooter_stats["elbow_angle"] = round(elbow_angle, 1)

        # Knees for jump detection
        l_hip = landmarks[23]
        l_knee = landmarks[25]
        l_ankle = landmarks[27]
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        self._active_shooter_stats["knee_angle"] = round(knee_angle, 1)

        # Detect the "Act of Shooting" (elbow extending, arms raising above shoulder)
        is_arms_up = l_wrist.y < l_shoulder.y  # y is inverted in image coords (smaller is higher)
        is_elbow_extended = elbow_angle > 140.0
        
        if is_arms_up and is_elbow_extended:
            if not self._shooting_motion:
                self._shooting_motion = True
                self._active_shooter_stats["status"] = "Act of Shooting"
                
                # Check 3-point line (Mock: check if feet are in bottom 20% of frame)
                is_3pt = l_ankle.y > 0.8 
                
                gs.append_event(
                    self.AGENT_NAME, "SHOOTING_MOTION",
                    {
                        "biomechanics": {
                            "release_angle": round(elbow_angle, 1),
                            "knee_angle": round(knee_angle, 1)
                        },
                        "is_3pt_attempt": is_3pt
                    },
                    frame_idx
                )
        else:
            if self._shooting_motion and elbow_angle < 100:
                self._shooting_motion = False
                self._active_shooter_stats["status"] = "Idle"

        # Update the live state so the dashboard can animate the angles
        gs.player_stats["biomechanics_live"] = self._active_shooter_stats
