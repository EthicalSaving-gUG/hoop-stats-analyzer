"""
Referee Signals Agent.

Uses Google MediaPipe PoseLandmarker (Tasks API) to track 33 3D body
landmarks and detect FIBA referee hand signals:
1. Stop Clock (One arm straight up)
2. Substitution (Crossed forearms/X shape)
3. Timeout (T-shape with hands)
"""
import time
import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base_agent import BaseAgent


class RefereeSignalsAgent(BaseAgent):
    AGENT_NAME = "referee_signals"

    def __init__(self, game_state, stream_manager, camera_ids=None, audio_ids=None, **kwargs):
        # Run at 10-15 Hz to save CPU while still catching 1-2 second gestures
        super().__init__(game_state, stream_manager, camera_ids, audio_ids, tick_rate=10, **kwargs)
        self._landmarker = None
        self._current_gesture = None
        self._gesture_start_time = 0.0
        
        # Debounce: Must hold gesture for X seconds before we trigger an event
        self._required_hold_time = 0.5 
        
        # Cooldown: Prevent spamming the same event instantly
        self._last_event_time = 0.0
        self._cooldown = 3.0

    def on_start(self):
        # We reuse the same pose_landmarker.task from Phase 6
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
        if self._landmarker is None:
            return

        now = time.time()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self._landmarker.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            self._current_gesture = None
            return
            
        # Analyze the primary detected person (assumed to be the referee if isolated)
        landmarks = detection_result.pose_landmarks[0]
        
        detected_gesture = self._classify_referee_gesture(landmarks)
        
        if detected_gesture:
            if self._current_gesture == detected_gesture:
                # Holding the gesture
                if now - self._gesture_start_time > self._required_hold_time:
                    # Time to trigger
                    if now - self._last_event_time > self._cooldown:
                        self._trigger_event(detected_gesture, frame_idx)
                        self._last_event_time = now
            else:
                # New gesture started
                self._current_gesture = detected_gesture
                self._gesture_start_time = now
        else:
            self._current_gesture = None

    def _trigger_event(self, gesture: str, frame_idx: int):
        print(f"[{self.AGENT_NAME}] Recognized FIBA Signal: {gesture}")
        self.game_state.append_event(
            self.AGENT_NAME, "REFEREE_SIGNAL",
            {
                "signal": gesture,
                "confidence": "High (AI Pose)"
            },
            frame_idx
        )

    def _classify_referee_gesture(self, marks) -> str:
        """
        Evaluate heuristics for FIBA hand signals using 3D coordinates.
        Note: Origin (0,0) is top-left. Y increases downwards.
        """
        
        # 11: l_shoulder, 12: r_shoulder
        # 13: l_elbow,    14: r_elbow
        # 15: l_wrist,    16: r_wrist
        ls, rs = marks[11], marks[12]
        le, re = marks[13], marks[14]
        lw, rw = marks[15], marks[16]

        # Basic presence check
        if ls.visibility < 0.5 or rs.visibility < 0.5:
            return None

        # Helper: Distance
        def dist(p1, p2):
            return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

        shoulder_width = dist(ls, rs)
        if shoulder_width < 0.01:
            return None

        # -- Check 1: Stop Clock (Right or Left hand straight up) --
        # Hand must be significantly above elbow, and elbow above shoulder
        r_arm_up = rw.y < re.y and re.y < rs.y and (rs.y - rw.y) > shoulder_width
        l_arm_up = lw.y < le.y and le.y < ls.y and (ls.y - lw.y) > shoulder_width
        
        if r_arm_up or l_arm_up:
            # Differentiate from other gestures: make sure it's purely vertical
            # Not crossing the body center heavily.
            return "Stop Clock (Foul / Violation)"

        # -- Check 2: Substitution (Crossed wrists in front of chest) --
        # X-shape constraint: Wrists close to each other, hands near shoulder level
        wrists_dist = dist(lw, rw)
        wrists_crossed = wrists_dist < (shoulder_width * 0.5)
        
        hands_at_chest_level = (lw.y < ls.y + 0.1) and (rw.y < rs.y + 0.1) and (lw.y > ls.y - 0.4)
        
        if wrists_crossed and hands_at_chest_level:
            return "Substitution / Auswechslung"

        # -- Check 3: Timeout (T-Shape) --
        # One hand horizontal (index finger to palm), one vertical.
        # Heuristic: hands are touching or very close, but elbows are far apart
        hands_touching = wrists_dist < (shoulder_width * 0.6)
        elbows_far = dist(le, re) > (shoulder_width * 0.8)
        hands_high = lw.y < ls.y and rw.y < rs.y  # T-shape usually done around face/neck level
        
        if hands_touching and elbows_far and hands_high and not wrists_crossed:
            return "Timeout"

        return None
