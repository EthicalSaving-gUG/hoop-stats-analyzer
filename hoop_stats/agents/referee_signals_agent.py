"""
Referee Signals Agent.

Uses Google MediaPipe PoseLandmarker (Tasks API) to track 33 3D body
landmarks and detect FIBA/NBA referee hand signals:
1. Stop Clock (One arm straight up)
2. Substitution (Crossed forearms/X shape)
3. Timeout (T-shape with hands)
4. Jump Ball (Both arms straight up)
5. 24-Second Violation (Tap head)
6. Blocking Foul (Hands on hips)

CRITICAL RULE:
Gesture recognition is strictly restricted to verified referees (identified by referee characteristics:
official DBB/FIBA grey shirt, long black trousers/pants, absence of player jersey numbers).
Gestures performed by regular basketball players are ignored.
"""
import time
import cv2
import numpy as np
import os
from typing import Any, Optional

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base_agent import BaseAgent
from ..referee_detector import RefereeDetector, RefereeClassification


class RefereeSignalsAgent(BaseAgent):
    AGENT_NAME = "referee_signals"

    def __init__(self, game_state, stream_manager, camera_ids=None, audio_ids=None, **kwargs):
        # Run at 10-15 Hz to save CPU while still catching 1-2 second gestures
        self._enforce_referee_check = kwargs.pop("enforce_referee_check", True)
        super().__init__(game_state, stream_manager, camera_ids, audio_ids, tick_rate=10, **kwargs)
        self._landmarker = None
        self._current_gesture = None
        self._current_referee_info = None
        self._gesture_start_time = 0.0

        # Debounce: Must hold gesture for X seconds before we trigger an event
        self._required_hold_time = 0.5 

        # Cooldown: Prevent spamming the same event instantly
        self._last_event_time = 0.0
        self._cooldown = 3.0

        # Referee Appearance Detector (grey shirt, long pants, etc.)
        self.referee_detector = RefereeDetector()

    def on_start(self):
        # We reuse the same pose_landmarker.task
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
        if self._landmarker is None or frame is None:
            return

        now = time.time()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = self._landmarker.detect(mp_image)

        if not detection_result.pose_landmarks:
            self._current_gesture = None
            self._current_referee_info = None
            return

        ruleset = self.game_state.rules.name.lower()

        # Iterate over all detected persons in the frame.
        # GESTURE RECOGNITION MUST ONLY APPLY TO CONFIRMED REFEREES!
        matched_gesture = None
        matched_ref_info = None

        for landmarks in detection_result.pose_landmarks:
            # 1. Verify referee characteristics (grey shirt, long pants, etc.)
            if self._enforce_referee_check:
                ref_eval = self.referee_detector.evaluate_from_landmarks(frame, landmarks)
                if not ref_eval.is_referee:
                    # Person is a player or spectator — do NOT process referee gestures on them!
                    continue
            else:
                ref_eval = None

            # 2. This person is an authentic referee: classify gesture
            gesture = self._classify_referee_gesture(landmarks, ruleset)
            if gesture:
                matched_gesture = gesture
                matched_ref_info = ref_eval
                break

        if matched_gesture:
            if self._current_gesture == matched_gesture:
                # Holding the gesture
                if now - self._gesture_start_time > self._required_hold_time:
                    # Time to trigger
                    if now - self._last_event_time > self._cooldown:
                        self._trigger_event(matched_gesture, frame_idx, ref_info=matched_ref_info)
                        self._last_event_time = now
            else:
                # New gesture started
                self._current_gesture = matched_gesture
                self._current_referee_info = matched_ref_info
                self._gesture_start_time = now
        else:
            self._current_gesture = None
            self._current_referee_info = None

    def process_person_pose(self, frame: Optional[np.ndarray], landmarks: Any, ruleset: str = "fiba") -> Optional[str]:
        """
        Public evaluation method for a single person pose.
        Enforces referee verification: returns None if person is not a verified referee.
        """
        if frame is not None and self._enforce_referee_check:
            ref_eval = self.referee_detector.evaluate_from_landmarks(frame, landmarks)
            if not ref_eval.is_referee:
                return None
        return self._classify_referee_gesture(landmarks, ruleset)

    def _trigger_event(self, gesture: str, frame_idx: int, ref_info: Optional[RefereeClassification] = None):
        print(f"[{self.AGENT_NAME}] Recognized Signal from Verified Referee: {gesture}")
        event_data = {
            "signal": gesture,
            "referee_verified": True,
            "confidence": "High (Verified Referee)",
            "ruleset": self.game_state.rules.name
        }
        if ref_info:
            event_data["referee_confidence"] = ref_info.confidence
            event_data["shirt_color"] = ref_info.shirt_color
            event_data["pants_type"] = ref_info.pants_type
            if ref_info.bbox:
                event_data["referee_bbox"] = list(ref_info.bbox)

        self.game_state.append_event(
            self.AGENT_NAME, "REFEREE_SIGNAL",
            event_data,
            frame_idx
        )

    def _classify_referee_gesture(self, marks, ruleset: str) -> Optional[str]:
        """
        Evaluate heuristics for FIBA/NBA hand signals using 3D coordinates.
        Note: Origin (0,0) is top-left. Y increases downwards.
        """
        nose = marks[0]
        ls, rs = marks[11], marks[12]
        le, re = marks[13], marks[14]
        lw, rw = marks[15], marks[16]
        lh, rh = marks[23], marks[24]

        # Basic presence check for upper body
        if getattr(ls, "visibility", 1.0) < 0.4 or getattr(rs, "visibility", 1.0) < 0.4:
            return None

        # Helper: Distance
        def dist(p1, p2):
            return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

        shoulder_width = dist(ls, rs)
        if shoulder_width < 0.01:
            return None

        shoulder_y = (ls.y + rs.y) / 2

        # Estimate waist if hips are not visible
        if getattr(lh, "visibility", 1.0) < 0.4 or getattr(rh, "visibility", 1.0) < 0.4:
            waist_y = shoulder_y + shoulder_width * 1.5
        else:
            waist_y = (lh.y + rh.y) / 2

        # Helper: Arm up check (wrist above elbow, elbow above shoulder)
        def is_arm_up(w, e, s):
            return w.y < e.y and e.y < s.y and (s.y - w.y) > shoulder_width * 0.8

        l_up = is_arm_up(lw, le, ls)
        r_up = is_arm_up(rw, re, rs)
        wrists_dist = dist(lw, rw)

        # 1. Both Arms Up (Jump Ball)
        if l_up and r_up:
            return "Jump Ball"

        # 2. Touch Head (24-Second Violation)
        def touches_head(w):
            return dist(w, nose) < shoulder_width * 0.6 and w.y < ls.y

        if (touches_head(lw) and not touches_head(rw)) or (touches_head(rw) and not touches_head(lw)):
            if not l_up and not r_up:
                return "24-Second Violation"

        # 3. Hands on Shoulders (NBA Full Timeout)
        if ruleset == "nba":
            def touches_shoulder(w):
                return dist(w, ls) < shoulder_width * 0.5 or dist(w, rs) < shoulder_width * 0.5
            if touches_shoulder(lw) and touches_shoulder(rw) and not l_up and not r_up:
                if le.y > shoulder_y and re.y > shoulder_y:
                    return "Full Timeout (NBA)"

        # 4. Hands on Hips (Blocking Foul)
        wrists_near_waist = abs(lw.y - waist_y) < shoulder_width * 0.6 and abs(rw.y - waist_y) < shoulder_width * 0.6
        elbows_wide = dist(le, re) > shoulder_width * 1.2
        if wrists_near_waist and elbows_wide and not l_up and not r_up:
            if ruleset == "nba":
                return "Blocking Foul (NBA)"
            return "Blocking / Illegal Screen (FIBA)"

        # 5. Substitution / Crossed Arms
        wrists_crossed = wrists_dist < (shoulder_width * 0.5)
        hands_at_chest = (lw.y > shoulder_y - shoulder_width * 0.4) and (lw.y < waist_y - shoulder_width * 0.2)
        elbows_not_too_far = dist(le, re) < shoulder_width * 1.2
        if wrists_crossed and hands_at_chest and elbows_not_too_far:
            if ruleset == "nba":
                return "Substitution / Illegal Screen (NBA)"
            return "Substitution (FIBA)"

        # 6. T-Shape (Timeout / Technical Foul)
        hands_touching = wrists_dist < (shoulder_width * 0.7)
        elbows_far = dist(le, re) > (shoulder_width * 0.9)
        hands_high = lw.y < shoulder_y + shoulder_width * 0.5 and rw.y < shoulder_y + shoulder_width * 0.5
        if hands_touching and elbows_far and hands_high and not wrists_crossed:
            if ruleset == "nba":
                return "20-Sec Timeout / Tech (NBA)"
            return "Timeout / Tech Foul (FIBA)"

        # 7. One Arm Up (Stop Clock)
        if l_up or r_up:
            return "Stop Clock (Foul / Violation)"

        return None
