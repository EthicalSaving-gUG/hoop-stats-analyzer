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
        ruleset = self.game_state.rules.name.lower()

        detected_gesture = self._classify_referee_gesture(landmarks, ruleset)

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
        print(f"[{self.AGENT_NAME}] Recognized Signal: {gesture}")
        self.game_state.append_event(
            self.AGENT_NAME, "REFEREE_SIGNAL",
            {
                "signal": gesture,
                "confidence": "High (AI Pose)"
            },
            frame_idx
        )

    @staticmethod
    def _dist(p1, p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

    @staticmethod
    def _is_arm_up(w, e, s, shoulder_width):
        return w.y < e.y and e.y < s.y and (s.y - w.y) > shoulder_width * 0.8

    def _check_jump_ball(self, l_up, r_up):
        if l_up and r_up:
            return "Jump Ball"
        return None

    def _check_24_second_violation(self, nose, lw, rw, ls, l_up, r_up, shoulder_width):
        def touches_head(w):
            return self._dist(w, nose) < shoulder_width * 0.6 and w.y < ls.y

        if (touches_head(lw) and not touches_head(rw)) or (touches_head(rw) and not touches_head(lw)):
            if not l_up and not r_up:
                return "24-Second Violation"
        return None

    def _check_nba_full_timeout(self, lw, rw, ls, rs, le, re, l_up, r_up, shoulder_width, shoulder_y, ruleset):
        if ruleset == "nba":
            def touches_shoulder(w):
                return self._dist(w, ls) < shoulder_width * 0.5 or self._dist(w, rs) < shoulder_width * 0.5
            if touches_shoulder(lw) and touches_shoulder(rw) and not l_up and not r_up:
                if le.y > shoulder_y and re.y > shoulder_y:
                    return "Full Timeout (NBA)"
        return None

    def _check_blocking_foul(self, lw, rw, le, re, l_up, r_up, waist_y, shoulder_width, ruleset):
        wrists_near_waist = abs(lw.y - waist_y) < shoulder_width * 0.6 and abs(rw.y - waist_y) < shoulder_width * 0.6
        elbows_wide = self._dist(le, re) > shoulder_width * 1.2
        if wrists_near_waist and elbows_wide and not l_up and not r_up:
            if ruleset == "nba":
                return "Blocking Foul (NBA)"
            return "Blocking / Illegal Screen (FIBA)"
        return None

    def _check_substitution(self, lw, le, re, wrists_dist, shoulder_width, shoulder_y, waist_y, ruleset):
        wrists_crossed = wrists_dist < (shoulder_width * 0.5)
        hands_at_chest = (lw.y > shoulder_y - shoulder_width * 0.4) and (lw.y < waist_y - shoulder_width * 0.2)
        elbows_not_too_far = self._dist(le, re) < shoulder_width * 1.2
        if wrists_crossed and hands_at_chest and elbows_not_too_far:
            if ruleset == "nba":
                return "Substitution / Illegal Screen (NBA)"
            return "Substitution (FIBA)"
        return None

    def _check_t_shape_timeout(self, lw, rw, le, re, wrists_dist, shoulder_width, shoulder_y, ruleset):
        wrists_crossed = wrists_dist < (shoulder_width * 0.5)
        hands_touching = wrists_dist < (shoulder_width * 0.7)
        elbows_far = self._dist(le, re) > (shoulder_width * 0.9)
        hands_high = lw.y < shoulder_y + shoulder_width * 0.5 and rw.y < shoulder_y + shoulder_width * 0.5
        if hands_touching and elbows_far and hands_high and not wrists_crossed:
            if ruleset == "nba":
                return "20-Sec Timeout / Tech (NBA)"
            return "Timeout / Tech Foul (FIBA)"
        return None

    def _check_stop_clock(self, l_up, r_up):
        if l_up or r_up:
            return "Stop Clock (Foul / Violation)"
        return None

    def _classify_referee_gesture(self, marks, ruleset: str) -> str:
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
        if ls.visibility < 0.5 or rs.visibility < 0.5:
            return None

        shoulder_width = self._dist(ls, rs)
        if shoulder_width < 0.01:
            return None

        shoulder_y = (ls.y + rs.y) / 2

        # Estimate waist if hips are not visible
        if lh.visibility < 0.5 or rh.visibility < 0.5:
            waist_y = shoulder_y + shoulder_width * 1.5
        else:
            waist_y = (lh.y + rh.y) / 2

        l_up = self._is_arm_up(lw, le, ls, shoulder_width)
        r_up = self._is_arm_up(rw, re, rs, shoulder_width)
        wrists_dist = self._dist(lw, rw)

        # 1. Both Arms Up (Jump Ball)
        gesture = self._check_jump_ball(l_up, r_up)
        if gesture: return gesture

        # 2. Touch Head (24-Second Violation)
        gesture = self._check_24_second_violation(nose, lw, rw, ls, l_up, r_up, shoulder_width)
        if gesture: return gesture

        # 3. Hands on Shoulders (NBA Full Timeout)
        gesture = self._check_nba_full_timeout(lw, rw, ls, rs, le, re, l_up, r_up, shoulder_width, shoulder_y, ruleset)
        if gesture: return gesture

        # 4. Hands on Hips (Blocking Foul)
        gesture = self._check_blocking_foul(lw, rw, le, re, l_up, r_up, waist_y, shoulder_width, ruleset)
        if gesture: return gesture

        # 5. Substitution / Crossed Arms
        gesture = self._check_substitution(lw, le, re, wrists_dist, shoulder_width, shoulder_y, waist_y, ruleset)
        if gesture: return gesture

        # 6. T-Shape (Timeout / Technical Foul)
        gesture = self._check_t_shape_timeout(lw, rw, le, re, wrists_dist, shoulder_width, shoulder_y, ruleset)
        if gesture: return gesture

        # 7. One Arm Up (Stop Clock)
        gesture = self._check_stop_clock(l_up, r_up)
        if gesture: return gesture

        return None
