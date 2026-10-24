"""
test_referee_detector.py
========================
Unit tests for basketball referee visual detection and gesture recognition isolation:
1. Grey shirt detection (DBB / FIBA standard) vs team colored jerseys.
2. Long black pants ("lange Hose") vs basketball shorts with exposed skin.
3. Referee appearance evaluation on full frames and bounding boxes.
4. Gesture recognition strict isolation: gestures performed by players are rejected,
   while gestures performed by verified referees are recognized.
"""

import numpy as np
import pytest
from types import SimpleNamespace

from hoop_stats.referee_detector import RefereeDetector, RefereeClassification
from hoop_stats.agents.referee_signals_agent import RefereeSignalsAgent
from hoop_stats.game_state import GameState


def create_person_frame(shirt_color_bgr, leg_type="long_black_pants", x1=200, y1=100, width=140, height=280, landmarks=None):
    """
    Creates a synthetic 720x1280 frame containing a person with specific clothing:
    - shirt_color_bgr: (B, G, R) for torso, or 'stripes'
    - leg_type:
        - 'long_black_pants': continuous dark fabric from thigh to ankle (referee)
        - 'shorts_with_skin': shorts on top, bare skin tone on calves/knees (player)
        - 'shorts_with_socks': shorts on top, white socks at bottom, bare skin in middle
    - landmarks: if provided, computes exact matching bounding box in the frame
    """
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)  # Court background
    fw, fh = 1280, 720

    if landmarks is not None:
        xs = [lm.x * fw for lm in landmarks if getattr(lm, 'visibility', 1.0) > 0.2]
        ys = [lm.y * fh for lm in landmarks if getattr(lm, 'visibility', 1.0) > 0.2]
        pad_x = (max(xs) - min(xs)) * 0.15
        pad_y = (max(ys) - min(ys)) * 0.10
        x1 = max(0, int(min(xs) - pad_x))
        y1 = max(0, int(min(ys) - pad_y))
        x2 = min(fw, int(max(xs) + pad_x))
        y2 = min(fh, int(max(ys) + pad_y))
        width = x2 - x1
        height = y2 - y1
    else:
        x2 = x1 + width
        y2 = y1 + height

    # 1. Head (y: 0% - 18%)
    head_y2 = y1 + int(height * 0.18)
    frame[y1:head_y2, x1 + int(width*0.3): x2 - int(width*0.3)] = (130, 160, 210)  # skin tone

    # 2. Torso / Shirt (y: 20% - 54%)
    torso_y1 = y1 + int(height * 0.20)
    torso_y2 = y1 + int(height * 0.54)
    if shirt_color_bgr == "stripes":
        stripe_pattern = np.zeros((torso_y2 - torso_y1, width, 3), dtype=np.uint8)
        for col in range(width):
            stripe_pattern[:, col] = (240, 240, 240) if (col // 8) % 2 == 0 else (20, 20, 20)
        frame[torso_y1:torso_y2, x1:x2] = stripe_pattern
    else:
        frame[torso_y1:torso_y2, x1:x2] = shirt_color_bgr

    # 3. Legs / Pants (y: 55% - 94%)
    legs_y1 = y1 + int(height * 0.55)
    legs_y2 = y1 + int(height * 0.94)

    if leg_type == "long_black_pants":
        # Continuous dark black trousers down to shoes
        frame[legs_y1:legs_y2, x1 + int(width*0.15): x2 - int(width*0.15)] = (25, 25, 25)
    elif leg_type == "shorts_with_skin":
        # Shorts (55% - 70%) in blue, exposed skin on knees/shins (70% - 94%)
        shorts_y2 = y1 + int(height * 0.70)
        frame[legs_y1:shorts_y2, x1 + int(width*0.15): x2 - int(width*0.15)] = (180, 50, 30)  # blue shorts
        # Bare leg skin
        frame[shorts_y2:legs_y2, x1 + int(width*0.18): x2 - int(width*0.18)] = (130, 160, 210)  # skin tone
    elif leg_type == "shorts_with_socks":
        s1 = y1 + int(height * 0.68)
        s2 = y1 + int(height * 0.82)
        frame[legs_y1:s1, x1 + int(width*0.15): x2 - int(width*0.15)] = (20, 20, 200)  # red shorts
        frame[s1:s2, x1 + int(width*0.18): x2 - int(width*0.18)] = (130, 160, 210)  # bare skin
        frame[s2:legs_y2, x1 + int(width*0.18): x2 - int(width*0.18)] = (230, 230, 230)  # white socks

    bbox = [x1, y1, x2, y2]
    return frame, bbox


def create_pose_landmarks(gesture_type="stop_clock"):
    """
    Creates synthetic MediaPipe 3D pose landmarks for 33 standard keypoints.
    Normalized coordinates: (x, y, z) in [0.0, 1.0].
    """
    marks = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)]

    # Center person at x=0.5, y=0.3 to 0.85
    # Nose (0)
    marks[0] = SimpleNamespace(x=0.50, y=0.20, z=0.0, visibility=0.95)

    # Shoulders: left (11), right (12)
    marks[11] = SimpleNamespace(x=0.45, y=0.30, z=0.0, visibility=0.95)
    marks[12] = SimpleNamespace(x=0.55, y=0.30, z=0.0, visibility=0.95)

    # Hips: left (23), right (24)
    marks[23] = SimpleNamespace(x=0.47, y=0.55, z=0.0, visibility=0.95)
    marks[24] = SimpleNamespace(x=0.53, y=0.55, z=0.0, visibility=0.95)

    # Knees: left (25), right (26)
    marks[25] = SimpleNamespace(x=0.47, y=0.72, z=0.0, visibility=0.95)
    marks[26] = SimpleNamespace(x=0.53, y=0.72, z=0.0, visibility=0.95)

    # Ankles: left (27), right (28)
    marks[27] = SimpleNamespace(x=0.47, y=0.88, z=0.0, visibility=0.95)
    marks[28] = SimpleNamespace(x=0.53, y=0.88, z=0.0, visibility=0.95)

    # Elbows & Wrists according to gesture:
    if gesture_type == "stop_clock":
        # Right arm straight up: wrist above elbow above shoulder
        marks[14] = SimpleNamespace(x=0.55, y=0.20, z=0.0, visibility=0.95)  # Right elbow
        marks[16] = SimpleNamespace(x=0.55, y=0.08, z=0.0, visibility=0.95)  # Right wrist
        # Left arm relaxed down
        marks[13] = SimpleNamespace(x=0.43, y=0.42, z=0.0, visibility=0.95)  # Left elbow
        marks[15] = SimpleNamespace(x=0.43, y=0.52, z=0.0, visibility=0.95)  # Left wrist

    elif gesture_type == "substitution":
        # Crossed forearms at chest (X shape)
        marks[13] = SimpleNamespace(x=0.46, y=0.38, z=0.0, visibility=0.95)  # Left elbow
        marks[14] = SimpleNamespace(x=0.54, y=0.38, z=0.0, visibility=0.95)  # Right elbow
        marks[15] = SimpleNamespace(x=0.51, y=0.35, z=0.0, visibility=0.95)  # Left wrist
        marks[16] = SimpleNamespace(x=0.49, y=0.35, z=0.0, visibility=0.95)  # Right wrist

    elif gesture_type == "timeout":
        # T-shape: hands touching/close, one horizontal, one vertical
        marks[13] = SimpleNamespace(x=0.42, y=0.36, z=0.0, visibility=0.95)
        marks[14] = SimpleNamespace(x=0.58, y=0.36, z=0.0, visibility=0.95)
        marks[15] = SimpleNamespace(x=0.49, y=0.32, z=0.0, visibility=0.95)
        marks[16] = SimpleNamespace(x=0.51, y=0.32, z=0.0, visibility=0.95)

    elif gesture_type == "idle":
        # Relaxed arms down by hips
        marks[13] = SimpleNamespace(x=0.43, y=0.45, z=0.0, visibility=0.95)
        marks[14] = SimpleNamespace(x=0.57, y=0.45, z=0.0, visibility=0.95)
        marks[15] = SimpleNamespace(x=0.43, y=0.55, z=0.0, visibility=0.95)
        marks[16] = SimpleNamespace(x=0.57, y=0.55, z=0.0, visibility=0.95)

    return marks


class TestRefereeDetector:
    def test_referee_grey_shirt_and_long_pants_detected(self):
        """DBB / FIBA referee wearing grey jersey and long black pants is classified as referee."""
        detector = RefereeDetector()
        grey_shirt = (135, 135, 135)  # Neutral grey in BGR
        frame, bbox = create_person_frame(grey_shirt, leg_type="long_black_pants")

        result = detector.evaluate_person(frame, bbox=bbox)

        assert result.is_referee is True
        assert result.shirt_color == "Grey"
        assert result.pants_type == "long_pants"
        assert result.has_skin_on_legs is False
        assert result.confidence >= 0.60
        assert "Schiedsrichter-Oberteil" in " ".join(result.reasons)
        assert "Lange Schiedsrichterhose" in " ".join(result.reasons)

    def test_referee_striped_shirt_detected(self):
        """Traditional striped referee shirt + long black pants is classified as referee."""
        detector = RefereeDetector()
        frame, bbox = create_person_frame("stripes", leg_type="long_black_pants")

        result = detector.evaluate_person(frame, bbox=bbox)

        assert result.is_referee is True
        assert result.shirt_color == "Stripes"
        assert result.pants_type == "long_pants"
        assert result.confidence >= 0.60

    def test_player_red_jersey_and_shorts_rejected(self):
        """Player with red jersey and shorts with exposed skin is NOT a referee."""
        detector = RefereeDetector()
        red_jersey = (20, 20, 220)  # Bright red
        frame, bbox = create_person_frame(red_jersey, leg_type="shorts_with_skin")

        result = detector.evaluate_person(frame, bbox=bbox)

        assert result.is_referee is False
        assert result.has_skin_on_legs is True
        assert result.pants_type == "shorts"
        assert result.confidence < 0.40

    def test_player_blue_jersey_and_shorts_rejected(self):
        """Player with blue jersey and shorts is NOT a referee."""
        detector = RefereeDetector()
        blue_jersey = (220, 100, 20)  # Blue
        frame, bbox = create_person_frame(blue_jersey, leg_type="shorts_with_socks")

        result = detector.evaluate_person(frame, bbox=bbox)

        assert result.is_referee is False
        assert result.confidence < 0.45

    def test_person_grey_shirt_with_shorts_rejected(self):
        """A person wearing a grey shirt but basketball shorts (exposed skin) is rejected."""
        detector = RefereeDetector()
        grey_shirt = (135, 135, 135)
        frame, bbox = create_person_frame(grey_shirt, leg_type="shorts_with_skin")

        result = detector.evaluate_person(frame, bbox=bbox)

        # Rejected because referees wear long pants, NOT shorts!
        assert result.is_referee is False
        assert result.has_skin_on_legs is True
        assert result.pants_type == "shorts"

    def test_person_team_jersey_with_long_black_tights_rejected(self):
        """A player wearing long black compression tights but a colored team jersey is rejected."""
        detector = RefereeDetector()
        bright_yellow_jersey = (20, 230, 230)
        frame, bbox = create_person_frame(bright_yellow_jersey, leg_type="long_black_pants")

        result = detector.evaluate_person(frame, bbox=bbox)

        # Rejected because shirt is not a referee shirt
        assert result.is_referee is False
        assert result.shirt_color == "TeamColor"


class TestRefereeGestureIsolation:
    def test_player_arm_up_gesture_is_ignored(self):
        """
        When a basketball player (e.g. shooting or calling for ball with arm up) makes a gesture,
        RefereeSignalsAgent MUST ignore it because the person is not a referee.
        """
        gs = GameState()
        agent = RefereeSignalsAgent(gs, None, enforce_referee_check=True)

        arm_up_marks = create_pose_landmarks("stop_clock")
        # Create player frame aligned with landmarks (Red jersey + shorts with bare skin)
        player_frame, _ = create_person_frame((20, 20, 220), leg_type="shorts_with_skin", landmarks=arm_up_marks)

        # Evaluate player pose: Must return None because person is a player!
        detected_gesture = agent.process_person_pose(player_frame, arm_up_marks)
        assert detected_gesture is None

        # Verify no event is logged in GameState
        assert len(gs.events) == 0

    def test_referee_arm_up_gesture_is_recognized(self):
        """
        When a verified referee (grey shirt, long pants) raises an arm,
        the gesture IS recognized as 'Stop Clock'.
        """
        gs = GameState()
        agent = RefereeSignalsAgent(gs, None, enforce_referee_check=True)

        arm_up_marks = create_pose_landmarks("stop_clock")
        # Create referee frame aligned with landmarks (Grey shirt + long black pants)
        ref_frame, _ = create_person_frame((135, 135, 135), leg_type="long_black_pants", landmarks=arm_up_marks)

        detected_gesture = agent.process_person_pose(ref_frame, arm_up_marks)
        assert detected_gesture == "Stop Clock (Foul / Violation)"

    def test_referee_substitution_gesture_is_recognized(self):
        """
        Verified referee crossing forearms triggers Substitution signal.
        """
        gs = GameState()
        agent = RefereeSignalsAgent(gs, None, enforce_referee_check=True)

        sub_marks = create_pose_landmarks("substitution")
        ref_frame, _ = create_person_frame((135, 135, 135), leg_type="long_black_pants", landmarks=sub_marks)

        detected_gesture = agent.process_person_pose(ref_frame, sub_marks)
        assert detected_gesture == "Substitution (FIBA)"

    def test_player_substitution_pose_is_rejected(self):
        """
        Player crossing arms is NOT recognized as referee substitution signal.
        """
        gs = GameState()
        agent = RefereeSignalsAgent(gs, None, enforce_referee_check=True)

        sub_marks = create_pose_landmarks("substitution")
        player_frame, _ = create_person_frame((200, 30, 20), leg_type="shorts_with_skin", landmarks=sub_marks)

        detected_gesture = agent.process_person_pose(player_frame, sub_marks)
        assert detected_gesture is None

    def test_referee_signal_event_contains_referee_verification_metadata(self):
        """
        Triggered referee event must include metadata confirming referee verification.
        """
        gs = GameState()
        agent = RefereeSignalsAgent(gs, None, enforce_referee_check=True)

        ref_frame, bbox = create_person_frame((135, 135, 135), leg_type="long_black_pants")
        sub_marks = create_pose_landmarks("substitution")

        ref_eval = agent.referee_detector.evaluate_person(ref_frame, bbox=bbox)
        assert ref_eval.is_referee is True

        agent._trigger_event("Substitution (FIBA)", frame_idx=100, ref_info=ref_eval)

        assert len(gs.events) == 1
        evt = gs.events[0]
        assert evt.agent == "referee_signals"
        assert evt.event_type == "REFEREE_SIGNAL"
        assert evt.data["signal"] == "Substitution (FIBA)"
        assert evt.data["referee_verified"] is True
        assert evt.data["shirt_color"] == "Grey"
        assert evt.data["pants_type"] == "long_pants"
        assert evt.data["referee_confidence"] >= 0.60
