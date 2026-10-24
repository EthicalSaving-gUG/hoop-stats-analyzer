"""
referee_detector.py
===================
Computer vision classifier for detecting basketball referees and distinguishing
them from basketball players based on physical and visual characteristics:

1. Upper Body (Shirt / Jersey):
   - Grey Shirt: DBB / FIBA standard grey referee jersey (low saturation, neutral tone, mid brightness).
   - Striped Shirt: NBA / NCAA traditional black & white vertical stripes.
   - Contrasts with player jerseys that feature vibrant team colors or large chest numbers.

2. Lower Body (Pants vs Shorts):
   - Long Pants ("Lange Hose"): Referees wear long black/dark trousers extending to ankles/shoes.
   - Basketball players wear shorts, exposing bare skin at knees/calves or wearing contrasting gear.
   - Color continuity and absence of bare leg skin confirm long referee trousers.

3. Absence of Player Jersey Numbers:
   - Referees do not carry large player roster numbers.

4. Gesture Isolation:
   - Ensures referee hand signal recognition is strictly restricted to verified referees.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class RefereeClassification:
    is_referee: bool
    confidence: float
    shirt_score: float
    shirt_color: str              # "Grey", "Stripes", "TeamColor", "Unknown"
    pants_score: float
    pants_type: str               # "long_pants", "shorts", "unknown"
    has_skin_on_legs: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    bbox: Optional[Tuple[float, float, float, float]] = None


class RefereeDetector:
    """
    Evaluates visual characteristics of persons on court to identify referees.
    """

    def __init__(self,
                 grey_sat_max: int = 55,
                 grey_val_min: int = 50,
                 grey_val_max: int = 215,
                 dark_pants_val_max: int = 75,
                 min_referee_conf: float = 0.52):
        self.grey_sat_max = grey_sat_max
        self.grey_val_min = grey_val_min
        self.grey_val_max = grey_val_max
        self.dark_pants_val_max = dark_pants_val_max
        self.min_referee_conf = min_referee_conf

    # ─────────────────────────────────────────────────────────────────────────
    # 1. SHIRT / TORSO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    def analyze_shirt(self, torso_crop: np.ndarray) -> Tuple[float, str, Dict[str, Any]]:
        """
        Analyzes the torso area to check for DBB/FIBA grey shirt or stripes.
        Returns: (shirt_score, shirt_color_label, metrics_dict)
        """
        if torso_crop is None or torso_crop.size == 0 or torso_crop.shape[0] < 5 or torso_crop.shape[1] < 5:
            return 0.0, "Unknown", {}

        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        b, g, r = cv2.split(torso_crop)

        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]

        # Criteria for Grey (low saturation, moderate brightness, balanced RGB)
        # 1. HSV Saturation <= grey_sat_max
        sat_mask = s <= self.grey_sat_max
        # 2. HSV Value in [grey_val_min, grey_val_max]
        val_mask = (v >= self.grey_val_min) & (v <= self.grey_val_max)
        # 3. Channel neutrality (R, G, B are very close)
        channel_diff = np.maximum.reduce([np.abs(r.astype(int) - g.astype(int)),
                                          np.abs(g.astype(int) - b.astype(int)),
                                          np.abs(r.astype(int) - b.astype(int))])
        neutral_mask = channel_diff <= 26

        grey_mask = sat_mask & val_mask & neutral_mask
        grey_ratio = float(np.count_nonzero(grey_mask)) / total_pixels

        mean_sat = float(np.mean(s))
        mean_val = float(np.mean(v))
        channel_std = float(np.std([np.mean(b), np.mean(g), np.mean(r)]))

        # Check for traditional vertical referee stripes (alternating light & dark)
        gray = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        vertical_edge_energy = float(np.mean(np.abs(sobel_x)))

        is_striped = False
        stripes_score = 0.0
        if vertical_edge_energy > 28.0 and channel_std < 15.0 and mean_sat < 70:
            # Significant vertical edges with neutral color = stripes!
            stripes_score = min(1.0, vertical_edge_energy / 50.0)
            is_striped = True

        # Base grey score calculation
        # If mean saturation is high, it is a colored team jersey (Red, Blue, Green, etc.)
        sat_penalty = min(1.0, mean_sat / 90.0)
        pure_grey_score = max(0.0, grey_ratio * 1.2 - sat_penalty * 0.4)
        pure_grey_score = min(1.0, pure_grey_score)

        if is_striped and stripes_score > pure_grey_score:
            shirt_score = stripes_score
            shirt_color = "Stripes"
        elif pure_grey_score >= 0.40:
            shirt_score = pure_grey_score
            shirt_color = "Grey"
        elif mean_sat > 70:
            shirt_score = 0.05
            shirt_color = "TeamColor"
        else:
            shirt_score = pure_grey_score
            shirt_color = "Neutral"

        metrics = {
            "grey_ratio": round(grey_ratio, 3),
            "mean_sat": round(mean_sat, 1),
            "mean_val": round(mean_val, 1),
            "channel_std": round(channel_std, 1),
            "vertical_edges": round(vertical_edge_energy, 1)
        }

        return round(shirt_score, 3), shirt_color, metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 2. PANTS / LEGS ANALYSIS (LONG PANTS VS SHORTS)
    # ─────────────────────────────────────────────────────────────────────────
    def analyze_pants(self, leg_crop: np.ndarray) -> Tuple[float, str, bool, Dict[str, Any]]:
        """
        Analyzes the leg area to distinguish referees' long black pants ("lange Hose")
        from basketball players' shorts with exposed skin.
        Returns: (pants_score, pants_type, has_skin_on_legs, metrics_dict)
        """
        if leg_crop is None or leg_crop.size == 0 or leg_crop.shape[0] < 6 or leg_crop.shape[1] < 6:
            return 0.0, "unknown", False, {}

        lh, lw = leg_crop.shape[:2]
        total_pixels = lh * lw

        hsv = cv2.cvtColor(leg_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 1. Dark Fabric Detection (Black / Dark Trousers)
        dark_mask = (v <= self.dark_pants_val_max) & (s <= 110)
        dark_ratio = float(np.count_nonzero(dark_mask)) / total_pixels

        # 2. Skin Tone Detection (detects bare knees and calves typical for basketball shorts)
        # Using combined HSV and YCrCb for high precision
        h_skin_mask = ((h <= 25) | (h >= 165)) & (s >= 35) & (s <= 190) & (v >= 65) & (v <= 250)
        
        ycrcb = cv2.cvtColor(leg_crop, cv2.COLOR_BGR2YCrCb)
        y_c, cr, cb = cv2.split(ycrcb)
        ycrcb_skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

        skin_mask = h_skin_mask & ycrcb_skin_mask
        skin_ratio = float(np.count_nonzero(skin_mask)) / total_pixels
        has_skin_on_legs = skin_ratio >= 0.07

        # 3. Leg Height Continuity Analysis (Upper Leg vs Lower Leg)
        # Referees wearing long pants have continuous dark fabric from upper thigh to ankle.
        # Players wearing shorts have a discontinuity: shorts color -> skin tone / socks.
        mid_y = int(lh * 0.45)
        upper_leg = leg_crop[:mid_y, :]
        lower_leg = leg_crop[mid_y:, :]

        upper_v = float(np.mean(cv2.cvtColor(upper_leg, cv2.COLOR_BGR2HSV)[:, :, 2])) if upper_leg.size > 0 else 100.0
        lower_v = float(np.mean(cv2.cvtColor(lower_leg, cv2.COLOR_BGR2HSV)[:, :, 2])) if lower_leg.size > 0 else 100.0
        v_diff = abs(upper_v - lower_v)

        lower_dark_ratio = float(np.count_nonzero(cv2.cvtColor(lower_leg, cv2.COLOR_BGR2HSV)[:, :, 2] <= self.dark_pants_val_max)) / (lower_leg.shape[0] * lower_leg.shape[1]) if lower_leg.size > 0 else 0.0

        # Scoring
        if has_skin_on_legs:
            # Significant bare skin on legs → clearly basketball shorts!
            pants_score = max(0.0, 0.25 - skin_ratio * 1.5)
            pants_type = "shorts"
        else:
            # No skin on legs. Check if it's continuous dark fabric
            if dark_ratio >= 0.45 and lower_dark_ratio >= 0.35 and v_diff < 35:
                # Strong long black pants signal
                pants_score = min(1.0, 0.50 + dark_ratio * 0.50)
                pants_type = "long_pants"
            elif dark_ratio >= 0.30:
                pants_score = min(1.0, 0.40 + dark_ratio * 0.40)
                pants_type = "long_pants"
            else:
                pants_score = 0.20
                pants_type = "shorts"

        metrics = {
            "dark_ratio": round(dark_ratio, 3),
            "lower_dark_ratio": round(lower_dark_ratio, 3),
            "skin_ratio": round(skin_ratio, 3),
            "v_diff": round(v_diff, 1)
        }

        return round(pants_score, 3), pants_type, has_skin_on_legs, metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 3. COMPOSITE EVALUATION FROM BOUNDING BOX OR LANDMARKS
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_person(self,
                        frame: np.ndarray,
                        bbox: Optional[Tuple[float, float, float, float] | List[float]] = None,
                        landmarks: Optional[Any] = None) -> RefereeClassification:
        """
        Evaluates whether a person detected in the frame is a referee.
        Supports either bounding box [x1, y1, x2, y2] or MediaPipe Pose landmarks.
        """
        if frame is None or frame.size == 0:
            return RefereeClassification(
                is_referee=False, confidence=0.0, shirt_score=0.0, shirt_color="Unknown",
                pants_score=0.0, pants_type="unknown", has_skin_on_legs=False,
                reasons=["Frame is empty"]
            )

        fh, fw = frame.shape[:2]

        # 1. Derive bounding box from landmarks if bbox not provided
        if bbox is None and landmarks is not None:
            xs = [lm.x * fw for lm in landmarks if hasattr(lm, 'x') and getattr(lm, 'visibility', 1.0) > 0.2]
            ys = [lm.y * fh for lm in landmarks if hasattr(lm, 'y') and getattr(lm, 'visibility', 1.0) > 0.2]
            if not xs or not ys:
                return RefereeClassification(
                    is_referee=False, confidence=0.0, shirt_score=0.0, shirt_color="Unknown",
                    pants_score=0.0, pants_type="unknown", has_skin_on_legs=False,
                    reasons=["Pose landmarks not visible"]
                )
            pad_x = (max(xs) - min(xs)) * 0.15
            pad_y = (max(ys) - min(ys)) * 0.10
            x1 = max(0, int(min(xs) - pad_x))
            y1 = max(0, int(min(ys) - pad_y))
            x2 = min(fw, int(max(xs) + pad_x))
            y2 = min(fh, int(max(ys) + pad_y))
            bbox = (x1, y1, x2, y2)
        elif bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(fw - 1, x1))
            y1 = max(0, min(fh - 1, y1))
            x2 = max(x1 + 1, min(fw, x2))
            y2 = max(y1 + 1, min(fh, y2))
            bbox = (x1, y1, x2, y2)
        else:
            return RefereeClassification(
                is_referee=False, confidence=0.0, shirt_score=0.0, shirt_color="Unknown",
                pants_score=0.0, pants_type="unknown", has_skin_on_legs=False,
                reasons=["Neither bbox nor landmarks provided"]
            )

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < 10 or box_h < 20:
            return RefereeClassification(
                is_referee=False, confidence=0.0, shirt_score=0.0, shirt_color="Unknown",
                pants_score=0.0, pants_type="unknown", has_skin_on_legs=False,
                reasons=["Person bounding box too small"]
            )

        # 2. Extract Torso & Leg Crops
        # Torso: upper 22% - 54% of bbox height, middle 20% - 80% width
        torso_y1 = y1 + int(box_h * 0.22)
        torso_y2 = y1 + int(box_h * 0.54)
        torso_x1 = x1 + int(box_w * 0.20)
        torso_x2 = x2 - int(box_w * 0.20)

        # Legs: 55% - 92% of bbox height, middle 20% - 80% width
        legs_y1 = y1 + int(box_h * 0.55)
        legs_y2 = y1 + int(box_h * 0.92)
        legs_x1 = x1 + int(box_w * 0.20)
        legs_x2 = x2 - int(box_w * 0.20)

        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        legs_crop = frame[legs_y1:legs_y2, legs_x1:legs_x2]

        # 3. Analyze Shirt & Pants
        shirt_score, shirt_color, shirt_metrics = self.analyze_shirt(torso_crop)
        pants_score, pants_type, has_skin_on_legs, pants_metrics = self.analyze_pants(legs_crop)

        # 4. Composite Confidence & Decision
        reasons = []
        if shirt_score >= 0.45:
            reasons.append(f"Schiedsrichter-Oberteil erkannt ({shirt_color}, Score: {shirt_score})")
        else:
            reasons.append(f"Kein Schiedsrichter-Oberteil ({shirt_color}, Score: {shirt_score})")

        if pants_score >= 0.45:
            reasons.append(f"Lange Schiedsrichterhose erkannt ({pants_type}, Score: {pants_score})")
        else:
            reasons.append(f"Keine lange Schiedsrichterhose ({pants_type}, Score: {pants_score})")

        if has_skin_on_legs:
            reasons.append("Freie Beinhaut detektiert (typisch für Spieler-Shorts)")

        confidence = 0.50 * shirt_score + 0.50 * pants_score

        # Both shirt and pants must show referee indicators
        is_referee = (
            shirt_score >= 0.42 and
            pants_score >= 0.42 and
            confidence >= self.min_referee_conf and
            not has_skin_on_legs
        )

        all_metrics = {
            "shirt": shirt_metrics,
            "pants": pants_metrics,
            "bbox": [x1, y1, x2, y2]
        }

        return RefereeClassification(
            is_referee=is_referee,
            confidence=round(confidence, 3),
            shirt_score=shirt_score,
            shirt_color=shirt_color,
            pants_score=pants_score,
            pants_type=pants_type,
            has_skin_on_legs=has_skin_on_legs,
            reasons=reasons,
            metrics=all_metrics,
            bbox=bbox
        )

    def evaluate_from_landmarks(self, frame: np.ndarray, landmarks: Any) -> RefereeClassification:
        """Helper to evaluate directly from MediaPipe landmarks."""
        return self.evaluate_person(frame, bbox=None, landmarks=landmarks)
