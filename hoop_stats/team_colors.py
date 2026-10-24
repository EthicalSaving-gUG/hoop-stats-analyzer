import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter

# Human-readable color names mapped from HSV ranges
COLOR_RANGES = {
    "Red":     [(0, 70, 70), (10, 255, 255)],
    "Red2":    [(165, 70, 70), (180, 255, 255)],  # Red wraps in HSV
    "Orange":  [(11, 100, 100), (25, 255, 255)],
    "Yellow":  [(26, 100, 100), (34, 255, 255)],
    "Green":   [(35, 60, 60), (85, 255, 255)],
    "Blue":    [(86, 60, 60), (130, 255, 255)],
    "Purple":  [(131, 40, 40), (160, 255, 255)],
    "White":   [(0, 0, 180), (180, 40, 255)],
    "Black":   [(0, 0, 0), (180, 255, 50)],
    "Grey":    [(0, 0, 51), (180, 39, 179)],
}

def _classify_color_name(h, s, v):
    """Map a single HSV pixel to a color name."""
    for name, (lo, hi) in COLOR_RANGES.items():
        if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
            return name
    return "Mixed"


def get_dominant_jersey_color(frame, bbox):
    """
    Crops the jersey (torso area) and returns the dominant color name.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_box = y2 - y1
    w_box = x2 - x1

    # Torso region: middle vertical strip, upper 30-65% of bbox height
    torso = frame[y1 + int(h_box * 0.25): y1 + int(h_box * 0.65),
                  x1 + int(w_box * 0.2): x2 - int(w_box * 0.2)]

    if torso.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Sample to speed up
    if len(pixels) > 500:
        idx = np.random.choice(len(pixels), 500, replace=False)
        pixels = pixels[idx]

    try:
        km = KMeans(n_clusters=3, n_init=5, random_state=42)
        km.fit(pixels)
        # Find the largest cluster
        counts = Counter(km.labels_)
        dominant_idx = counts.most_common(1)[0][0]
        center = km.cluster_centers_[dominant_idx]
        h, s, v = int(center[0]), int(center[1]), int(center[2])
        return _classify_color_name(h, s, v)
    except Exception:
        return "Unknown"


def assign_teams_by_color(player_colors, referee_ids=None):
    """
    Given {player_id: color_name}, groups players into at most 2 teams.
    Referees (either listed in referee_ids or with color 'Referee') are
    explicitly isolated and assigned to 'Referee'.
    Returns {player_id: team_name}.
    """
    if not player_colors:
        return {}

    ref_set = set(referee_ids or [])

    # Exclude referees from team color clustering
    candidate_colors = {
        pid: c for pid, c in player_colors.items()
        if pid not in ref_set and c not in ("Referee", "Unknown", "Mixed")
    }

    color_counts = Counter(candidate_colors.values())
    clean = {c: n for c, n in color_counts.items() if n > 0}

    if not clean:
        return {
            pid: ("Referee" if pid in ref_set or player_colors[pid] == "Referee" else "Unknown")
            for pid in player_colors
        }

    sorted_colors = sorted(clean, key=clean.get, reverse=True)
    team_a_color = sorted_colors[0]
    team_b_color = sorted_colors[1] if len(sorted_colors) > 1 else "Other"

    result = {}
    for pid, color in player_colors.items():
        if pid in ref_set or color == "Referee":
            result[pid] = "Referee"
        elif color == team_a_color:
            result[pid] = f"Team {team_a_color}"
        elif color == team_b_color:
            result[pid] = f"Team {team_b_color}"
        else:
            result[pid] = "Other"
    return result

def classify_person_referee_or_player(frame, bbox):
    """
    Convenience function to classify whether a detected person bounding box
    corresponds to a referee (grey shirt, long black pants) or a player.
    Returns: (is_referee: bool, details: dict)
    """
    from .referee_detector import RefereeDetector
    detector = RefereeDetector()
    res = detector.evaluate_person(frame, bbox=bbox)
    return res.is_referee, {
        "confidence": res.confidence,
        "shirt_color": res.shirt_color,
        "pants_type": res.pants_type,
        "has_skin_on_legs": res.has_skin_on_legs,
        "reasons": res.reasons
    }

