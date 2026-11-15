"""
benchmark_evaluator.py
======================
Ground Truth Benchmarking and Automated Data Comparison Engine.
Compares Computer Vision detected match statistics against official DBB (Deutscher Basketball Bund)
scoresheet and video tracking ground truth data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


# Official Ground Truth data for SB Uffenheim (Spieltag 2: SG Oerlenbach/Ebenhausen 2 vs. SB Uffenheim 49 : 55)
OFFICIAL_GROUND_TRUTH_UFFENHEIM = {
    "team": "SB Uffenheim",
    "opponent": "SG Oerlenbach/Ebenhausen 2",
    "final_score": {"uffenheim": 55, "oerlenbach": 49},
    "team_totals": {
        "points": 55,
        "fgm": 17,
        "fga": 56,
        "fg_pct": 0.304,
        "fg2m": 10,
        "fg2a": 37,
        "fg2_pct": 0.270,
        "fg3m": 7,
        "fg3a": 19,
        "fg3_pct": 0.368,
        "ftm": 14,
        "fta": 26,
        "ft_pct": 0.538,
        "efg_pct": 0.366,
        "ts_pct": 0.408,
        "oreb": 7,
        "dreb": 38,
        "reb": 45,
        "assists": 7,
        "turnovers": 20,
        "turnovers_traveling": 6,
        "turnovers_bad_pass": 11,
        "steals": 6,
        "blocks": 3,
        "fouls_drawn": 18,
    },
    "players": {
        "11": {
            "name": "Pierre Reichl",
            "minutes": "32:22",
            "points": 6,
            "fgm": 2, "fga": 7, "fg_pct": 0.29,
            "fg2m": 1, "fg2a": 4, "fg2_pct": 0.25,
            "fg3m": 1, "fg3a": 3, "fg3_pct": 0.33,
            "ftm": 1, "fta": 2, "ft_pct": 0.50,
            "efg_pct": 0.36, "ts_pct": 0.38,
            "oreb": 2, "dreb": 6, "reb": 8,
            "assists": 1, "turnovers": 5,
            "steals": 0, "blocks": 0, "fouls_drawn": 2,
            "plus_minus": -3,
        },
        "14": {
            "name": "Ralf Dubiel",
            "minutes": "38:35",
            "points": 16,
            "fgm": 5, "fga": 13, "fg_pct": 0.38,
            "fg2m": 3, "fg2a": 10, "fg2_pct": 0.30,
            "fg3m": 2, "fg3a": 3, "fg3_pct": 0.67,
            "ftm": 4, "fta": 7, "ft_pct": 0.57,
            "efg_pct": 0.46, "ts_pct": 0.50,
            "oreb": 2, "dreb": 13, "reb": 15,
            "assists": 3, "turnovers": 1,
            "steals": 2, "blocks": 0, "fouls_drawn": 6,
            "plus_minus": 6,
        },
        "20": {
            "name": "Roman Blank",
            "minutes": "40:00",
            "points": 23,
            "fgm": 7, "fga": 20, "fg_pct": 0.35,
            "fg2m": 4, "fg2a": 10, "fg2_pct": 0.40,
            "fg3m": 3, "fg3a": 10, "fg3_pct": 0.30,
            "ftm": 6, "fta": 13, "ft_pct": 0.46,
            "efg_pct": 0.43, "ts_pct": 0.45,
            "oreb": 2, "dreb": 8, "reb": 10,
            "assists": 1, "turnovers": 6,
            "steals": 3, "blocks": 3, "fouls_drawn": 8,
            "plus_minus": 6,
        },
        "26": {
            "name": "Helmut Schwebel",
            "minutes": "25:54",
            "points": 5,
            "fgm": 1, "fga": 6, "fg_pct": 0.17,
            "fg2m": 0, "fg2a": 3, "fg2_pct": 0.00,
            "fg3m": 1, "fg3a": 3, "fg3_pct": 0.33,
            "ftm": 2, "fta": 2, "ft_pct": 1.00,
            "efg_pct": 0.25, "ts_pct": 0.36,
            "oreb": 1, "dreb": 5, "reb": 6,
            "assists": 0, "turnovers": 2,
            "steals": 1, "blocks": 0, "fouls_drawn": 1,
            "plus_minus": 9,
        },
        "35": {
            "name": "Christian Drumm",
            "minutes": "40:00",
            "points": 3,
            "fgm": 1, "fga": 2, "fg_pct": 0.50,
            "fg2m": 1, "fg2a": 2, "fg2_pct": 0.50,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.00,
            "ftm": 1, "fta": 2, "ft_pct": 0.50,
            "efg_pct": 0.50, "ts_pct": 0.52,
            "oreb": 0, "dreb": 4, "reb": 4,
            "assists": 2, "turnovers": 4,
            "steals": 0, "blocks": 0, "fouls_drawn": 1,
            "plus_minus": 6,
        },
        "66": {
            "name": "Dominik Krämer",
            "minutes": "23:06",
            "points": 2,
            "fgm": 1, "fga": 8, "fg_pct": 0.13,
            "fg2m": 1, "fg2a": 8, "fg2_pct": 0.13,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.00,
            "ftm": 0, "fta": 0, "ft_pct": 0.00,
            "efg_pct": 0.13, "ts_pct": 0.13,
            "oreb": 0, "dreb": 2, "reb": 2,
            "assists": 0, "turnovers": 2,
            "steals": 0, "blocks": 0, "fouls_drawn": 0,
            "plus_minus": 6,
        },
    }
}


# Official Ground Truth data for SB Uffenheim (Spieltag 4: TSV Brendlorenzen vs. SB Uffenheim 69 : 67 n.V.)
OFFICIAL_GROUND_TRUTH_BRENDLORENZEN = {
    "team": "SB Uffenheim",
    "opponent": "TSV Brendlorenzen",
    "final_score": {"uffenheim": 67, "brendlorenzen": 69},
    "quarter_scores": {
        "Q1": [17, 21],
        "Q2": [17, 13],
        "Q3": [14, 12],
        "Q4": [12, 14],
        "OT1": [7, 9],
    },
    "team_totals": {
        "points": 67,
        "fgm": 23,
        "fga": 72,
        "fg_pct": 0.319,
        "fg2m": 14,
        "fg2a": 40,
        "fg2_pct": 0.350,
        "fg3m": 9,
        "fg3a": 32,
        "fg3_pct": 0.281,
        "ftm": 12,
        "fta": 19,
        "ft_pct": 0.632,
        "efg_pct": 0.382,
        "ts_pct": 0.417,
        "oreb": 8,
        "dreb": 35,
        "reb": 43,
        "assists": 14,
        "steals": 5,
        "blocks": 5,
        "turnovers": 25,
        "turnovers_traveling": 7,
        "turnovers_bad_pass": 12,
        "fouls_drawn": 16,
        "personal_fouls": 18,
        "plus_minus": -2,
    },
    "players": {
        "7": {
            "name": "Paul Kevin Krämer-Moore",
            "minutes": "33:06",
            "points": 4,
            "fgm": 2, "fga": 9, "fg_pct": 0.22,
            "fg2m": 2, "fg2a": 9, "fg2_pct": 0.22,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.00,
            "ftm": 0, "fta": 0, "ft_pct": 0.00,
            "oreb": 1, "dreb": 1, "reb": 2,
            "assists": 1, "steals": 1, "blocks": 0, "turnovers": 1,
            "fouls_drawn": 0, "personal_fouls": 2, "plus_minus": 5,
        },
        "11": {
            "name": "Pierre Reichl",
            "minutes": "07:17",
            "points": 2,
            "fgm": 1, "fga": 1, "fg_pct": 1.00,
            "fg2m": 1, "fg2a": 1, "fg2_pct": 1.00,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.00,
            "ftm": 0, "fta": 0, "ft_pct": 0.00,
            "oreb": 0, "dreb": 0, "reb": 0,
            "assists": 1, "steals": 0, "blocks": 0, "turnovers": 2,
            "fouls_drawn": 0, "personal_fouls": 3, "plus_minus": -4,
        },
        "14": {
            "name": "Ralf Dubiel",
            "minutes": "14:52",
            "points": 7,
            "fgm": 2, "fga": 3, "fg_pct": 0.67,
            "fg2m": 2, "fg2a": 3, "fg2_pct": 0.67,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.00,
            "ftm": 3, "fta": 4, "ft_pct": 0.75,
            "oreb": 0, "dreb": 1, "reb": 1,
            "assists": 1, "steals": 1, "blocks": 0, "turnovers": 2,
            "fouls_drawn": 1, "personal_fouls": 4, "plus_minus": -2,
        },
        "15": {
            "name": "Rainer Dehner",
            "minutes": "00:00",
            "points": 0,
            "fgm": 0, "fga": 0, "fg_pct": 0.0,
            "fg2m": 0, "fg2a": 0, "fg2_pct": 0.0,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.0,
            "ftm": 0, "fta": 0, "ft_pct": 0.0,
            "oreb": 0, "dreb": 0, "reb": 0,
            "assists": 0, "steals": 0, "blocks": 0, "turnovers": 0,
            "fouls_drawn": 0, "personal_fouls": 0, "plus_minus": 0,
        },
        "18": {
            "name": "Amir Puljic",
            "minutes": "00:00",
            "points": 0,
            "fgm": 0, "fga": 0, "fg_pct": 0.0,
            "fg2m": 0, "fg2a": 0, "fg2_pct": 0.0,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.0,
            "ftm": 0, "fta": 0, "ft_pct": 0.0,
            "oreb": 0, "dreb": 0, "reb": 0,
            "assists": 0, "steals": 0, "blocks": 0, "turnovers": 0,
            "fouls_drawn": 0, "personal_fouls": 0, "plus_minus": 0,
        },
        "20": {
            "name": "Roman Blank",
            "minutes": "44:15",
            "points": 16,
            "fgm": 4, "fga": 11, "fg_pct": 0.36,
            "fg2m": 3, "fg2a": 8, "fg2_pct": 0.38,
            "fg3m": 1, "fg3a": 3, "fg3_pct": 0.33,
            "ftm": 7, "fta": 10, "ft_pct": 0.70,
            "oreb": 3, "dreb": 3, "reb": 6,
            "assists": 3, "steals": 1, "blocks": 2, "turnovers": 4,
            "fouls_drawn": 6, "personal_fouls": 1, "plus_minus": -2,
        },
        "21": {
            "name": "Dominik Schäfer",
            "minutes": "45:00",
            "points": 28,
            "fgm": 10, "fga": 19, "fg_pct": 0.53,
            "fg2m": 4, "fg2a": 7, "fg2_pct": 0.57,
            "fg3m": 6, "fg3a": 12, "fg3_pct": 0.50,
            "ftm": 2, "fta": 5, "ft_pct": 0.40,
            "oreb": 3, "dreb": 17, "reb": 20,
            "assists": 0, "steals": 1, "blocks": 1, "turnovers": 4,
            "fouls_drawn": 5, "personal_fouls": 4, "plus_minus": -2,
        },
        "26": {
            "name": "Helmut Schwebel",
            "minutes": "02:40",
            "points": 0,
            "fgm": 0, "fga": 1, "fg_pct": 0.00,
            "fg2m": 0, "fg2a": 1, "fg2_pct": 0.00,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.00,
            "ftm": 0, "fta": 0, "ft_pct": 0.00,
            "oreb": 0, "dreb": 0, "reb": 0,
            "assists": 0, "steals": 0, "blocks": 0, "turnovers": 0,
            "fouls_drawn": 1, "personal_fouls": 0, "plus_minus": -6,
        },
        "33": {
            "name": "Fabian Temme",
            "minutes": "00:00",
            "points": 0,
            "fgm": 0, "fga": 0, "fg_pct": 0.0,
            "fg2m": 0, "fg2a": 0, "fg2_pct": 0.0,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.0,
            "ftm": 0, "fta": 0, "ft_pct": 0.0,
            "oreb": 0, "dreb": 0, "reb": 0,
            "assists": 0, "steals": 0, "blocks": 0, "turnovers": 0,
            "fouls_drawn": 0, "personal_fouls": 0, "plus_minus": 0,
        },
        "35": {
            "name": "Christian Drumm",
            "minutes": "41:42",
            "points": 0,
            "fgm": 0, "fga": 3, "fg_pct": 0.00,
            "fg2m": 0, "fg2a": 0, "fg2_pct": 0.00,
            "fg3m": 0, "fg3a": 3, "fg3_pct": 0.00,
            "ftm": 0, "fta": 0, "ft_pct": 0.00,
            "oreb": 0, "dreb": 8, "reb": 8,
            "assists": 7, "steals": 0, "blocks": 0, "turnovers": 6,
            "fouls_drawn": 2, "personal_fouls": 2, "plus_minus": -1,
        },
        "66": {
            "name": "Dominik Krämer",
            "minutes": "00:00",
            "points": 0,
            "fgm": 0, "fga": 0, "fg_pct": 0.0,
            "fg2m": 0, "fg2a": 0, "fg2_pct": 0.0,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.0,
            "ftm": 0, "fta": 0, "ft_pct": 0.0,
            "oreb": 0, "dreb": 0, "reb": 0,
            "assists": 0, "steals": 0, "blocks": 0, "turnovers": 0,
            "fouls_drawn": 0, "personal_fouls": 0, "plus_minus": 0,
        },
        "91": {
            "name": "Arthur Apinyan",
            "minutes": "36:08",
            "points": 10,
            "fgm": 4, "fga": 25, "fg_pct": 0.16,
            "fg2m": 2, "fg2a": 11, "fg2_pct": 0.18,
            "fg3m": 2, "fg3a": 14, "fg3_pct": 0.14,
            "ftm": 0, "fta": 0, "ft_pct": 0.00,
            "oreb": 1, "dreb": 5, "reb": 6,
            "assists": 1, "steals": 1, "blocks": 2, "turnovers": 6,
            "fouls_drawn": 1, "personal_fouls": 2, "plus_minus": 2,
        },
    },
}


def get_ground_truth_for_match(match_id: int = 2) -> dict:
    if match_id == 4:
        return OFFICIAL_GROUND_TRUTH_BRENDLORENZEN
    return OFFICIAL_GROUND_TRUTH_UFFENHEIM


class BenchmarkEvaluator:
    """
    Compares detected match metrics against official Ground Truth.
    """

    def __init__(self, ground_truth: dict = None, match_id: int = 2):
        self.gt = ground_truth or get_ground_truth_for_match(match_id)

    def evaluate_team_totals(self, detected_totals: dict) -> dict:
        """
        Computes metric deltas and accuracy percentages between detected and ground truth totals.
        """
        gt_totals = self.gt.get("team_totals", {})
        comparison = {}

        metrics_to_check = [
            ("points", "Points (Punkte gesamt)"),
            ("fgm", "FGM (Field Goals Made)"),
            ("fga", "FGA (Field Goals Attempted)"),
            ("fg2m", "2PM (2-Punkte Treffer)"),
            ("fg2a", "2PA (2-Punkte Versuche)"),
            ("fg3m", "3PM (3-Punkte Treffer)"),
            ("fg3a", "3PA (3-Punkte Versuche)"),
            ("ftm", "FTM (Freiwürfe Treffer)"),
            ("fta", "FTA (Freiwürfe Versuche)"),
            ("reb", "REB (Rebounds gesamt)"),
            ("oreb", "OREB (Offensiv-Rebounds)"),
            ("dreb", "DREB (Defensiv-Rebounds)"),
            ("assists", "AST (Assists)"),
            ("turnovers", "TO (Turnovers / Ballverluste)"),
            ("turnovers_traveling", "  ↳ davon Schrittfehler"),
            ("turnovers_bad_pass", "  ↳ davon Fehlpässe"),
            ("steals", "STL (Steals / Ballgewinne)"),
            ("blocks", "BLK (Blocks)"),
            ("fouls_drawn", "FD (Gezogene Fouls)"),
        ]

        for key, label in metrics_to_check:
            gt_val = gt_totals.get(key, 0)
            det_val = detected_totals.get(key, 0)
            delta = det_val - gt_val
            pct_error = (abs(delta) / gt_val * 100.0) if gt_val > 0 else 0.0
            accuracy = max(0.0, 100.0 - pct_error)
            capture_rate = (det_val / gt_val * 100.0) if gt_val > 0 else 0.0

            comparison[key] = {
                "label": label,
                "ground_truth": gt_val,
                "detected": det_val,
                "delta": round(delta, 3),
                "accuracy_pct": round(accuracy, 1),
                "capture_rate_pct": round(capture_rate, 1),
            }

        return comparison

    def evaluate_player_recognition(self, detected_jerseys: List[str]) -> dict:
        """
        Calculates recall and precision of detected player jersey numbers.
        """
        gt_jerseys = set(self.gt.get("players", {}).keys())
        det_jerseys = set(str(j) for j in detected_jerseys if str(j) not in ("Unknown", "None", ""))

        true_positives = len(det_jerseys.intersection(gt_jerseys))
        recall = (true_positives / len(gt_jerseys) * 100.0) if gt_jerseys else 0.0
        precision = (true_positives / len(det_jerseys) * 100.0) if det_jerseys else 0.0

        return {
            "gt_jerseys": sorted(list(gt_jerseys)),
            "detected_jerseys": sorted(list(det_jerseys)),
            "matched_jerseys": sorted(list(det_jerseys.intersection(gt_jerseys))),
            "recall_pct": round(recall, 1),
            "precision_pct": round(precision, 1),
        }
