"""
manual_stats.py
===============
Backend logic and data models for manual basketball data entry and box score calculations.
Models SB Uffenheim 2026/27 official scoresheet and protocol conventions:
- Kader (Roster)
- Action Codes (2T, 2V, 3T, 3V, FT, FV, OR, DR, AS, ST, BL, TO, FZ, OF)
- Erfassungsarten: Variante A (Summen) and Variante B (Protokoll)
- Live Box Score computation with official FIBA / DBB metrics (eFG%, TS%, AS/TO, Pro 40)
- Consistency checking against official match final scores
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any


ACTION_CODES = {
    "2T": {"label": "Zweier getroffen", "pts": 2, "fg2m": 1, "fg2a": 1, "fgm": 1, "fga": 1},
    "2V": {"label": "Zweier verworfen", "pts": 0, "fg2m": 0, "fg2a": 1, "fgm": 0, "fga": 1},
    "3T": {"label": "Dreier getroffen", "pts": 3, "fg3m": 1, "fg3a": 1, "fgm": 1, "fga": 1},
    "3V": {"label": "Dreier verworfen", "pts": 0, "fg3m": 0, "fg3a": 1, "fgm": 0, "fga": 1},
    "FT": {"label": "Freiwurf getroffen", "pts": 1, "ftm": 1, "fta": 1},
    "FV": {"label": "Freiwurf verworfen", "pts": 0, "ftm": 0, "fta": 1},
    "OR": {"label": "Offensivrebound", "oreb": 1, "reb": 1},
    "DR": {"label": "Defensivrebound", "dreb": 1, "reb": 1},
    "AS": {"label": "Assist", "assists": 1},
    "ST": {"label": "Steal", "steals": 1},
    "BL": {"label": "Block", "blocks": 1},
    "TO": {"label": "Ballverlust (Turnover)", "turnovers": 1},
    "FZ": {"label": "Foul gezogen (im Angriff)", "fouls_drawn_attack": 1, "fouls_drawn_total": 1},
    "OF": {"label": "Offensivfoul gezogen (in der Verteidigung)", "fouls_drawn_defense": 1, "fouls_drawn_total": 1},
}

OFFICIAL_ROSTER_UFFENHEIM_26_27 = [
    {"jersey": "4", "name": "Rainer Dehner"},
    {"jersey": "7", "name": "Paul Kevin Krämer-Moore"},
    {"jersey": "11", "name": "Pierre Reichl"},
    {"jersey": "14", "name": "Ralf Dubiel"},
    {"jersey": "20", "name": "Roman Blank"},
    {"jersey": "21", "name": "Dominik Schäfer"},
    {"jersey": "26", "name": "Helmut Schwebel"},
    {"jersey": "33", "name": "Fabian Temme"},
    {"jersey": "35", "name": "Christian Drumm"},
    {"jersey": "66", "name": "Dominik Krämer"},
    {"jersey": "91", "name": "Arthur Apinyan"},
]

DEFAULT_MATCHES_SCHEDULE = [
    {"spt": 1, "date": "11.10.2026", "home_away": "Heim", "opponent": "SV Kürnach BasketWalls", "sbu_score": 57, "opp_score": 50, "has_app_data": True},
    {"spt": 2, "date": "17.10.2026", "home_away": "Ausw.", "opponent": "SG Oerlenbach/Ebenhausen 2", "sbu_score": 55, "opp_score": 49, "has_app_data": True},
    {"spt": 3, "date": "25.10.2026", "home_away": "Heim", "opponent": "TB Arnstein 2", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 4, "date": "14.11.2026", "home_away": "Ausw.", "opponent": "TSV Brendlorenzen", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 5, "date": "22.11.2026", "home_away": "Heim", "opponent": "TG Kitzingen", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 6, "date": "06.12.2026", "home_away": "Heim", "opponent": "TSV Grombühl 2", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 7, "date": "11.12.2026", "home_away": "Ausw.", "opponent": "SF Waschküch Würzburg", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 8, "date": "10.01.2027", "home_away": "Ausw.", "opponent": "SV Kürnach BasketWalls", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 9, "date": "17.01.2027", "home_away": "Heim", "opponent": "SG Oerlenbach/Ebenhausen 2", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 10, "date": "31.01.2027", "home_away": "Ausw.", "opponent": "TB Arnstein 2", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 11, "date": "14.02.2027", "home_away": "Heim", "opponent": "TSV Brendlorenzen", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 12, "date": "20.02.2027", "home_away": "Ausw.", "opponent": "TG Kitzingen", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 13, "date": "07.03.2027", "home_away": "Ausw.", "opponent": "TSV Grombühl 2", "sbu_score": None, "opp_score": None, "has_app_data": False},
    {"spt": 14, "date": "14.03.2027", "home_away": "Heim", "opponent": "SF Waschküch Würzburg", "sbu_score": None, "opp_score": None, "has_app_data": False},
]


def parse_time_to_seconds(time_str: str) -> float:
    """Parses mm:ss or hh:mm:ss to total seconds."""
    if not time_str or not isinstance(time_str, str):
        return 0.0
    parts = time_str.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1 and parts[0]:
            return float(parts[0])
    except ValueError:
        return 0.0
    return 0.0


def format_seconds_to_time(seconds: float) -> str:
    """Formats total seconds to mm:ss string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


@dataclass
class ProtocolEntry:
    """Single action entry in Variante B (Protokoll)."""
    jersey: str
    code: str
    timestamp_s: Optional[float] = None
    quarter: Optional[int] = None
    note: str = ""

    def validate(self) -> bool:
        return self.code in ACTION_CODES and bool(self.jersey)


class BoxscoreCalculator:
    """Computes comprehensive individual and team basketball statistics."""

    @staticmethod
    def calculate_player_line(jersey: str,
                              name: str,
                              inputs: dict,
                              has_app_data: bool = False) -> dict:
        """
        Calculates a full box score line for a single player.
        Handles both Variante A (Summen) and App-Data overlay rules.
        """
        # Determine played status
        played = bool(inputs.get("played", False))
        if isinstance(inputs.get("played"), str):
            played = inputs.get("played").strip().lower() in ("x", "1", "true", "ja", "yes")

        # Basic Stats from inputs
        fg2m = int(inputs.get("2T", 0) or 0)
        fg2v = int(inputs.get("2V", 0) or 0)
        fg3m = int(inputs.get("3T", 0) or 0)
        fg3v = int(inputs.get("3V", 0) or 0)
        ftm = int(inputs.get("FT", 0) or 0)
        ftv = int(inputs.get("FV", 0) or 0)

        # Attempts & Totals
        fg2a = fg2m + fg2v
        fg3a = fg3m + fg3v
        fgm = fg2m + fg3m
        fga = fg2a + fg3a
        fta = ftm + ftv
        pts = 2 * fg2m + 3 * fg3m + ftm

        oreb = int(inputs.get("OR", 0) or 0)
        dreb = int(inputs.get("DR", 0) or 0)
        reb = oreb + dreb

        ast = int(inputs.get("AS", 0) or 0)
        stl = int(inputs.get("ST", 0) or 0)
        blk = int(inputs.get("BL", 0) or 0)
        to = int(inputs.get("TO", 0) or 0)

        fz = int(inputs.get("FZ", 0) or 0)
        of = int(inputs.get("OF", 0) or 0)
        fouls_drawn = fz + of

        pf = int(inputs.get("PF", 0) or 0)
        plus_minus = int(inputs.get("plus_minus", 0) or 0)
        min_str = str(inputs.get("min", "00:00")).strip()
        min_sec = parse_time_to_seconds(min_str)

        # Automatically mark played if player has any positive stat or minutes
        if (fga > 0 or fta > 0 or reb > 0 or ast > 0 or stl > 0 or blk > 0 or to > 0 or fouls_drawn > 0 or pf > 0 or min_sec > 0):
            played = True

        # Percentages
        fg2_pct = round(fg2m / fg2a, 3) if fg2a > 0 else None
        fg3_pct = round(fg3m / fg3a, 3) if fg3a > 0 else None
        fg_pct = round(fgm / fga, 3) if fga > 0 else None
        ft_pct = round(ftm / fta, 3) if fta > 0 else None

        # Advanced Analytics
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        efg_pct = round((fgm + 0.5 * fg3m) / fga, 3) if fga > 0 else None
        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        denom_ts = 2.0 * (fga + 0.44 * fta)
        ts_pct = round(pts / denom_ts, 3) if denom_ts > 0 else None
        # AS/TO ratio
        ast_to = round(ast / to, 2) if to > 0 else (float(ast) if ast > 0 else None)

        # Pro 40 scaling (Wert / Minuten * 40)
        min_val = min_sec / 60.0
        pro_40 = {}
        if min_val > 0:
            scale = 40.0 / min_val
            pro_40 = {
                "pts_40": round(pts * scale, 1),
                "reb_40": round(reb * scale, 1),
                "ast_40": round(ast * scale, 1),
                "stl_40": round(stl * scale, 1),
                "blk_40": round(blk * scale, 1),
                "to_40": round(to * scale, 1),
            }

        return {
            "jersey": jersey,
            "name": name,
            "played": 1 if played else 0,
            "min": min_str if min_sec > 0 else "",
            "min_sec": min_sec,
            "pts": pts,
            "2P_T": fg2m,
            "2P_V": fg2a,
            "2P_PCT": fg2_pct,
            "3P_T": fg3m,
            "3P_V": fg3a,
            "3P_PCT": fg3_pct,
            "FG_T": fgm,
            "FG_V": fga,
            "FG_PCT": fg_pct,
            "FW_T": ftm,
            "FW_V": fta,
            "FW_PCT": ft_pct,
            "OR": oreb,
            "DR": dreb,
            "Reb": reb,
            "AS": ast,
            "ST": stl,
            "BL": blk,
            "TO": to,
            "FZ": fz,
            "OF": of,
            "fouls_drawn": fouls_drawn,
            "PF": pf,
            "plus_minus": plus_minus,
            "efg_pct": efg_pct,
            "ts_pct": ts_pct,
            "ast_to": ast_to,
            "pro_40": pro_40,
        }

    @classmethod
    def calculate_boxscore(cls,
                            roster: List[dict],
                            players_data: Dict[str, dict],
                            has_app_data: bool = False,
                            target_sbu_score: Optional[int] = None) -> dict:
        """Computes entire team boxscore and control balance."""
        player_lines = []
        team_totals = {
            "played": 0, "min_sec": 0, "pts": 0,
            "2P_T": 0, "2P_V": 0, "3P_T": 0, "3P_V": 0,
            "FG_T": 0, "FG_V": 0, "FW_T": 0, "FW_V": 0,
            "OR": 0, "DR": 0, "Reb": 0, "AS": 0, "ST": 0,
            "BL": 0, "TO": 0, "FZ": 0, "OF": 0, "fouls_drawn": 0,
            "PF": 0, "plus_minus": 0
        }

        for p in roster:
            j = str(p["jersey"])
            name = p.get("name", f"#{j}")
            p_inputs = players_data.get(j, {})
            line = cls.calculate_player_line(j, name, p_inputs, has_app_data=has_app_data)
            player_lines.append(line)

            if line["played"]:
                team_totals["played"] += 1
            team_totals["min_sec"] += line["min_sec"]
            team_totals["pts"] += line["pts"]
            team_totals["2P_T"] += line["2P_T"]
            team_totals["2P_V"] += line["2P_V"]
            team_totals["3P_T"] += line["3P_T"]
            team_totals["3P_V"] += line["3P_V"]
            team_totals["FG_T"] += line["FG_T"]
            team_totals["FG_V"] += line["FG_V"]
            team_totals["FW_T"] += line["FW_T"]
            team_totals["FW_V"] += line["FW_V"]
            team_totals["OR"] += line["OR"]
            team_totals["DR"] += line["DR"]
            team_totals["Reb"] += line["Reb"]
            team_totals["AS"] += line["AS"]
            team_totals["ST"] += line["ST"]
            team_totals["BL"] += line["BL"]
            team_totals["TO"] += line["TO"]
            team_totals["FZ"] += line["FZ"]
            team_totals["OF"] += line["OF"]
            team_totals["fouls_drawn"] += line["fouls_drawn"]
            team_totals["PF"] += line["PF"]
            team_totals["plus_minus"] += line["plus_minus"]

        # Team percentages
        team_totals["2P_PCT"] = round(team_totals["2P_T"] / team_totals["2P_V"], 3) if team_totals["2P_V"] > 0 else None
        team_totals["3P_PCT"] = round(team_totals["3P_T"] / team_totals["3P_V"], 3) if team_totals["3P_V"] > 0 else None
        team_totals["FG_PCT"] = round(team_totals["FG_T"] / team_totals["FG_V"], 3) if team_totals["FG_V"] > 0 else None
        team_totals["FW_PCT"] = round(team_totals["FW_T"] / team_totals["FW_V"], 3) if team_totals["FW_V"] > 0 else None

        # Team advanced metrics
        fga = team_totals["FG_V"]
        fgm = team_totals["FG_T"]
        fg3m = team_totals["3P_T"]
        fta = team_totals["FW_V"]
        pts = team_totals["pts"]
        team_totals["efg_pct"] = round((fgm + 0.5 * fg3m) / fga, 3) if fga > 0 else None
        denom_ts = 2.0 * (fga + 0.44 * fta)
        team_totals["ts_pct"] = round(pts / denom_ts, 3) if denom_ts > 0 else None
        team_totals["ast_to"] = round(team_totals["AS"] / max(1, team_totals["TO"]), 2)

        # Team total time
        team_totals["min_formatted"] = format_seconds_to_time(team_totals["min_sec"])

        # Control Check: Boxscore Points vs Official Final Score
        control_status = "PENDING"
        control_message = "Endergebnis oben eintragen"
        diff = 0
        if target_sbu_score is not None:
            diff = team_totals["pts"] - target_sbu_score
            if diff == 0:
                control_status = "OK"
                control_message = "✓ Boxscore-Punkte = Endergebnis"
            else:
                control_status = "ERROR"
                control_message = f"⚠️ Differenz: Boxscore ({team_totals['pts']}) ≠ Endergebnis ({target_sbu_score}) [Diff: {diff:+d}]"

        return {
            "players": player_lines,
            "team_totals": team_totals,
            "control": {
                "status": control_status,
                "message": control_message,
                "diff": diff,
                "boxscore_pts": team_totals["pts"],
                "target_score": target_sbu_score
            }
        }

    @classmethod
    def aggregate_protocol(cls,
                           protocol: List[dict],
                           base_app_data: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
        """
        Converts a list of sequential protocol actions (Variante B) into aggregated player stats.
        If base_app_data is provided (from App with 2T, 3T, FT, Min, PF, +/-), it merges video stats!
        """
        aggregated: Dict[str, dict] = {}

        # Initialize with app data if present
        if base_app_data:
            for j, data in base_app_data.items():
                aggregated[str(j)] = dict(data)

        for entry in protocol:
            j = str(entry.get("jersey", "")).strip()
            code = str(entry.get("code", "")).strip().upper()
            if not j or code not in ACTION_CODES:
                continue

            if j not in aggregated:
                aggregated[j] = {
                    "played": True,
                    "2T": 0, "2V": 0, "3T": 0, "3V": 0,
                    "FT": 0, "FV": 0, "OR": 0, "DR": 0,
                    "AS": 0, "ST": 0, "BL": 0, "TO": 0,
                    "FZ": 0, "OF": 0, "min": "00:00", "PF": 0, "plus_minus": 0
                }

            aggregated[j]["played"] = True
            cfg = ACTION_CODES[code]

            # In games with App-Data: 2T, 3T, FT are fixed from app!
            # Only count video actions if not duplicate
            if code in ("2T", "2V", "3T", "3V", "FT", "FV", "OR", "DR", "AS", "ST", "BL", "TO", "FZ", "OF"):
                aggregated[j][code] = aggregated[j].get(code, 0) + 1

        return aggregated
