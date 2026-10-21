"""
coach_report.py
===============
Automated AI Coach Game Summary & Four Factors Analytics Report.

Replicates and extends commercial coach features found in SportsVisio 'Coach Mode'
and Hudl Assist:
- Dean Oliver's Four Factors of Basketball (eFG%, TOV%, ORB%, FT Rate)
- Advanced Pace and Possession Ratings (ORTG, DRTG, Net Rating)
- Granular Turnover Diagnosis across all FIBA rule violations
- Automated Executive Game Summary with coaching insights and tactical takeaways
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional


class CoachReportGenerator:
    """Generates comprehensive coach reports and Four Factors analytics."""

    def __init__(self, team_name: str = "SB Uffenheim"):
        self.team_name = team_name

    def generate_report(self,
                        tracked_events: dict,
                        canonical_players: dict,
                        game_duration_min: float = 40.0) -> dict:
        """
        Computes all advanced coach metrics and returns a structured report dictionary.
        """
        pts = tracked_events.get("points", 0)
        fgm = tracked_events.get("fgm", 0)
        fga = tracked_events.get("fga", 0)
        fg2m = tracked_events.get("fg2m", 0)
        fg2a = tracked_events.get("fg2a", 0)
        fg3m = tracked_events.get("fg3m", 0)
        fg3a = tracked_events.get("fg3a", 0)
        ftm = tracked_events.get("ftm", 0)
        fta = tracked_events.get("fta", 0)
        oreb = tracked_events.get("oreb", 0)
        dreb = tracked_events.get("dreb", 0)
        reb = tracked_events.get("reb", oreb + dreb)
        ast = tracked_events.get("assists", 0)
        to = tracked_events.get("turnovers", 0)
        stl = tracked_events.get("steals", 0)
        blk = tracked_events.get("blocks", 0)

        # Granular turnover breakdown
        to_travel = tracked_events.get("turnovers_traveling", 0)
        to_pass = tracked_events.get("turnovers_bad_pass", 0)
        to_3sec = tracked_events.get("turnovers_three_seconds", 0)
        to_backcourt = tracked_events.get("turnovers_backcourt", 0)
        to_oob = tracked_events.get("turnovers_out_of_bounds", 0)
        to_double_dribble = tracked_events.get("turnovers_double_dribble", 0)
        to_8sec = tracked_events.get("turnovers_eight_seconds", 0)
        to_5sec = tracked_events.get("turnovers_five_seconds", 0)

        # 1. Possessions Estimation (FIBA / Oliver Standard Formula)
        # Possessions = FGA + 0.44 * FTA - OREB + TO
        est_possessions = max(1.0, fga + 0.44 * fta - oreb + to)
        pace = round((est_possessions / game_duration_min) * 40.0, 1)

        # 2. Dean Oliver's Four Factors
        # Factor 1: Effective Field Goal % (40% weight)
        efg_pct = round(((fgm + 0.5 * fg3m) / fga * 100.0), 1) if fga > 0 else 0.0
        # Factor 2: Turnover Percentage (25% weight)
        tov_pct = round((to / est_possessions) * 100.0, 1)
        # Factor 3: Offensive Rebound Percentage (20% weight)
        # Assuming opponent defensive rebounds ~ 25 if not provided
        opp_dreb_est = max(10, 35 - oreb)
        orb_pct = round((oreb / (oreb + opp_dreb_est)) * 100.0, 1)
        # Factor 4: Free Throw Rate (15% weight)
        ft_rate = round((fta / fga), 3) if fga > 0 else 0.0

        # 3. Efficiency Ratings
        ortg = round((pts / est_possessions) * 100.0, 1)
        # Estimated True Shooting Percentage (TS%)
        ts_pct = round((pts / (2.0 * max(1.0, fga + 0.44 * fta))) * 100.0, 1)
        ast_to_ratio = round(ast / max(1, to), 2)

        # 4. Shot Selection Splits
        pct_3pt_attempts = round((fg3a / max(1, fga)) * 100.0, 1)
        fg2_pct = round((fg2m / max(1, fg2a)) * 100.0, 1)
        fg3_pct = round((fg3m / max(1, fg3a)) * 100.0, 1)
        ft_pct = round((ftm / max(1, fta)) * 100.0, 1)

        # 5. Top Performers
        top_scorers = sorted(
            [{"jersey": j, "name": s.get("name", f"#{j}"), "pts": s.get("points", 0), "fgm": s.get("fgm", 0), "fga": s.get("fga", 0)}
             for j, s in canonical_players.items()],
            key=lambda p: p["pts"], reverse=True
        )

        # 6. Automated Coach Insights / Takeaways
        takeaways = []
        if efg_pct >= 48.0:
            takeaways.append(f"✅ Starke Wurfeffizienz: eFG% von {efg_pct}% liegt im überdurchschnittlichen Bereich.")
        else:
            takeaways.append(f"⚠️ Ausbaufähige Wurfeffizienz: eFG% von {efg_pct}% (3er: {fg3_pct}%, 2er: {fg2_pct}%).")

        if tov_pct <= 18.0:
            takeaways.append(f"✅ Gute Ballkontrolle: TOV% von {tov_pct}% ist stabil.")
        else:
            takeaways.append(f"🚨 Hohe Ballverlustrate: TOV% von {tov_pct}% ({to} Turnovers: {to_travel} Schrittfehler, {to_pass} Fehlpässe).")

        if ast_to_ratio >= 1.0:
            takeaways.append(f"✅ Ausgewogenes Ball-Movement mit AST/TO Ratio von {ast_to_ratio}.")
        else:
            takeaways.append(f"⚠️ Ball-Movement verbesserungsfähig: AST/TO Ratio liegt bei {ast_to_ratio}.")

        report_data = {
            "team_name": self.team_name,
            "summary": {
                "points": pts,
                "possessions": round(est_possessions, 1),
                "pace": pace,
                "ortg": ortg,
                "ts_pct": ts_pct,
                "efg_pct": efg_pct,
                "ast_to_ratio": ast_to_ratio,
            },
            "four_factors": {
                "efg_pct": efg_pct,
                "tov_pct": tov_pct,
                "orb_pct": orb_pct,
                "ft_rate": ft_rate,
            },
            "shot_splits": {
                "fgm": fgm, "fga": fga, "fg_pct": round(fgm / max(1, fga) * 100, 1),
                "fg2m": fg2m, "fg2a": fg2a, "fg2_pct": fg2_pct,
                "fg3m": fg3m, "fg3a": fg3a, "fg3_pct": fg3_pct,
                "three_point_attempt_rate": pct_3pt_attempts,
                "ftm": ftm, "fta": fta, "ft_pct": ft_pct,
            },
            "turnover_diagnosis": {
                "total_turnovers": to,
                "traveling": to_travel,
                "bad_pass": to_pass,
                "three_seconds": to_3sec,
                "backcourt": to_backcourt,
                "out_of_bounds": to_oob,
                "double_dribble": to_double_dribble,
                "eight_seconds": to_8sec,
                "five_seconds": to_5sec,
            },
            "defense_and_hustle": {
                "steals": stl,
                "blocks": blk,
                "dreb": dreb,
                "oreb": oreb,
                "reb": reb,
            },
            "top_scorers": top_scorers,
            "coach_takeaways": takeaways,
        }
        return report_data

    def export_markdown_report(self, report_data: dict, output_path: str):
        """Writes a formatted GitHub-style markdown coach report."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        s = report_data["summary"]
        ff = report_data["four_factors"]
        ss = report_data["shot_splits"]
        to = report_data["turnover_diagnosis"]

        md = f"""# 🏀 AI Coach Game Report: {report_data['team_name']}

## 1. Executive Summary & Pace
* **Punkte:** {s['points']}
* **Geschätzte Ballbesitze (Possessions):** {s['possessions']} | **Pace (40 Min):** {s['pace']}
* **Offensive Rating (ORTG):** {s['ortg']} Pkt / 100 Possessions
* **True Shooting (TS%):** {s['ts_pct']}% | **Effective FG (eFG%):** {s['efg_pct']}%
* **Assist-to-Turnover Ratio:** {s['ast_to_ratio']}

## 2. Dean Oliver's Four Factors of Basketball
| Faktor | Wert | Benchmark-Ziel | Bewertung |
| :--- | :---: | :---: | :--- |
| **1. Shooting (eFG%)** | **{ff['efg_pct']}%** | > 50.0% | {'✅ Gut' if ff['efg_pct'] >= 48 else '⚠️ Steigerungsfähig'} |
| **2. Turnovers (TOV%)** | **{ff['tov_pct']}%** | < 16.0% | {'✅ Stabil' if ff['tov_pct'] <= 18 else '🚨 Zu hoch'} |
| **3. Offensiv-Rebound (ORB%)** | **{ff['orb_pct']}%** | > 28.0% | {'✅ Aktiv' if ff['orb_pct'] >= 25 else '⚠️ Box-Out schärfen'} |
| **4. Free Throw Rate (FTR)** | **{ff['ft_rate']}** | > 0.220 | {'✅ Zug zum Korb' if ff['ft_rate'] >= 0.20 else '⚠️ Mehr Freiwürfe ziehen'} |

## 3. Detaillierte Turnover-Diagnose (FIBA Rules)
* **Gesamt-Turnovers:** {to['total_turnovers']}
  * **Schrittfehler (Traveling, Art. 25):** {to['traveling']}
  * **Fehlpässe / Interceptions:** {to['bad_pass']}
  * **3 Sekunden in der Zone (Art. 26):** {to['three_seconds']}
  * **Rückspiel / Over-and-Back (Art. 30):** {to['backcourt']}
  * **Ballführer im Aus (Art. 23):** {to['out_of_bounds']}
  * **Zweimaliges Dribbeln (Art. 24):** {to['double_dribble']}
  * **8 Sekunden im Rückfeld (Art. 28):** {to['eight_seconds']}
  * **5 Sekunden eng gedeckt (Art. 27):** {to['five_seconds']}

## 4. Taktische Coach-Takeaways
"""
        for t in report_data["coach_takeaways"]:
            md += f"- {t}\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
