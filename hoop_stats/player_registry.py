"""
player_registry.py
==================
Dynamic Player Registry with Late Identity Binding and Substitution Handling.

Architecture:
1. Tracks each player via anonymous ByteTrack IDs (Tracklets).
2. Buffers actions (shots, rebounds, turnovers, fouls) under the active track ID.
3. Opportunistically collects OCR jersey number observations during high-clarity moments
   (e.g., dead balls, free throws, inbounds, facing camera).
4. Applies Roster Constraint & Confidence Voting to eliminate visual noise/hallucinations.
5. Once a jersey number is confirmed, retroactively binds all historical and future events
   of that tracklet to the official jersey number.
6. Seamlessly handles basketball substitutions by merging multiple stints/tracklets of the
   same player into a single canonical stat profile.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class PlayerStatSheet:
    jersey: str
    team: str = "home"
    name: Optional[str] = None
    points: int = 0
    fgm: int = 0
    fga: int = 0
    fg2m: int = 0
    fg2a: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    reb: int = 0
    assists: int = 0
    turnovers: int = 0
    turnovers_traveling: int = 0
    turnovers_bad_pass: int = 0
    turnovers_three_seconds: int = 0
    turnovers_backcourt: int = 0
    turnovers_out_of_bounds: int = 0
    turnovers_double_dribble: int = 0
    turnovers_eight_seconds: int = 0
    turnovers_five_seconds: int = 0
    steals: int = 0
    blocks: int = 0
    fouls_drawn: int = 0
    associated_tracks: Set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "jersey": self.jersey,
            "team": self.team,
            "name": self.name,
            "points": self.points,
            "fgm": self.fgm,
            "fga": self.fga,
            "fg2m": self.fg2m,
            "fg2a": self.fg2a,
            "fg3m": self.fg3m,
            "fg3a": self.fg3a,
            "ftm": self.ftm,
            "fta": self.fta,
            "oreb": self.oreb,
            "dreb": self.dreb,
            "reb": self.reb,
            "assists": self.assists,
            "turnovers": self.turnovers,
            "turnovers_traveling": self.turnovers_traveling,
            "turnovers_bad_pass": self.turnovers_bad_pass,
            "turnovers_three_seconds": self.turnovers_three_seconds,
            "turnovers_backcourt": self.turnovers_backcourt,
            "turnovers_out_of_bounds": self.turnovers_out_of_bounds,
            "turnovers_double_dribble": self.turnovers_double_dribble,
            "turnovers_eight_seconds": self.turnovers_eight_seconds,
            "turnovers_five_seconds": self.turnovers_five_seconds,
            "steals": self.steals,
            "blocks": self.blocks,
            "fouls_drawn": self.fouls_drawn,
            "associated_tracks": sorted([int(t) for t in self.associated_tracks]),
        }


class PlayerRegistry:
    """
    Manages tracklets, late jersey binding, substitutions, and canonical player statistics.
    """

    def __init__(self,
                 roster_home: Optional[List[str]] = None,
                 roster_away: Optional[List[str]] = None,
                 voting_threshold: float = 1.35,
                 single_hit_conf: float = 0.82):
        # Known team rosters (filters out impossible OCR digits)
        self.roster_home: Set[str] = set(str(j) for j in (roster_home or []))
        self.roster_away: Set[str] = set(str(j) for j in (roster_away or []))

        self.voting_threshold = voting_threshold
        self.single_hit_conf = single_hit_conf

        # Active Tracklet -> Canonical Jersey Mapping
        self.track_to_jersey: Dict[int, str] = {}
        # Tracklet -> Team ("home", "away", "unknown")
        self.track_to_team: Dict[int, str] = {}
        # Tracklet -> On court status
        self.track_on_court: Dict[int, bool] = {}

        # Tracklet -> Dict of {jersey_str: cumulative_confidence}
        self.track_votes: Dict[int, Dict[str, float]] = {}

        # Tracklet -> List of buffered raw action events before jersey is identified
        self.pending_track_events: Dict[int, List[dict]] = {}

        # Canonical Player Database: jersey_str -> PlayerStatSheet
        self.canonical_players: Dict[str, PlayerStatSheet] = {}

    def register_track(self, track_id: int, team: str = "home", is_on_court: bool = True):
        """Registers or updates a tracklet with its team and on-court state."""
        if track_id not in self.track_to_team or self.track_to_team[track_id] == "unknown":
            self.track_to_team[track_id] = team
        self.track_on_court[track_id] = is_on_court
        if track_id not in self.track_votes:
            self.track_votes[track_id] = {}
        if track_id not in self.pending_track_events:
            self.pending_track_events[track_id] = []

    def set_substitution_state(self, track_id: int, is_on_court: bool):
        """Updates whether a tracklet is currently on the court or on the bench."""
        self.track_on_court[track_id] = is_on_court

    def record_ocr_observation(self, track_id: int, number_str: Optional[str],
                                confidence: float, team: Optional[str] = None) -> Optional[str]:
        """
        Records an OCR number observation for track_id.
        Applies roster constraints, accumulates confidence votes, and triggers binding when consensus is reached.
        Returns the bound jersey number if resolved, else None.
        """
        if not number_str or confidence <= 0.10:
            return self.track_to_jersey.get(track_id)

        clean_num = str(number_str).strip()

        # If already bound, verify if consistent
        if track_id in self.track_to_jersey:
            return self.track_to_jersey[track_id]

        track_team = team or self.track_to_team.get(track_id, "home")

        # Roster constraint check
        if track_team == "home" and self.roster_home:
            if clean_num not in self.roster_home:
                return None
        elif track_team == "away" and self.roster_away:
            if clean_num not in self.roster_away:
                return None

        # Record vote
        if track_id not in self.track_votes:
            self.track_votes[track_id] = {}
        curr_vote = self.track_votes[track_id].get(clean_num, 0.0) + confidence
        self.track_votes[track_id][clean_num] = curr_vote

        # Decision threshold: High single confidence or cumulative weighted threshold
        if confidence >= self.single_hit_conf or curr_vote >= self.voting_threshold:
            self.bind_jersey(track_id, clean_num, team=track_team)
            return clean_num

        return None

    def bind_jersey(self, track_id: int, jersey_str: str, team: Optional[str] = None):
        """
        Binds a tracklet to a canonical jersey number.
        Retroactively flushes all pending actions of this tracklet to the canonical player.
        """
        jersey_clean = str(jersey_str).strip()
        player_team = team or self.track_to_team.get(track_id, "home")

        self.track_to_jersey[track_id] = jersey_clean
        self.track_to_team[track_id] = player_team

        # Retrieve or initialize canonical player
        if jersey_clean not in self.canonical_players:
            self.canonical_players[jersey_clean] = PlayerStatSheet(
                jersey=jersey_clean,
                team=player_team,
            )

        player = self.canonical_players[jersey_clean]
        player.associated_tracks.add(track_id)

        # Retroactive Event Migration: Flush buffered events for this track
        if track_id in self.pending_track_events:
            for evt in self.pending_track_events[track_id]:
                self._apply_event_to_player(player, evt)
            self.pending_track_events[track_id].clear()

    def record_action(self, track_id: int, action_type: str, points: int = 0,
                      details: Optional[dict] = None) -> str:
        """
        Records an event (e.g. SHOT_MAKE, SHOT_MISS, REBOUND, TURNOVER).
        If track_id is already bound to a jersey, applies immediately to canonical player.
        Otherwise, buffers it under track_id pending identity resolution.
        Returns the resolved identifier (jersey string or 'Track#X').
        """
        event = {
            "type": action_type,
            "points": points,
            "details": details or {},
        }

        jersey = self.track_to_jersey.get(track_id)
        if jersey and jersey in self.canonical_players:
            player = self.canonical_players[jersey]
            self._apply_event_to_player(player, event)
            return jersey

        # Buffer for late binding
        if track_id not in self.pending_track_events:
            self.pending_track_events[track_id] = []
        self.pending_track_events[track_id].append(event)
        return f"Track#{track_id}"

    def _apply_event_to_player(self, player: PlayerStatSheet, event: dict):
        """Applies a single basketball event to a player's canonical stat sheet."""
        evt_type = event.get("type", "").upper()
        pts = event.get("points", 0)
        details = event.get("details", {})
        zone = str(details.get("zone", "")).lower()

        is_three = "three" in zone
        is_ft = "free_throw" in zone or evt_type.startswith("FREE_THROW")

        if evt_type in ("SHOT_MAKE", "FREE_THROW_MAKE"):
            player.points += pts
            if is_ft:
                player.ftm += 1
                player.fta += 1
            elif is_three:
                player.fg3m += 1
                player.fg3a += 1
                player.fgm += 1
                player.fga += 1
            else:
                player.fg2m += 1
                player.fg2a += 1
                player.fgm += 1
                player.fga += 1

        elif evt_type in ("SHOT_MISS", "FREE_THROW_MISS"):
            if is_ft:
                player.fta += 1
            elif is_three:
                player.fg3a += 1
                player.fga += 1
            else:
                player.fg2a += 1
                player.fga += 1

        elif evt_type == "OFFENSIVE_REBOUND":
            player.oreb += 1
            player.reb += 1
        elif evt_type in ("DEFENSIVE_REBOUND", "REBOUND"):
            player.dreb += 1
            player.reb += 1
        elif evt_type == "ASSIST":
            player.assists += 1
        elif evt_type in ("TURNOVER", "TRAVELING", "BAD_PASS", "THREE_SECONDS_VIOLATION", "BACKCOURT_VIOLATION", "PLAYER_OUT_OF_BOUNDS", "DOUBLE_DRIBBLE", "EIGHT_SECONDS_VIOLATION", "FIVE_SECONDS_VIOLATION"):
            player.turnovers += 1
            reason = str(details.get("reason", "")).lower()
            if evt_type == "TRAVELING" or "travel" in reason or "schritt" in reason:
                player.turnovers_traveling += 1
            elif evt_type == "BAD_PASS" or "pass" in reason or "intercept" in reason:
                player.turnovers_bad_pass += 1
            elif evt_type == "THREE_SECONDS_VIOLATION" or "three_second" in reason or "3-sekund" in reason:
                player.turnovers_three_seconds += 1
            elif evt_type == "BACKCOURT_VIOLATION" or "backcourt" in reason or "rückspiel" in reason:
                player.turnovers_backcourt += 1
            elif evt_type == "PLAYER_OUT_OF_BOUNDS" or "out_of_bounds" in reason or "aus" in reason:
                player.turnovers_out_of_bounds += 1
            elif evt_type == "DOUBLE_DRIBBLE" or "double_dribble" in reason or "zweimal" in reason:
                player.turnovers_double_dribble += 1
            elif evt_type == "EIGHT_SECONDS_VIOLATION" or "eight_second" in reason or "8-sekund" in reason:
                player.turnovers_eight_seconds += 1
            elif evt_type == "FIVE_SECONDS_VIOLATION" or "five_second" in reason or "5-sekund" in reason:
                player.turnovers_five_seconds += 1
        elif evt_type == "STEAL":
            player.steals += 1
        elif evt_type == "BLOCK":
            player.blocks += 1
        elif evt_type == "FOUL_DRAWN":
            player.fouls_drawn += 1

    def get_canonical_roster_summary(self, team: str = "home") -> Dict[str, dict]:
        """Returns clean dictionary of all identified players and their stats for a team."""
        result = {}
        for jersey, sheet in self.canonical_players.items():
            if sheet.team == team:
                result[jersey] = sheet.to_dict()
        return result

    def get_team_totals(self, team: str = "home") -> dict:
        """Aggregates all canonical player stats for a team into team totals."""
        totals = {
            "points": 0,
            "fgm": 0, "fga": 0,
            "fg2m": 0, "fg2a": 0,
            "fg3m": 0, "fg3a": 0,
            "ftm": 0, "fta": 0,
            "oreb": 0, "dreb": 0, "reb": 0,
            "assists": 0, "turnovers": 0,
            "turnovers_traveling": 0, "turnovers_bad_pass": 0,
            "turnovers_three_seconds": 0, "turnovers_backcourt": 0, "turnovers_out_of_bounds": 0,
            "turnovers_double_dribble": 0, "turnovers_eight_seconds": 0, "turnovers_five_seconds": 0,
            "steals": 0, "blocks": 0,
            "fouls_drawn": 0,
        }
        for sheet in self.canonical_players.values():
            if sheet.team == team:
                totals["points"] += sheet.points
                totals["fgm"] += sheet.fgm
                totals["fga"] += sheet.fga
                totals["fg2m"] += sheet.fg2m
                totals["fg2a"] += sheet.fg2a
                totals["fg3m"] += sheet.fg3m
                totals["fg3a"] += sheet.fg3a
                totals["ftm"] += sheet.ftm
                totals["fta"] += sheet.fta
                totals["oreb"] += sheet.oreb
                totals["dreb"] += sheet.dreb
                totals["reb"] += sheet.reb
                totals["assists"] += sheet.assists
                totals["turnovers"] += sheet.turnovers
                totals["turnovers_traveling"] += sheet.turnovers_traveling
                totals["turnovers_bad_pass"] += sheet.turnovers_bad_pass
                totals["turnovers_three_seconds"] += sheet.turnovers_three_seconds
                totals["turnovers_backcourt"] += sheet.turnovers_backcourt
                totals["turnovers_out_of_bounds"] += sheet.turnovers_out_of_bounds
                totals["turnovers_double_dribble"] += sheet.turnovers_double_dribble
                totals["turnovers_eight_seconds"] += sheet.turnovers_eight_seconds
                totals["turnovers_five_seconds"] += sheet.turnovers_five_seconds
                totals["steals"] += sheet.steals
                totals["blocks"] += sheet.blocks
                totals["fouls_drawn"] += sheet.fouls_drawn

        # Include unbound pending events for players not yet identified
        for track_id, events in self.pending_track_events.items():
            if self.track_to_team.get(track_id, "home") == team:
                for evt in events:
                    evt_type = evt.get("type", "").upper()
                    pts = evt.get("points", 0)
                    zone = str(evt.get("details", {}).get("zone", "")).lower()
                    is_three = "three" in zone
                    is_ft = "free_throw" in zone or evt_type.startswith("FREE_THROW")

                    if evt_type in ("SHOT_MAKE", "FREE_THROW_MAKE"):
                        totals["points"] += pts
                        if is_ft:
                            totals["ftm"] += 1
                            totals["fta"] += 1
                        elif is_three:
                            totals["fg3m"] += 1
                            totals["fg3a"] += 1
                            totals["fgm"] += 1
                            totals["fga"] += 1
                        else:
                            totals["fg2m"] += 1
                            totals["fg2a"] += 1
                            totals["fgm"] += 1
                            totals["fga"] += 1
                    elif evt_type in ("SHOT_MISS", "FREE_THROW_MISS"):
                        if is_ft:
                            totals["fta"] += 1
                        elif is_three:
                            totals["fg3a"] += 1
                            totals["fga"] += 1
                        else:
                            totals["fg2a"] += 1
                            totals["fga"] += 1
                    elif evt_type == "OFFENSIVE_REBOUND":
                        totals["oreb"] += 1
                        totals["reb"] += 1
                    elif evt_type in ("DEFENSIVE_REBOUND", "REBOUND"):
                        totals["dreb"] += 1
                        totals["reb"] += 1
                    elif evt_type == "ASSIST":
                        totals["assists"] += 1
                    elif evt_type in ("TURNOVER", "TRAVELING", "BAD_PASS", "THREE_SECONDS_VIOLATION", "BACKCOURT_VIOLATION", "PLAYER_OUT_OF_BOUNDS"):
                        totals["turnovers"] += 1
                        reason = str(evt.get("details", {}).get("reason", "")).lower()
                        if evt_type == "TRAVELING" or "travel" in reason or "schritt" in reason:
                            totals["turnovers_traveling"] += 1
                        elif evt_type == "BAD_PASS" or "pass" in reason or "intercept" in reason:
                            totals["turnovers_bad_pass"] += 1
                        elif evt_type == "THREE_SECONDS_VIOLATION" or "three_second" in reason or "3-sekund" in reason:
                            totals["turnovers_three_seconds"] += 1
                        elif evt_type == "BACKCOURT_VIOLATION" or "backcourt" in reason or "rückspiel" in reason:
                            totals["turnovers_backcourt"] += 1
                        elif evt_type == "PLAYER_OUT_OF_BOUNDS" or "out_of_bounds" in reason or "aus" in reason:
                            totals["turnovers_out_of_bounds"] += 1
                        elif evt_type == "DOUBLE_DRIBBLE" or "double_dribble" in reason or "zweimal" in reason:
                            totals["turnovers_double_dribble"] += 1
                        elif evt_type == "EIGHT_SECONDS_VIOLATION" or "eight_second" in reason or "8-sekund" in reason:
                            totals["turnovers_eight_seconds"] += 1
                        elif evt_type == "FIVE_SECONDS_VIOLATION" or "five_second" in reason or "5-sekund" in reason:
                            totals["turnovers_five_seconds"] += 1
                    elif evt_type == "STEAL":
                        totals["steals"] += 1
                    elif evt_type == "BLOCK":
                        totals["blocks"] += 1
                    elif evt_type == "FOUL_DRAWN":
                        totals["fouls_drawn"] += 1

        return totals
