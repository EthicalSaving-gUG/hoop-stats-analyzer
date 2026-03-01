"""
Thread-safe shared game state and event bus for multi-agent Kampfgericht.
All agents read/write through this central state object.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .basketball_rules import BasketballRules, FIBA_RULES


@dataclass
class GameEvent:
    """A single timestamped event published by any agent."""
    timestamp: float          # time.time()
    agent: str                # e.g. "shot_clock", "game_clock", "scoresheet", "referee"
    event_type: str           # e.g. "SCORE", "FOUL", "VIOLATION", "QUARTER_END"
    data: Dict[str, Any] = field(default_factory=dict)
    frame_ref: Optional[int] = None   # video frame index for replay


class GameState:
    """
    Central, thread-safe state shared by all Kampfgericht agents.
    Protected by a reentrant lock so any agent thread can safely read/write.
    """

    def __init__(self, rules: BasketballRules = None):
        self._lock = threading.RLock()
        self.rules = rules or FIBA_RULES

        # ── Game Clock ──────────────────────────────────────────────────────
        self.quarter: int = 1
        self.quarter_length_s: float = self.rules.quarter_length_min * 60
        self.game_clock_s: float = self.quarter_length_s
        self.game_clock_running: bool = False
        self.game_over: bool = False

        # ── Shot Clock ──────────────────────────────────────────────────────
        self.shot_clock_s: float = self.rules.shot_clock_s
        self.shot_clock_running: bool = False
        self.shot_clock_max: float = self.rules.shot_clock_s
        self.shot_clock_short: float = self.rules.shot_clock_reset_offensive_reb

        # ── Score ───────────────────────────────────────────────────────────
        self.score_home: int = 0
        self.score_away: int = 0
        self.team_home_name: str = "Team Schwarz"
        self.team_away_name: str = "Team Gelb"

        # ── Fouls (per quarter) ─────────────────────────────────────────────
        self.team_fouls_home: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        self.team_fouls_away: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        self.player_fouls: Dict[str, int] = {}   # player_id -> total personal fouls

        # ── Possession ──────────────────────────────────────────────────────
        self.possession: Optional[str] = None     # "home" / "away" / None

        # ── Event Log ───────────────────────────────────────────────────────
        self.events: List[GameEvent] = []

        # ── Player Stats (Kampfbericht) ─────────────────────────────────────
        self.player_stats: Dict[str, Dict[str, Any]] = {}
        # { player_id: { "name": "", "jersey": "", "team": "", "points": 0,
        #                "fouls": 0, "fg_made": 0, "fg_att": 0, ... } }

    # ── Thread-safe helpers ─────────────────────────────────────────────────

    def append_event(self, agent: str, event_type: str,
                     data: dict = None, frame_ref: int = None):
        with self._lock:
            evt = GameEvent(
                timestamp=time.time(),
                agent=agent,
                event_type=event_type,
                data=data or {},
                frame_ref=frame_ref,
            )
            self.events.append(evt)
            return evt

    def add_score(self, team: str, points: int, scorer_id: str = None):
        with self._lock:
            if team == "home":
                self.score_home += points
            else:
                self.score_away += points
            if scorer_id and scorer_id in self.player_stats:
                self.player_stats[scorer_id]["points"] = \
                    self.player_stats[scorer_id].get("points", 0) + points

    def add_foul(self, team: str, player_id: str = None):
        with self._lock:
            q = self.quarter
            if team == "home":
                self.team_fouls_home[q] = self.team_fouls_home.get(q, 0) + 1
            else:
                self.team_fouls_away[q] = self.team_fouls_away.get(q, 0) + 1
            if player_id:
                self.player_fouls[player_id] = \
                    self.player_fouls.get(player_id, 0) + 1

    def set_possession(self, team: Optional[str]):
        with self._lock:
            self.possession = team

    def reset_shot_clock(self, short: bool = False):
        with self._lock:
            self.shot_clock_s = self.shot_clock_short if short else self.shot_clock_max
            self.shot_clock_running = True

    def tick(self, dt: float):
        """Advance clocks by dt seconds. Called by game_clock_agent."""
        with self._lock:
            if self.game_clock_running and self.game_clock_s > 0:
                self.game_clock_s = max(0.0, self.game_clock_s - dt)
            if self.shot_clock_running and self.shot_clock_s > 0:
                self.shot_clock_s = max(0.0, self.shot_clock_s - dt)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot for the API."""
        with self._lock:
            return {
                "quarter": self.quarter,
                "game_clock": round(self.game_clock_s, 1),
                "game_clock_running": self.game_clock_running,
                "shot_clock": round(self.shot_clock_s, 1),
                "shot_clock_running": self.shot_clock_running,
                "possession": self.possession,
                "score_home": self.score_home,
                "score_away": self.score_away,
                "team_home": self.team_home_name,
                "team_away": self.team_away_name,
                "team_fouls_home": dict(self.team_fouls_home),
                "team_fouls_away": dict(self.team_fouls_away),
                "game_over": self.game_over,
                "ruleset": self.rules.name,
                "personal_foul_limit": self.rules.personal_foul_limit,
                "recent_events": [
                    {
                        "time": e.timestamp,
                        "agent": e.agent,
                        "type": e.event_type,
                        "data": e.data,
                    }
                    for e in self.events[-20:]   # last 20 events
                ],
            }
