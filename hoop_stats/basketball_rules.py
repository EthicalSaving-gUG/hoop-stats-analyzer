"""
Basketball Rule Sets — FIBA 2024 and NBA 2024/25.

Provides dataclass-based rule configurations that agents use
to enforce the correct thresholds for each league.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class BasketballRules:
    """All rule parameters that differ between FIBA and NBA."""

    name: str

    # ── Game Clock ───────────────────────────────────────────────────────
    quarters: int                       # Number of regulation quarters
    quarter_length_min: float           # Minutes per quarter
    overtime_length_min: float          # Minutes per overtime period

    # ── Shot Clock ──────────────────────────────────────────────────────
    shot_clock_s: float                 # Full shot clock (seconds)
    shot_clock_reset_offensive_reb: float  # Reset after offensive rebound
    shot_clock_reset_foul_frontcourt: float  # Reset after foul in frontcourt

    # ── Fouls ───────────────────────────────────────────────────────────
    personal_foul_limit: int            # Fouls before disqualification
    team_foul_bonus_threshold: int      # Team fouls per quarter for bonus
    technical_counts_as_personal: bool  # Does a technical count toward foul limit?

    # ── Timeouts ────────────────────────────────────────────────────────
    timeouts_first_half: int
    timeouts_second_half: int
    timeout_duration_s: int             # Seconds per timeout
    players_can_call_timeout: bool

    # ── Court Dimensions ────────────────────────────────────────────────
    court_length_m: float
    court_width_m: float
    three_point_distance_m: float       # Arc distance (not corner)

    # ── Special Rules ───────────────────────────────────────────────────
    defensive_three_seconds: bool       # Defensive 3-second violation?
    goaltending_after_rim_touch: bool   # Can you swat after rim touch?
    backcourt_violation_s: float        # Seconds to advance past half-court
    free_throw_lane_width_m: float      # Paint width

    # ── Foul Types ──────────────────────────────────────────────────────
    has_flagrant_fouls: bool            # NBA-style Flagrant 1/2?
    has_unsportsmanlike_fouls: bool     # FIBA-style Unsportsmanlike?
    unsportsmanlike_disqualify_count: int  # How many before ejection?


# ── FIBA Official Rules 2024 ────────────────────────────────────────────
# Source: FIBA Official Basketball Rules 2024 (v10a)
FIBA_RULES = BasketballRules(
    name="FIBA 2024",

    quarters=4,
    quarter_length_min=10.0,
    overtime_length_min=5.0,

    shot_clock_s=24.0,
    shot_clock_reset_offensive_reb=14.0,
    shot_clock_reset_foul_frontcourt=14.0,

    personal_foul_limit=5,
    team_foul_bonus_threshold=5,       # Art. 41 — bonus after 4th team foul
    technical_counts_as_personal=True,

    timeouts_first_half=2,
    timeouts_second_half=3,
    timeout_duration_s=60,
    players_can_call_timeout=False,     # Only coach

    court_length_m=28.0,
    court_width_m=15.0,
    three_point_distance_m=6.75,

    defensive_three_seconds=False,
    goaltending_after_rim_touch=False,  # Ball is live after rim contact
    backcourt_violation_s=8.0,          # 8-second backcourt rule
    free_throw_lane_width_m=4.9,       # Trapezoid/rectangular lane

    has_flagrant_fouls=False,
    has_unsportsmanlike_fouls=True,
    unsportsmanlike_disqualify_count=2,
)


# ── NBA Official Rules 2024-25 ──────────────────────────────────────────
# Source: NBA Official Rulebook (official.nba.com/rulebook/)
NBA_RULES = BasketballRules(
    name="NBA 2024-25",

    quarters=4,
    quarter_length_min=12.0,
    overtime_length_min=5.0,

    shot_clock_s=24.0,
    shot_clock_reset_offensive_reb=14.0,
    shot_clock_reset_foul_frontcourt=14.0,

    personal_foul_limit=6,
    team_foul_bonus_threshold=5,       # Rule 12-B — bonus on 5th team foul
    technical_counts_as_personal=False,

    timeouts_first_half=4,             # 7 total, max 4 in 4th quarter
    timeouts_second_half=3,
    timeout_duration_s=75,             # Full timeout = 75s, 20s = short
    players_can_call_timeout=True,

    court_length_m=28.65,              # 94 feet
    court_width_m=15.24,               # 50 feet
    three_point_distance_m=7.24,       # 23 feet 9 inches (arc)

    defensive_three_seconds=True,
    goaltending_after_rim_touch=True,   # Cannot touch after rim in NBA
    backcourt_violation_s=8.0,          # 8-second backcourt (was 10)
    free_throw_lane_width_m=4.88,       # 16 feet

    has_flagrant_fouls=True,
    has_unsportsmanlike_fouls=False,
    unsportsmanlike_disqualify_count=0,  # Not applicable
)


# ── Registry ────────────────────────────────────────────────────────────
RULE_SETS: Dict[str, BasketballRules] = {
    "fiba": FIBA_RULES,
    "nba": NBA_RULES,
}

def get_rules(name: str = "fiba") -> BasketballRules:
    """Return a rule set by key ('fiba' or 'nba')."""
    return RULE_SETS.get(name.lower(), FIBA_RULES)
