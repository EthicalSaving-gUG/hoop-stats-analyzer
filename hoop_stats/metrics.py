import numpy as np
from .primitives import Point2D


class PlayerMetrics:
    def __init__(self, fps=30, sprint_speed_threshold=4.5):
        self.fps = fps
        self.dt = 1.0 / fps
        self.sprint_speed_threshold = sprint_speed_threshold
        self.trajectories = {}  # {player_id: [(Point2D, timestamp)]}
        self.metrics = {}  # {player_id: {"max_speed": 0, "max_accel": 0, "total_distance_m": 0, "sprint_distance_m": 0}}

    def update_position(self, player_id, pos, frame_idx):
        """
        Update a player's position.
        pos: Point2D (meter-space coordinates) or (mx, my) tuple for backward compat.
        """
        if not isinstance(pos, Point2D):
            pos = Point2D(pos[0], pos[1])

        timestamp = frame_idx * self.dt
        if player_id not in self.trajectories:
            self.trajectories[player_id] = []
            self.metrics[player_id] = {
                "max_speed": 0,
                "max_accel": 0,
                "last_speed": 0,
                "total_distance_m": 0.0,
                "sprint_distance_m": 0.0,
            }

        self.trajectories[player_id].append((pos, timestamp))

        if len(self.trajectories[player_id]) >= 2:
            self._calculate_metrics(player_id)

    def _calculate_metrics(self, player_id):
        traj = self.trajectories[player_id]
        pos1, t1 = traj[-2]
        pos2, t2 = traj[-1]

        # Velocity v = ds / dt
        ds = pos1.distance_to(pos2)
        speed = ds / self.dt  # m/s

        # Acceleration a = dv / dt
        last_speed = self.metrics[player_id]["last_speed"]
        accel = abs(speed - last_speed) / self.dt  # m/s^2

        self.metrics[player_id]["total_distance_m"] += ds
        if speed >= self.sprint_speed_threshold:
            self.metrics[player_id]["sprint_distance_m"] += ds

        self.metrics[player_id]["max_speed"] = max(self.metrics[player_id]["max_speed"], speed)
        self.metrics[player_id]["max_accel"] = max(self.metrics[player_id]["max_accel"], accel)
        self.metrics[player_id]["last_speed"] = speed

    def get_player_stats(self, player_id):
        return self.metrics.get(player_id, {
            "max_speed": 0,
            "max_accel": 0,
            "total_distance_m": 0.0,
            "sprint_distance_m": 0.0,
        })

    def get_trajectory_lines(self, player_id):
        """Returns simplified movement points for the player as (x, y) tuples."""
        traj = self.trajectories.get(player_id, [])
        return [(pos.x, pos.y) for pos, _ in traj]


def calculate_shooting_splits(shots: list) -> dict:
    """
    Computes professional basketball shooting splits and efficiency:
    FG%, 2P%, 3P%, eFG% (Effective Field Goal %), TS% (True Shooting %),
    xPTS (Expected Points), and PAOE (Points Added Over Expected).
    """
    fgm = 0
    fga = len(shots)
    fg2m = 0
    fg2a = 0
    fg3m = 0
    fg3a = 0
    total_pts = 0
    total_expected_pts = 0.0

    for shot in shots:
        is_make = shot.get("status") == "made"
        zone = str(shot.get("zone", "")).lower()
        is_three = "three" in zone or shot.get("points_value") == 3

        sq = shot.get("shot_quality", 0.45)
        xpts = shot.get("expected_points", (3.0 if is_three else 2.0) * sq)
        total_expected_pts += xpts

        if is_three:
            fg3a += 1
            if is_make:
                fg3m += 1
                fgm += 1
                total_pts += 3
        else:
            fg2a += 1
            if is_make:
                fg2m += 1
                fgm += 1
                total_pts += 2

    fg_pct = round(fgm / fga, 3) if fga > 0 else 0.0
    fg2_pct = round(fg2m / fg2a, 3) if fg2a > 0 else 0.0
    fg3_pct = round(fg3m / fg3a, 3) if fg3a > 0 else 0.0
    efg_pct = round((fgm + 0.5 * fg3m) / fga, 3) if fga > 0 else 0.0
    # TS% estimation (standard formula: PTS / (2 * (FGA + 0.44 * FTA)))
    ts_pct = round(total_pts / (2.0 * fga), 3) if fga > 0 else 0.0
    paoe = round(total_pts - total_expected_pts, 2)

    return {
        "fga": fga,
        "fgm": fgm,
        "fg_pct": fg_pct,
        "fg2a": fg2a,
        "fg2m": fg2m,
        "fg2_pct": fg2_pct,
        "fg3a": fg3a,
        "fg3m": fg3m,
        "fg3_pct": fg3_pct,
        "efg_pct": efg_pct,
        "ts_pct": ts_pct,
        "points": total_pts,
        "expected_points": round(total_expected_pts, 2),
        "paoe": paoe,
    }


def calculate_team_ratings(points_scored: int, points_allowed: int, possessions: int, minutes: float = 40.0) -> dict:
    """
    Computes advanced pace and efficiency ratings:
    Offensive Rating (ORtg), Defensive Rating (DRtg), Net Rating, and Pace.
    """
    if possessions <= 0:
        return {"ortg": 0.0, "drtg": 0.0, "net_rtg": 0.0, "pace": 0.0}

    ortg = round((points_scored / possessions) * 100.0, 1)
    drtg = round((points_allowed / possessions) * 100.0, 1)
    net_rtg = round(ortg - drtg, 1)
    pace = round((possessions / minutes) * 40.0, 1) if minutes > 0 else 0.0

    return {
        "possessions": possessions,
        "ortg": ortg,
        "drtg": drtg,
        "net_rtg": net_rtg,
        "pace": pace,
    }


def get_game_analytics_payload(game_state) -> dict:
    """Computes real-time advanced basketball analytics from active GameState."""
    if game_state is None:
        return {
            "status": "no_game",
            "shooting_splits": {"fga": 0, "fgm": 0, "fg_pct": 0.0, "efg_pct": 0.0, "ts_pct": 0.0, "paoe": 0.0},
            "team_ratings": {
                "home": {"ortg": 0.0, "drtg": 0.0, "net_rtg": 0.0, "pace": 0.0},
                "away": {"ortg": 0.0, "drtg": 0.0, "net_rtg": 0.0, "pace": 0.0}
            }
        }

    with game_state._lock:
        events = list(game_state.events)
        home_score = game_state.score_home
        away_score = game_state.score_away

    shots = []
    for evt in events:
        if evt.event_type in ("SHOT_MAKE", "SHOT_MISS"):
            shots.append({
                "status": "made" if evt.event_type == "SHOT_MAKE" else "missed",
                "shooter_id": evt.data.get("shooter_id"),
                "zone": evt.data.get("zone", "paint"),
                "points_value": evt.data.get("points_value", 2),
                "shot_quality": evt.data.get("shot_quality", 0.45),
                "expected_points": evt.data.get("expected_points", 0.9),
            })

    splits = calculate_shooting_splits(shots)
    possessions = max(1, (home_score + away_score) // 2)

    ratings_home = calculate_team_ratings(home_score, away_score, possessions)
    ratings_away = calculate_team_ratings(away_score, home_score, possessions)

    return {
        "status": "success",
        "shooting_splits": splits,
        "team_ratings": {
            "home": ratings_home,
            "away": ratings_away,
        },
        "score": {"home": home_score, "away": away_score},
        "events_count": len(events),
        "total_shots": len(shots),
    }


