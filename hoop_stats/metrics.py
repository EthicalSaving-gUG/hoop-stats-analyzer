from .primitives import Point2D


class PlayerMetrics:
    def __init__(self, fps=30):
        self.fps = fps
        self.dt = 1.0 / fps
        self.trajectories = {}  # {player_id: [(Point2D, timestamp)]}
        self.metrics = {}  # {player_id: {"max_speed": 0, "max_accel": 0}}

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
            self.metrics[player_id] = {"max_speed": 0, "max_accel": 0, "last_speed": 0}

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

        self.metrics[player_id]["max_speed"] = max(self.metrics[player_id]["max_speed"], speed)
        self.metrics[player_id]["max_accel"] = max(self.metrics[player_id]["max_accel"], accel)
        self.metrics[player_id]["last_speed"] = speed

    def get_player_stats(self, player_id):
        return self.metrics.get(player_id, {"max_speed": 0, "max_accel": 0})

    def get_trajectory_lines(self, player_id):
        """Returns simplified movement points for the player as (x, y) tuples."""
        traj = self.trajectories.get(player_id, [])
        return [(pos.x, pos.y) for pos, _ in traj]
