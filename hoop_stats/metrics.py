import numpy as np

class PlayerMetrics:
    def __init__(self, fps=30):
        self.fps = fps
        self.dt = 1.0 / fps
        self.trajectories = {}  # {player_id: [(Mx, My, timestamp)]}
        self.metrics = {}  # {player_id: {"max_speed": 0, "max_accel": 0}}

    def update_position(self, player_id, mx, my, frame_idx):
        timestamp = frame_idx * self.dt
        if player_id not in self.trajectories:
            self.trajectories[player_id] = []
            self.metrics[player_id] = {"max_speed": 0, "max_accel": 0, "last_speed": 0}
            
        self.trajectories[player_id].append((mx, my, timestamp))
        
        if len(self.trajectories[player_id]) >= 2:
            self._calculate_metrics(player_id)

    def _calculate_metrics(self, player_id):
        traj = self.trajectories[player_id]
        p1, p2 = traj[-2], traj[-1]
        
        # Velocity v = ds / dt
        ds = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
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
        """Returns simplified movement lines for the player."""
        return [(float(p[0]), float(p[1])) for p in self.trajectories.get(player_id, [])]
