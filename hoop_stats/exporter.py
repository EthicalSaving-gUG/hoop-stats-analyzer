import json
import csv
import os

class DataExporter:
    def __init__(self, output_base_path):
        # Default to 'results' directory if no path is provided
        if not os.path.dirname(output_base_path):
            self.output_base_path = os.path.join("results", output_base_path)
        else:
            self.output_base_path = output_base_path
            
        # Ensure the output directory exists
        out_dir = os.path.dirname(self.output_base_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def export_json(self, data):
        path = f"{self.output_base_path}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"JSON saved → {path}")

    def export_csv_players(self, player_data, scores=None):
        path = f"{self.output_base_path}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "PlayerID", "Team", "Jersey Color", "Jersey Number",
                "Max Speed (m/s)", "Max Accel (m/s²)", "Trajectory Points"
            ])
            for pid, d in player_data.items():
                writer.writerow([
                    pid,
                    d.get("team", "Unknown"),
                    d.get("jersey_color", "Unknown"),
                    d.get("jersey", "Unknown"),
                    d.get("max_speed_ms", 0),
                    d.get("max_accel_ms2", 0),
                    d.get("trajectory_pts", 0),
                ])

            if scores:
                writer.writerow([])
                writer.writerow(["=== SCORING ==="])
                for k, v in scores.items():
                    writer.writerow([k, v])
        print(f"CSV saved  → {path}")
