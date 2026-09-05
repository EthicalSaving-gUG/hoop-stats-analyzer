import numpy as np
import cv2
import json
import os

class CourtGeometry:
    """
    Handles the transformation between video pixel coordinates and real-world basketball court coordinates.
    Standard FIBA court dimensions: 28m x 15m.
    """
    def __init__(self, court_width=28.0, court_height=15.0, config_path=None):
        self.court_width = court_width
        self.court_height = court_height
        self.homography_matrix = None
        self.zones = {}
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "court_config.json")
            
        self.load_config(config_path)

    def load_config(self, config_path):
        """Loads physical court zones from JSON and converts them to OpenCV contours."""
        if not os.path.exists(config_path):
            print(f"Warning: Court config not found at {config_path}")
            return
            
        with open(config_path, "r") as f:
            cfg = json.load(f)
            
        for zone_name, data in cfg.get("zones", {}).items():
            if data["type"] == "rectangle":
                # Convert rect to 4-point polygon
                p1, p2 = data["p1"], data["p2"]
                pts = [
                    [p1[0], p1[1]],
                    [p2[0], p1[1]],
                    [p2[0], p2[1]],
                    [p1[0], p2[1]]
                ]
            else:
                pts = data["points"]
                
            # Convert to numpy array shape (N, 1, 2) required by cv2.pointPolygonTest
            contour = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
            self.zones[zone_name] = contour

    def set_homography(self, video_points, real_world_points):
        """
        Calculates the homography matrix and its inverse.
        :param video_points: List of (x, y) pixels from video.
        :param real_world_points: List of (x, y) meters on court.
        """
        video_points = np.array(video_points, dtype=np.float32)
        real_world_points = np.array(real_world_points, dtype=np.float32)
        
        self.homography_matrix, _ = cv2.findHomography(video_points, real_world_points)
        if self.homography_matrix is not None:
            try:
                self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
            except np.linalg.LinAlgError:
                self.inv_homography_matrix = None
        else:
            self.inv_homography_matrix = None

    def pixel_to_meter(self, x, y):
        """
        Transforms pixel (x, y) to real-world (Mx, My).
        """
        if self.homography_matrix is None:
            return x, y  # Fallback to pixels if no homography is set
            
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return transformed_point[0][0][0], transformed_point[0][0][1]

    def meter_to_pixel(self, mx, my):
        """
        Transforms real-world court coordinate (Mx, My) back to camera pixel (x, y).
        """
        if self.inv_homography_matrix is None:
            if self.homography_matrix is not None:
                try:
                    self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
                except np.linalg.LinAlgError:
                    return mx, my
            else:
                return mx, my
                
        point = np.array([[[mx, my]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, self.inv_homography_matrix)
        return float(transformed_point[0][0][0]), float(transformed_point[0][0][1])

    def get_distance(self, p1, p2):
        """
        Calculates distance in meters between two transformed points.
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_zone(self, mx, my):
        """
        Given a real-world coordinate (mx, my) in meters, returns the name of the shot zone.
        Returns "out_of_bounds" if outside all defined zones.
        """
        pt = (float(mx), float(my))
        
        # Check specific zones; order matters if zones overlap. 
        # Usually checking smaller zones first (like paint) is better, but here we assume disjoint or we just return the first match.
        # So let's check paint, then mid_range, then three_point.
        order = ["paint", "mid_range", "three_point"]
        
        for z_name in order:
            if z_name in self.zones:
                # pointPolygonTest returns +1 if inside, 0 if on edge, -1 if outside
                dist = cv2.pointPolygonTest(self.zones[z_name], pt, measureDist=False)
                if dist >= 0:
                    return z_name
                    
        return "out_of_bounds"
