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
        
        # Check specific zones from config if loaded
        order = ["paint", "mid_range", "three_point"]
        
        for z_name in order:
            if z_name in self.zones:
                # pointPolygonTest returns +1 if inside, 0 if on edge, -1 if outside
                dist = cv2.pointPolygonTest(self.zones[z_name], pt, measureDist=False)
                if dist >= 0:
                    return z_name

        if self.zones:
            return "out_of_bounds"

        # Fallback to FIBA standard geometry if zones not explicitly in config
        if self.is_out_of_bounds(mx, my):
            return "out_of_bounds"
        if self.is_in_paint(mx, my):
            return "paint"
        if self.is_three_pointer(mx, my, basket_idx=1) or self.is_three_pointer(mx, my, basket_idx=0):
            return "three_point"
        return "mid_range"

    # ── Official FIBA Court Floor Contact & Rule Geometry ──────────────────────────

    def is_out_of_bounds(self, mx: float, my: float, line_thickness_m: float = 0.05) -> bool:
        """
        FIBA Art. 23: The boundary lines belong to OUT OF BOUNDS.
        Any player floor contact on or beyond the 28.0m x 15.0m perimeter is out-of-bounds.
        """
        return (mx <= line_thickness_m or
                mx >= (self.court_width - line_thickness_m) or
                my <= line_thickness_m or
                my >= (self.court_height - line_thickness_m))

    def is_in_paint(self, mx: float, my: float, basket_idx: int = None) -> bool:
        """
        FIBA Art. 2.4.3: Restricted Area (Key / Paint).
        Standard dimensions: 5.80m length x 4.90m width (centered at y=7.5m, y in [5.05, 9.95]).
        Bounding lines belong to the paint.
        """
        in_y = (5.05 <= my <= 9.95)
        in_left = (0.0 <= mx <= 5.80) and in_y
        in_right = ((self.court_width - 5.80) <= mx <= self.court_width) and in_y

        if basket_idx == 0:
            return in_left
        elif basket_idx == 1:
            return in_right
        return in_left or in_right

    def is_three_pointer(self, mx: float, my: float, basket_idx: int = 1) -> bool:
        """
        FIBA Art. 16.2.3 & Art. 2.4.4: 3-Point Field Goal Area.
        - Radius: 6.75m from basket center projection on floor.
        - Corner lines: parallel to sidelines, 0.90m from sideline (6.60m from basket).
        - The 3-point line itself is PART OF THE 2-POINT ZONE (dist <= 6.75m is a 2-pointer).
        - Status is determined by where the shooter had floor contact prior to takeoff!
        """
        # Basket floor projections (1.575m from baseline, y=7.5m)
        rx = 26.425 if basket_idx == 1 else 1.575
        ry = self.court_height / 2.0  # 7.5m

        # Corner three territory: sidelines at 0.90m and 14.10m
        is_corner_region = (my <= 0.90 or my >= (self.court_height - 0.90))

        if basket_idx == 1:  # Attacking right rim
            if is_corner_region:
                # Straight corner 3 line runs from baseline (x=28.0) to arc junction (x=25.01)
                return bool(mx < 25.01)
            else:
                dist = np.hypot(mx - rx, my - ry)
                return bool(dist > 6.75)
        else:  # Attacking left rim
            if is_corner_region:
                # Straight corner 3 line runs from baseline (x=0.0) to arc junction (x=2.99)
                return bool(mx > 2.99)
            else:
                dist = np.hypot(mx - rx, my - ry)
                return bool(dist > 6.75)

    def is_on_centerline(self, mx: float, my: float, tolerance_m: float = 0.20) -> bool:
        """
        FIBA Art. 2.2: Centerline (Mittellinie) is located at x=14.0m.
        The centerline BELONGS TO THE BACKCOURT!
        """
        center_x = self.court_width / 2.0  # 14.0m
        return abs(mx - center_x) <= tolerance_m

    def is_backcourt(self, mx: float, my: float, attacking_basket: int = 1) -> bool:
        """
        FIBA Art. 30: Returns True if the point is in the team's backcourt.
        Note: The centerline (x=14.0m) belongs to the backcourt!
        """
        center_x = self.court_width / 2.0
        if attacking_basket == 1:
            return mx <= center_x
        else:
            return mx >= center_x

    def is_frontcourt(self, mx: float, my: float, attacking_basket: int = 1) -> bool:
        """
        FIBA Art. 30: Returns True if the point is strictly in the frontcourt.
        """
        center_x = self.court_width / 2.0
        if attacking_basket == 1:
            return mx > center_x
        else:
            return mx < center_x

    def evaluate_player_floor_status(self, player_capsule, attacking_basket: int = 1) -> dict:
        """
        Evaluates physical floor contact points for a player according to FIBA rules:
        - Central ground contact
        - Left foot ground contact
        - Right foot ground contact
        """
        # Central contact
        c_pt = player_capsule.ground_contact
        c_m = self.pixel_to_meter(c_pt.x, c_pt.y)

        # Feet contacts
        fl_pt = player_capsule.foot_left
        fr_pt = player_capsule.foot_right
        fl_m = self.pixel_to_meter(fl_pt.x, fl_pt.y)
        fr_m = self.pixel_to_meter(fr_pt.x, fr_pt.y)

        # 1. Paint status (FIBA Art. 26):
        # In the key if EVEN ONE foot touches the paint/boundary lines.
        # Fully cleared only if BOTH feet are outside.
        in_paint = (self.is_in_paint(fl_m[0], fl_m[1], attacking_basket) or
                    self.is_in_paint(fr_m[0], fr_m[1], attacking_basket) or
                    self.is_in_paint(c_m[0], c_m[1], attacking_basket))

        paint_cleared = (not self.is_in_paint(fl_m[0], fl_m[1], attacking_basket) and
                         not self.is_in_paint(fr_m[0], fr_m[1], attacking_basket) and
                         not self.is_in_paint(c_m[0], c_m[1], attacking_basket))

        # 2. 3-Point status (FIBA Art. 16.2.3):
        # 3-pointer ONLY if BOTH feet are outside the 3-point line.
        # Touching the line with even one foot = 2-pointer!
        is_three = (self.is_three_pointer(fl_m[0], fl_m[1], attacking_basket) and
                    self.is_three_pointer(fr_m[0], fr_m[1], attacking_basket))

        # 3. Out of Bounds status (FIBA Art. 23):
        # Out if ANY part of foot touches boundary line or beyond.
        is_oob = (self.is_out_of_bounds(fl_m[0], fl_m[1]) or
                  self.is_out_of_bounds(fr_m[0], fr_m[1]) or
                  self.is_out_of_bounds(c_m[0], c_m[1]))

        # 4. Frontcourt / Backcourt status (FIBA Art. 30):
        # Frontcourt established only when BOTH feet are in frontcourt.
        frontcourt_established = (self.is_frontcourt(fl_m[0], fl_m[1], attacking_basket) and
                                  self.is_frontcourt(fr_m[0], fr_m[1], attacking_basket))

        # Backcourt contact occurs if either foot touches backcourt or centerline
        touches_backcourt = (self.is_backcourt(fl_m[0], fl_m[1], attacking_basket) or
                             self.is_backcourt(fr_m[0], fr_m[1], attacking_basket))

        touches_centerline = (self.is_on_centerline(fl_m[0], fl_m[1]) or
                              self.is_on_centerline(fr_m[0], fr_m[1]))

        return {
            "center_m": c_m,
            "foot_left_m": fl_m,
            "foot_right_m": fr_m,
            "in_paint": in_paint,
            "paint_cleared": paint_cleared,
            "is_three": is_three,
            "is_out_of_bounds": is_oob,
            "frontcourt_established": frontcourt_established,
            "touches_backcourt": touches_backcourt,
            "touches_centerline": touches_centerline,
        }
