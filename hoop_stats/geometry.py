import numpy as np
import cv2

class CourtGeometry:
    """
    Handles the transformation between video pixel coordinates and real-world basketball court coordinates.
    Standard FIBA court dimensions: 28m x 15m.
    """
    def __init__(self, court_width=28.0, court_height=15.0):
        self.court_width = court_width
        self.court_height = court_height
        self.homography_matrix = None

    def set_homography(self, video_points, real_world_points):
        """
        Calculates the homography matrix.
        :param video_points: List of (x, y) pixels from video.
        :param real_world_points: List of (x, y) meters on court.
        """
        video_points = np.array(video_points, dtype=np.float32)
        real_world_points = np.array(real_world_points, dtype=np.float32)
        
        self.homography_matrix, _ = cv2.findHomography(video_points, real_world_points)

    def pixel_to_meter(self, x, y):
        """
        Transforms pixel (x, y) to real-world (Mx, My).
        """
        if self.homography_matrix is None:
            return x, y  # Fallback to pixels if no homography is set
            
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return transformed_point[0][0][0], transformed_point[0][0][1]

    def get_distance(self, p1, p2):
        """
        Calculates distance in meters between two transformed points.
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
