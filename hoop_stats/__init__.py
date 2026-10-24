from .analyzer import BasketballAnalyzer
from .detector import HoopDetector
from .geometry import CourtGeometry
from .metrics import PlayerMetrics
from .ocr import JerseyOCR
from .exporter import DataExporter
from .team_colors import get_dominant_jersey_color, assign_teams_by_color
from .shot_tracker import ShotTracker
from .rim_detector import RimDetector
from .player_registry import PlayerRegistry, PlayerStatSheet
from .event_detector import EventDetector
from .primitives import (
    Point2D, Ball, Rim, Backboard, PlayerCapsule, CourtLine,
    bbox_to_ball, bbox_to_player, bbox_to_rim, bbox_to_backboard,
    ball_distance, rim_contains_ball, capsule_distance, ball_near_player,
    Point3D, Sphere3D, Cylinder3D, Ring3D, Rect3D,
)
from .camera3d import Camera3D, make_virtual_camera

from .referee_detector import RefereeDetector, RefereeClassification
