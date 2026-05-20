"""
primitives.py
=============
Lightweight geometric primitives for representing basketball objects.

2D primitives (pixel/meter space):
  - Ball        → circle  (center + radius)
  - Rim         → circle  (center + radius)
  - Backboard   → rectangle (center + width × height)
  - Player      → capsule (head + foot + width)
  - Court lines → line segments (start + end)

3D primitives (camera coordinate system, Y=up, meters):
  - Sphere3D    → ball
  - Cylinder3D  → player
  - Ring3D      → rim
  - Rect3D      → backboard
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─── Core types ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Point2D:
    """A 2-D point in pixel or meter space."""
    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(slots=True)
class Ball:
    """Basketball reduced to a center point and radius."""
    center: Point2D
    radius: float

    @property
    def diameter(self) -> float:
        return self.radius * 2


@dataclass(slots=True)
class Rim:
    """Basket hoop reduced to a 2-D circle."""
    center: Point2D
    radius: float

    @property
    def diameter(self) -> float:
        return self.radius * 2


@dataclass(slots=True)
class Backboard:
    """Backboard reduced to a 2-D rectangle."""
    center: Point2D
    width: float
    height: float

    @property
    def x1(self) -> float:
        return self.center.x - self.width / 2

    @property
    def y1(self) -> float:
        return self.center.y - self.height / 2

    @property
    def x2(self) -> float:
        return self.center.x + self.width / 2

    @property
    def y2(self) -> float:
        return self.center.y + self.height / 2


@dataclass(slots=True)
class PlayerCapsule:
    """
    Player reduced to a vertical capsule (tube):
    a line segment from foot to head with a shoulder-width diameter.
    """
    head: Point2D
    foot: Point2D
    width: float
    # Optional ID and meter-space position (filled after court transform)
    obj_id: Optional[int] = field(default=None, repr=False)
    meter_pos: Optional[Point2D] = field(default=None, repr=False)

    @property
    def height(self) -> float:
        return self.head.distance_to(self.foot)

    @property
    def center(self) -> Point2D:
        return Point2D(
            (self.head.x + self.foot.x) / 2,
            (self.head.y + self.foot.y) / 2,
        )

    @property
    def ground_contact(self) -> Point2D:
        """The foot point — used for court-coordinate transforms."""
        return self.foot

    @property
    def pixel_x(self) -> float:
        """Horizontal center (same as bbox center-X)."""
        return (self.head.x + self.foot.x) / 2


@dataclass(slots=True)
class CourtLine:
    """A single court marking line segment."""
    start: Point2D
    end: Point2D

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)


# ─── Converters: bbox → primitive ────────────────────────────────────────────

def bbox_to_ball(x1: float, y1: float, x2: float, y2: float) -> Ball:
    """Convert a YOLO bounding box [x1, y1, x2, y2] to a Ball primitive."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # Use average of width/height as diameter (ball is roughly circular)
    radius = ((x2 - x1) + (y2 - y1)) / 4
    return Ball(center=Point2D(cx, cy), radius=radius)


def bbox_to_player(x1: float, y1: float, x2: float, y2: float,
                   obj_id: int | None = None) -> PlayerCapsule:
    """
    Convert a YOLO bounding box [x1, y1, x2, y2] to a PlayerCapsule.

    Head = top-center of bbox, Foot = bottom-center, Width = bbox width.
    """
    cx = (x1 + x2) / 2
    head = Point2D(cx, y1)
    foot = Point2D(cx, y2)
    width = x2 - x1
    return PlayerCapsule(head=head, foot=foot, width=width, obj_id=obj_id)


def bbox_to_rim(x1: float, y1: float, x2: float, y2: float) -> Rim:
    """Convert a bounding rectangle to a Rim circle (inscribed)."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    radius = min(x2 - x1, y2 - y1) / 2
    return Rim(center=Point2D(cx, cy), radius=radius)


def bbox_to_backboard(x1: float, y1: float, x2: float, y2: float) -> Backboard:
    """Convert a bounding rectangle to a Backboard."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return Backboard(center=Point2D(cx, cy), width=x2 - x1, height=y2 - y1)


# ─── Geometric helpers ───────────────────────────────────────────────────────

def ball_distance(ball: Ball, point: Point2D) -> float:
    """Euclidean distance from a ball's center to a point."""
    return ball.center.distance_to(point)


def rim_contains_ball(rim: Rim, ball: Ball) -> bool:
    """
    Returns True if the ball overlaps the rim circle
    (center-to-center distance < sum of radii).
    """
    dist = rim.center.distance_to(ball.center)
    return dist < (rim.radius + ball.radius)


def capsule_distance(a: PlayerCapsule, b: PlayerCapsule) -> float:
    """Approximate distance between two player capsules (center-to-center)."""
    return a.center.distance_to(b.center)


def ball_near_player(ball: Ball, player: PlayerCapsule,
                     threshold: float = 200.0) -> bool:
    """True if the ball center is within `threshold` pixels of the player center."""
    return ball.center.distance_to(player.center) < threshold


# ─── 3D Primitives (camera coordinate system, Y = up, meters) ───────────────

@dataclass(slots=True)
class Point3D:
    """A point in 3D world space. Y is up, X is right, Z is forward (into the scene)."""
    x: float
    y: float
    z: float

    def distance_to(self, other: Point3D) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def xz(self) -> Tuple[float, float]:
        """Return (x, z) ground-plane position."""
        return (self.x, self.z)


@dataclass(slots=True)
class Sphere3D:
    """Ball in 3D space — a sphere."""
    center: Point3D
    radius: float  # meters (~0.12 for a basketball)


@dataclass(slots=True)
class Cylinder3D:
    """Player in 3D space — a vertical cylinder from base up to height."""
    base: Point3D       # foot position on the ground plane (y ≈ 0)
    height: float       # meters (typically 1.7–2.1)
    radius: float       # shoulder half-width in meters (~0.25)
    obj_id: Optional[int] = field(default=None, repr=False)

    @property
    def top(self) -> Point3D:
        """Head position."""
        return Point3D(self.base.x, self.base.y + self.height, self.base.z)

    @property
    def center(self) -> Point3D:
        """Center of the cylinder."""
        return Point3D(self.base.x, self.base.y + self.height / 2, self.base.z)


@dataclass(slots=True)
class Ring3D:
    """Basket rim in 3D space — a horizontal ring at a fixed height."""
    center: Point3D     # center of the ring (y ≈ 3.05 for FIBA)
    radius: float       # meters (~0.23 for a standard rim)
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # ring faces upward


@dataclass(slots=True)
class Rect3D:
    """Backboard in 3D — a vertical rectangle."""
    center: Point3D     # center of the backboard
    width: float        # meters (1.80 FIBA)
    height: float       # meters (1.05 FIBA)
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # faces into the court

