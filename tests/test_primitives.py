"""test_primitives.py — unit tests for hoop_stats.primitives"""
import math
import pytest
from hoop_stats.primitives import (
    Point2D, Ball, Rim, Backboard, PlayerCapsule, CourtLine,
    bbox_to_ball, bbox_to_player, bbox_to_rim, bbox_to_backboard,
    ball_distance, rim_contains_ball, capsule_distance, ball_near_player,
)


class TestPoint2D:
    def test_distance_to_self_is_zero(self):
        p = Point2D(3.0, 4.0)
        assert p.distance_to(p) == 0.0

    def test_distance_to_known(self):
        p1 = Point2D(0, 0)
        p2 = Point2D(3, 4)
        assert p1.distance_to(p2) == pytest.approx(5.0)

    def test_as_tuple(self):
        p = Point2D(1.5, 2.5)
        assert p.as_tuple() == (1.5, 2.5)


class TestBboxToBall:
    def test_center_and_radius(self):
        ball = bbox_to_ball(10, 20, 30, 40)
        assert ball.center.x == pytest.approx(20.0)
        assert ball.center.y == pytest.approx(30.0)
        # avg diameter = ((30-10) + (40-20)) / 2 = 20, radius = 10
        assert ball.radius == pytest.approx(10.0)

    def test_diameter_property(self):
        ball = bbox_to_ball(0, 0, 10, 10)
        assert ball.diameter == pytest.approx(10.0)


class TestBboxToPlayer:
    def test_head_foot_width(self):
        cap = bbox_to_player(100, 50, 200, 450)
        # Head = top-center = (150, 50)
        assert cap.head.x == pytest.approx(150.0)
        assert cap.head.y == pytest.approx(50.0)
        # Foot = bottom-center = (150, 450)
        assert cap.foot.x == pytest.approx(150.0)
        assert cap.foot.y == pytest.approx(450.0)
        # Width = 200 - 100 = 100
        assert cap.width == pytest.approx(100.0)

    def test_obj_id_attached(self):
        cap = bbox_to_player(0, 0, 50, 100, obj_id=7)
        assert cap.obj_id == 7

    def test_height_property(self):
        cap = bbox_to_player(0, 0, 50, 100)
        assert cap.height == pytest.approx(100.0)

    def test_center_property(self):
        cap = bbox_to_player(0, 0, 100, 200)
        assert cap.center.x == pytest.approx(50.0)
        assert cap.center.y == pytest.approx(100.0)

    def test_ground_contact(self):
        cap = bbox_to_player(10, 20, 60, 300)
        gc = cap.ground_contact
        assert gc.x == pytest.approx(35.0)
        assert gc.y == pytest.approx(300.0)


class TestBboxToRim:
    def test_center_and_radius(self):
        rim = bbox_to_rim(100, 200, 200, 300)
        assert rim.center.x == pytest.approx(150.0)
        assert rim.center.y == pytest.approx(250.0)
        # min dimension / 2 = 50
        assert rim.radius == pytest.approx(50.0)


class TestBboxToBackboard:
    def test_properties(self):
        bb = bbox_to_backboard(50, 100, 150, 200)
        assert bb.center.x == pytest.approx(100.0)
        assert bb.center.y == pytest.approx(150.0)
        assert bb.width == pytest.approx(100.0)
        assert bb.height == pytest.approx(100.0)
        assert bb.x1 == pytest.approx(50.0)
        assert bb.y1 == pytest.approx(100.0)
        assert bb.x2 == pytest.approx(150.0)
        assert bb.y2 == pytest.approx(200.0)


class TestRimContainsBall:
    def test_overlapping(self):
        rim = Rim(center=Point2D(100, 100), radius=30)
        ball = Ball(center=Point2D(110, 100), radius=10)
        assert rim_contains_ball(rim, ball) is True

    def test_not_overlapping(self):
        rim = Rim(center=Point2D(100, 100), radius=30)
        ball = Ball(center=Point2D(300, 300), radius=10)
        assert rim_contains_ball(rim, ball) is False

    def test_edge_case_touching(self):
        rim = Rim(center=Point2D(0, 0), radius=10)
        # Ball center at distance exactly == sum of radii → NOT contained (< not <=)
        ball = Ball(center=Point2D(15, 0), radius=5)
        assert rim_contains_ball(rim, ball) is False


class TestBallDistance:
    def test_known_distance(self):
        ball = Ball(center=Point2D(0, 0), radius=5)
        point = Point2D(3, 4)
        assert ball_distance(ball, point) == pytest.approx(5.0)


class TestCapsuleDistance:
    def test_distance_between_two_players(self):
        a = bbox_to_player(0, 0, 50, 200)
        b = bbox_to_player(300, 0, 350, 200)
        # Centers: (25, 100) and (325, 100) → distance = 300
        assert capsule_distance(a, b) == pytest.approx(300.0)


class TestBallNearPlayer:
    def test_near(self):
        ball = Ball(center=Point2D(100, 100), radius=5)
        player = bbox_to_player(80, 50, 120, 150)
        # Player center is at (100, 100), ball center at (100, 100) → distance 0
        assert ball_near_player(ball, player, threshold=200) is True

    def test_far(self):
        ball = Ball(center=Point2D(1000, 1000), radius=5)
        player = bbox_to_player(0, 0, 50, 200)
        assert ball_near_player(ball, player, threshold=200) is False
