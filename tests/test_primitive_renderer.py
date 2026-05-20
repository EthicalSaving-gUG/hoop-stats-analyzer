"""test_primitive_renderer.py — unit tests for hoop_stats.primitive_renderer"""
import numpy as np
import pytest
from hoop_stats.primitive_renderer import PrimitiveRenderer, COLORS
from hoop_stats.primitives import Ball, Rim, PlayerCapsule, Point2D


@pytest.fixture
def renderer():
    return PrimitiveRenderer(640, 360, mode="wireframe")


@pytest.fixture
def overlay_renderer():
    return PrimitiveRenderer(640, 360, mode="overlay")


def _make_ball(x=320, y=180, r=10):
    return Ball(center=Point2D(x, y), radius=r)


def _make_capsule(px=300, py_head=100, py_foot=250, width=40, obj_id=1):
    return PlayerCapsule(
        head=Point2D(px, py_head),
        foot=Point2D(px, py_foot),
        width=width,
        obj_id=obj_id,
    )


def _make_rim(x=100, y=180, r=30):
    return Rim(center=Point2D(x, y), radius=r)


class TestRendererInit:
    def test_wireframe_mode(self, renderer):
        assert renderer.mode == "wireframe"
        assert renderer.w == 640
        assert renderer.h == 360
        assert len(renderer.court_lines) > 0

    def test_overlay_mode(self, overlay_renderer):
        assert overlay_renderer.mode == "overlay"


class TestDrawBall:
    def test_modifies_frame(self, renderer):
        canvas = np.full((360, 640, 3), COLORS["bg"], dtype=np.uint8)
        original_sum = canvas.sum()
        renderer._draw_ball(canvas, _make_ball())
        assert canvas.sum() != original_sum, "draw_ball should modify the canvas"

    def test_draws_near_center(self, renderer):
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        ball = _make_ball(320, 180, 12)
        renderer._draw_ball(canvas, ball)
        # Check that pixels near the ball center are non-zero
        region = canvas[170:190, 310:330]
        assert region.sum() > 0


class TestDrawPlayer:
    def test_modifies_frame(self, renderer):
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        cap = _make_capsule()
        renderer._draw_player(canvas, cap, team_idx=0)
        assert canvas.sum() > 0

    def test_different_teams_different_colors(self, renderer):
        canvas_a = np.zeros((360, 640, 3), dtype=np.uint8)
        canvas_b = np.zeros((360, 640, 3), dtype=np.uint8)
        cap = _make_capsule()
        renderer._draw_player(canvas_a, cap, team_idx=0)
        renderer._draw_player(canvas_b, cap, team_idx=1)
        # They should look different (different team colors)
        assert not np.array_equal(canvas_a, canvas_b)


class TestDrawRim:
    def test_modifies_frame(self, renderer):
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        rim = _make_rim()
        renderer._draw_rim(canvas, rim)
        assert canvas.sum() > 0


class TestDrawHUD:
    def test_draws_score(self, renderer):
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        renderer._draw_hud(canvas, {"left_basket_pts": 10, "right_basket_pts": 8}, 900, 30.0)
        # Top bar should have non-zero pixels
        top_region = canvas[0:48, :]
        assert top_region.sum() > 0


class TestRenderFrame:
    def test_wireframe_produces_frame(self, renderer):
        balls = [_make_ball()]
        players = {1: _make_capsule()}
        rims = [_make_rim(100, 180), _make_rim(540, 180)]
        scores = {"left_basket_pts": 0, "right_basket_pts": 0}

        frame = renderer.render_frame(
            original_frame=None,
            balls=balls,
            player_capsules=players,
            rims=rims,
            scores=scores,
        )
        assert frame.shape == (360, 640, 3)
        assert frame.sum() > 0

    def test_overlay_uses_original_frame(self, overlay_renderer):
        original = np.full((720, 1280, 3), 128, dtype=np.uint8)
        balls = [_make_ball()]
        players = {1: _make_capsule()}
        rims = [_make_rim(100, 180), _make_rim(540, 180)]
        scores = {"left_basket_pts": 4, "right_basket_pts": 2}

        frame = overlay_renderer.render_frame(
            original_frame=original,
            balls=balls,
            player_capsules=players,
            rims=rims,
            scores=scores,
        )
        assert frame.shape == (360, 640, 3)
        # Should not be all black — original was dimmed gray + primitives drawn
        assert frame.sum() > 0

    def test_wireframe_differs_from_overlay(self):
        renderer_w = PrimitiveRenderer(640, 360, mode="wireframe")
        renderer_o = PrimitiveRenderer(640, 360, mode="overlay")
        original = np.full((720, 1280, 3), 180, dtype=np.uint8)
        balls = [_make_ball()]
        players = {1: _make_capsule()}
        rims = [_make_rim(100, 180)]
        scores = {"left_basket_pts": 0, "right_basket_pts": 0}

        frame_w = renderer_w.render_frame(original, balls, players, rims, scores)
        frame_o = renderer_o.render_frame(original, balls, players, rims, scores)
        assert not np.array_equal(frame_w, frame_o)


class TestBallTrail:
    def test_trail_accumulates(self, renderer):
        for i in range(5):
            renderer.render_frame(
                None,
                [_make_ball(100 + i * 20, 180)],
                {},
                [],
                {"left_basket_pts": 0, "right_basket_pts": 0},
            )
        assert len(renderer.ball_trail) == 5

    def test_trail_max_length(self, renderer):
        for i in range(30):
            renderer.render_frame(
                None,
                [_make_ball(50 + i * 5, 180)],
                {},
                [],
                {"left_basket_pts": 0, "right_basket_pts": 0},
            )
        assert len(renderer.ball_trail) <= renderer.max_trail
