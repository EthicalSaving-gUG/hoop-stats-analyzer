import pytest

from hoop_stats.grounding import (
    GroundingUnavailable,
    boxes_to_yolo_labels,
    build_prompt,
    create_grounder,
    parse_boxes,
    parse_points,
)


def test_parse_locateanything_boxes_and_points():
    answer = "<box><100><200><500><800></box> <box><250><750></box>"

    boxes = parse_boxes(answer, image_width=1280, image_height=720)
    points = parse_points(answer, image_width=1280, image_height=720)

    assert len(boxes) == 1
    assert boxes[0].as_dict() == {
        "x1": 128.0,
        "y1": 144.0,
        "x2": 640.0,
        "y2": 576.0,
        "label": None,
    }
    assert len(points) == 1
    assert points[0].as_dict() == {"x": 320.0, "y": 540.0, "label": None}


def test_build_prompt_for_categories():
    assert build_prompt("detect", "basketball rim, shot clock") == (
        "Locate all the instances that matches the following description: basketball rim</c>shot clock."
    )


def test_boxes_to_yolo_labels():
    rows = boxes_to_yolo_labels(
        [{"x1": 100, "y1": 50, "x2": 300, "y2": 250}],
        image_width=1000,
        image_height=500,
        class_id=2,
    )

    assert rows == ["2 0.200000 0.300000 0.200000 0.400000"]


def test_grounding_status_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("HOOP_STATS_GROUNDING_BACKEND", raising=False)
    from hoop_stats.grounding import grounding_status

    status = grounding_status()

    assert status["backend"] == "disabled"
    assert status["enabled"] is False


def test_disabled_grounding_backend_raises_clear_error(monkeypatch):
    monkeypatch.setenv("HOOP_STATS_GROUNDING_BACKEND", "disabled")

    with pytest.raises(GroundingUnavailable, match="HOOP_STATS_GROUNDING_BACKEND"):
        create_grounder()
