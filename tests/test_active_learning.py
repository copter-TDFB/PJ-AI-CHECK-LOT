"""Unit tests for the pure prelabel bbox filter (no YOLO/torch needed)."""

from services.active_learning import filter_prelabel_bboxes


def test_filter_keeps_only_matching_prefix():
    boxes = [
        (0, 0, 10, 10, "box_lot"),
        (0, 0, 10, 10, "sachet_lot"),
    ]
    out = filter_prelabel_bboxes(boxes, class_prefixes=["box_"])
    assert len(out) == 1
    assert out[0]["label"] == "prelabel"
    assert out[0]["x2"] == 10.0


def test_filter_drops_zero_or_negative_area_boxes():
    boxes = [(5, 5, 5, 9, "box_lot"), (5, 5, 9, 5, "box_lot")]
    assert filter_prelabel_bboxes(boxes, class_prefixes=["box_"]) == []


def test_filter_no_prefix_keeps_all_positive_area():
    boxes = [(0, 0, 2, 2, "anything"), (0, 0, 1, 1, "other")]
    out = filter_prelabel_bboxes(boxes, class_prefixes=None)
    assert len(out) == 2
