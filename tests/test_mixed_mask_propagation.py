from __future__ import annotations

from scripts.mixed_stage02_propagate_masks import per_object_qc


def _summary(areas: list[int]) -> dict:
    return {
        "object_ids": ["cup"],
        "frames": [
            {"objects": {"cup": {"area_pixels": area}}}
            for area in areas
        ],
    }


def test_per_object_qc_allows_terminal_occlusion() -> None:
    qc = per_object_qc(_summary([100, 90, 20, 0, 0]), 4.5, True)["objects"]["cup"]

    assert qc["passed"] is True
    assert qc["visibility_end_frame"] == 2
    assert qc["terminal_empty_frames"] == [3, 4]
    assert qc["nonterminal_empty_frames"] == []


def test_per_object_qc_rejects_intermediate_empty_mask() -> None:
    qc = per_object_qc(_summary([100, 0, 80, 70]), 4.5, True)["objects"]["cup"]

    assert qc["passed"] is False
    assert qc["terminal_empty_frames"] == []
    assert qc["nonterminal_empty_frames"] == [1]


def test_per_object_qc_can_require_nonempty_full_timeline() -> None:
    qc = per_object_qc(_summary([100, 80, 0]), 4.5, False)["objects"]["cup"]

    assert qc["passed"] is False
