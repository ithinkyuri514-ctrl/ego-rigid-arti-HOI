from __future__ import annotations

import numpy as np
import pytest

from vlm_sam2_recon.rigid_pipeline.sam2_video import add_mask


class _FakeTensor:
    def __init__(self, value: np.ndarray):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakePredictor:
    def __init__(self):
        self.call = None

    def add_new_mask(self, **kwargs):
        self.call = kwargs
        logits = np.where(kwargs["mask"], 2.0, -2.0)[None, None]
        return kwargs["frame_idx"], [kwargs["obj_id"]], _FakeTensor(logits)


def test_add_mask_conditions_video_predictor() -> None:
    predictor = _FakePredictor()
    mask = np.asarray([[False, True], [True, False]])

    result = add_mask(
        predictor,
        {"state": "opaque"},
        frame_index=3,
        object_id="cup",
        mask=mask,
    )

    assert predictor.call["frame_idx"] == 3
    assert predictor.call["obj_id"] == "cup"
    np.testing.assert_array_equal(predictor.call["mask"], mask)
    np.testing.assert_array_equal(result["cup"], mask)


@pytest.mark.parametrize("mask", [np.zeros((2, 2), dtype=bool), np.zeros((1, 2, 2), dtype=bool)])
def test_add_mask_rejects_invalid_conditioning_masks(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        add_mask(_FakePredictor(), {}, frame_index=0, object_id="cup", mask=mask)
