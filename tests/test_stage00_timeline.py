from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_timestamp_sampling_keeps_explicit_anchor_as_frame0() -> None:
    module = load_script("rigid_stage00_prepare.py")
    timestamps = [index / 72.0 for index in range(588)]

    selected = module.choose_source_indices(
        timestamps,
        15.0,
        minimum_timestamp=0.166677777777778,
        maximum_timestamp=7.972255555555556,
    )

    assert selected[0] == 12
    assert len(selected) == 118
    assert all(index >= 12 for index in selected)


def test_sampling_does_not_round_to_a_pre_anchor_frame() -> None:
    module = load_script("rigid_stage00_prepare.py")

    selected = module.choose_source_indices(
        [0.0, 0.1, 0.2, 0.3],
        5.0,
        minimum_timestamp=0.11,
        maximum_timestamp=0.3,
    )

    assert selected == [2]


def test_mixed_export_bounds_use_first_and_last_rgb_rows(tmp_path: Path) -> None:
    module = load_script("mixed_stage00_prepare.py")
    frames_csv = tmp_path / "frames.csv"
    with frames_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rgb_timestamp_s"])
        writer.writeheader()
        writer.writerows(
            [
                {"rgb_timestamp_s": "0.16667777777777779"},
                {"rgb_timestamp_s": "0.3610333333333334"},
                {"rgb_timestamp_s": "7.972255555555556"},
            ]
        )

    assert module.exported_rgb_bounds(frames_csv) == (
        0.16667777777777779,
        7.972255555555556,
    )
