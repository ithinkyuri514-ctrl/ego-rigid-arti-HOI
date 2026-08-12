import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.contact_driven_screen import (  # noqa: E402
    resolve_vlm_contact_frame,
    select_contact_fingertips,
)
from vlm_sam2_recon.stages.vlm_contact_semantics import (  # noqa: E402
    aggregate_contact_windows,
    make_windows,
    match_depth_frame,
    normalize_fingers,
)


class VlmContactSemanticsTest(unittest.TestCase):
    def test_five_finger_normalization_and_mano_filter(self):
        self.assertEqual(
            normalize_fingers(["index_tip", "thumb", "little_finger", "invalid"]),
            ["thumb", "index", "pinky"],
        )
        tips = np.arange(15, dtype=np.float64).reshape(5, 3)
        ids = [4, 8, 12, 16, 20]
        names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        selected, selected_ids, selected_names = select_contact_fingertips(
            tips, ids, names, ("thumb", "index")
        )
        np.testing.assert_array_equal(selected, tips[:2])
        self.assertEqual(selected_ids, [4, 8])
        self.assertEqual(selected_names, ["thumb_tip", "index_tip"])

    def test_overlapping_window_vote_produces_first_contact_and_finger(self):
        votes = [
            [
                {"frame": 19, "right_contact": False, "left_contact": False, "right_fingers": [], "left_fingers": [], "contacted_part": "unknown", "confidence": 0.9},
                {"frame": 20, "right_contact": True, "left_contact": False, "right_fingers": ["index"], "left_fingers": [], "contacted_part": "screen", "confidence": 0.9},
            ],
            [
                {"frame": 20, "right_contact": True, "left_contact": False, "right_fingers": ["index"], "left_fingers": [], "contacted_part": "lid_edge", "confidence": 0.8},
                {"frame": 21, "right_contact": True, "left_contact": False, "right_fingers": ["index", "thumb"], "left_fingers": [], "contacted_part": "screen", "confidence": 0.9},
            ],
            [
                {"frame": 20, "right_contact": False, "left_contact": False, "right_fingers": [], "left_fingers": [], "contacted_part": "unknown", "confidence": 0.7},
                {"frame": 21, "right_contact": True, "left_contact": False, "right_fingers": ["index", "thumb"], "left_fingers": [], "contacted_part": "screen", "confidence": 0.9},
            ],
        ]
        result = aggregate_contact_windows(votes, "target_laptop", 15.0, "right")
        first = result["first_contact_frame"]
        self.assertEqual(first["frame_index"], 20)
        self.assertEqual(first["hand_side"], "right")
        self.assertEqual(first["contact_fingers"], ["index"])
        self.assertEqual(first["primary_contact_finger"], "index")

    def test_new_contact_json_is_compatible_with_dynamic_resolver(self):
        payload = {
            "vlm_result": {
                "contact_analysis": {
                    "target_object_id": "target_laptop",
                    "event_type": "first_contact",
                    "first_contact_frame": {
                        "frame_index": 20,
                        "hand_side": "right",
                        "contacted_part": "screen",
                        "contact_fingers": ["thumb", "index"],
                        "primary_contact_finger": "index",
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contact.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            frame, source = resolve_vlm_contact_frame(path, "target_laptop", list(range(40)))
        self.assertEqual(frame, 20)
        self.assertEqual(source["contact_fingers"], ["thumb", "index"])
        self.assertEqual(source["primary_contact_finger"], "index")

    def test_depth_matching_uses_nearest_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "depth").mkdir()
            for idx in range(2):
                np.save(root / "depth" / f"{idx}.npy", np.ones((2, 2), dtype=np.float32))
            (root / "frames.csv").write_text(
                "index,rgb_timestamp_s,depth_timestamp_s,depth_meters_npy\n"
                "0,0.10,0.12,depth/0.npy\n"
                "1,0.30,0.32,depth/1.npy\n",
                encoding="utf-8",
            )
            import csv

            with (root / "frames.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            match = match_depth_frame(4, 15.0, root, rows)
        self.assertEqual(match.frame_index, 1)
        self.assertAlmostEqual(match.delta_s, 0.32 - 4 / 15.0)

    def test_window_tail_is_included(self):
        self.assertEqual(make_windows(10, 15, 3, 2), [[10, 11, 12], [12, 13, 14], [13, 14, 15]])


if __name__ == "__main__":
    unittest.main()

