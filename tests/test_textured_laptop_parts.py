import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_textured_laptop_parts.py"
SPEC = importlib.util.spec_from_file_location("build_textured_laptop_parts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TexturedLaptopPartsTest(unittest.TestCase):
    def test_reads_screen_hinge_refinement(self):
        result = {
            "alignment": {
                "hinge_refine": {
                    "screen_part_label": "15",
                    "joint_name": "joint_14_15",
                    "chosen_angle_deg": -4.25,
                }
            }
        }
        got = MODULE.initial_screen_hinge_refinement(result, "15")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got["angle_deg"], -4.25)
        self.assertEqual(got["joint_name"], "joint_14_15")

    def test_does_not_apply_base_moving_refinement_to_screen(self):
        result = {
            "alignment": {
                "hinge_refine": {
                    "moving_part_label": "14",
                    "chosen_angle_deg": 8.0,
                }
            }
        }
        self.assertIsNone(MODULE.initial_screen_hinge_refinement(result, "15"))


if __name__ == "__main__":
    unittest.main()
