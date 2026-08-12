from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from vlm_sam2_recon.manifest_io import read_project_manifest, validate_or_raise, write_project_manifest
from vlm_sam2_recon.schemas import (
    BBox2D,
    FrameRef,
    MaskArtifact,
    ProjectManifest,
    SourceData,
    TargetObject,
)


class ManifestSchemaTest(unittest.TestCase):
    def test_round_trip_and_validate(self):
        source = SourceData(sequence_id="seq_test", frame_dir="/tmp/frames")
        manifest = ProjectManifest.new(project_root="/tmp/project", source=source)
        manifest.targets.append(
            TargetObject(
                object_id="target_box",
                name_en="box",
                object_class="rigid",
                selected_keyframe=FrameRef(frame_index=3, frame_file="000003.png"),
                selected_bbox=BBox2D(xyxy=[1, 2, 30, 40], coordinate_space="pixel"),
                selected_mask_id="sam2_batch:target_box:frame_3",
            )
        )
        manifest.masks.append(
            MaskArtifact(
                mask_id="sam2_batch:target_box:frame_3",
                target_id="target_box",
                frame=FrameRef(frame_index=3, frame_file="000003.png"),
                mask_png="/tmp/mask.png",
                bbox=BBox2D(xyxy=[1, 2, 30, 40], coordinate_space="pixel"),
            )
        )
        validate_or_raise(manifest)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "project_manifest.json"
            write_project_manifest(path, manifest)
            loaded = read_project_manifest(path)

        self.assertEqual(loaded.schema_version, manifest.schema_version)
        self.assertEqual(loaded.source.sequence_id, "seq_test")
        self.assertEqual(loaded.targets[0].selected_mask_id, "sam2_batch:target_box:frame_3")
        self.assertEqual(loaded.masks[0].bbox.xyxy, [1, 2, 30, 40])
        validate_or_raise(loaded)


if __name__ == "__main__":
    unittest.main()
