# Current Pipeline: Hunyuan Textured Contact-Driven Laptop Reconstruction

This is the accepted mainline for the current laptop sequence. Earlier
TRELLIS2, CoTracker-only, free ICP, and no-head-pose-compensation attempts are
legacy/baseline only.

Detailed engineering memory lives in:

```text
docs/PIPELINE_MEMORY.md
```

## Accepted Mainline

```text
Qwen3VL scene understanding
  -> SAM2 masks
  -> Hunyuan3D textured laptop mesh
  -> Particulate part/joint prediction
  -> base-first RGB-D static alignment into frame0_right_camera
  -> EgoForce hand reconstruction on 15fps RGB
  -> head-pose compensation into frame0_right_camera
  -> contact-driven hinge screen motion
  -> textured Viser playback
```

## Key Scripts

- `scripts/analyze_qwen3vl_hand_interaction.py`
- `scripts/run_sam2_interactive_from_vlm.py`
- `scripts/run_hunyuan3d_local.py`
- `scripts/run_particulate_local.py`
- `scripts/align_laptop_to_camera.py`
- `scripts/run_contact_driven_laptop.py`
- `scripts/build_textured_laptop_parts.py`
- `scripts/serve_dynamic_laptop_hand_viser.py`

Core modules:

- `vlm_sam2_recon/stages/hunyuan3d_local.py`
- `vlm_sam2_recon/stages/hunyuan3d_client.py`
- `vlm_sam2_recon/stages/particulate_local.py`
- `vlm_sam2_recon/stages/camera_alignment.py`
- `vlm_sam2_recon/stages/contact_driven_screen.py`

## Accepted Inputs / Outputs

Spatial export:

```text
/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export
```

Hunyuan3D original textured mesh:

```text
inputs/hunyuan3d_meshes/target_laptop/whole/target_laptop.glb
```

Particulate Hunyuan output:

```text
outputs/particulate_hunyuan/target_laptop_decimated_50000
```

Accepted static alignment:

```text
outputs/object_alignment_hunyuan_base_first_nohinge/target_laptop/frame_000000
```

Accepted dynamic contact output:

```text
outputs/contact_driven_laptop/hunyuan_base_first_fixed_frame0_tight_contact_000000_000057
```

Textured laptop parts:

```text
outputs/contact_driven_laptop/hunyuan_base_first_fixed_frame0_tight_contact_000000_000057/textured_laptop_parts
```

## Critical Conventions

- Coordinate frame: `frame0_right_camera`.
- Head-mounted camera motion must be compensated with pose data.
- `part_14 = laptop base`.
- `part_15 = laptop screen`.
- Base stays fixed.
- Hinge axis stays fixed.
- Screen only rotates around the fixed hinge axis.
- Hand correction is currently mesh-level global translation, not MANO pose
  optimization.
- Textured visualization uses Hunyuan original UV/material transferred back to
  part meshes.

## Current Viser Command

```bash
/opt/conda/envs/egoforce/bin/python scripts/serve_dynamic_laptop_hand_viser.py \
  --dynamic-dir outputs/contact_driven_laptop/hunyuan_base_first_fixed_frame0_tight_contact_000000_000057 \
  --egoforce-dir outputs/egoforce_rgb_right_15fps \
  --port 8115 \
  --fps 15 \
  --hand-side left \
  --use-textured-laptop \
  --textured-laptop-dir outputs/contact_driven_laptop/hunyuan_base_first_fixed_frame0_tight_contact_000000_000057/textured_laptop_parts \
  --no-show-rgb \
  --no-show-rgbd
```

## Legacy / Baseline

These are kept for comparison, not current mainline:

- TRELLIS2 laptop mesh route.
- CoTracker-only screen angle estimation.
- `scripts/run_screen_cotracker_dynamic.py`.
- `scripts/run_screen_hinge_rgbd_stable.py`.
- Free Kabsch/SVD or free ICP as final screen motion.
- Dynamic visualization without head-pose compensation.
- MANO/global-rigid proxy experiments that steal screen motion.

## Next Step

The next useful pipeline stage is rigid object handling for `target_phone`:

```text
SAM2 mask + Hunyuan/TRELLIS mesh
  -> RGB-D rigid 6DoF alignment
  -> pose trajectory through the placing event
  -> integrate with hand + laptop dynamic scene
```
