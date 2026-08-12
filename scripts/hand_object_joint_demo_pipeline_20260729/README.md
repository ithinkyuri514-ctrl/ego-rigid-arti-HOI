# Hand-Object Joint Reconstruction Demo Pipeline

This folder records the pipeline used for the current hand-object reconstruction
demo. The numbered Python entries are relative symbolic links to the canonical
implementations in `/code/vlm_sam2_recon/scripts`, so fixes remain shared and
`PROJECT_ROOT` continues to resolve correctly.

## Recorded Run

- Video: `/code/3DVideo_2026-07-28-20-37-34-217.mp4`
- Spatial export: `/tmp/3DVideo_2026-07-28-20-37-34-217_spatialmp4_export`
- Full 15 fps source run: `/code/vlm_sam2_recon/run_mixed_20260728_203734`
- Accepted native RGB-D run: `/code/vlm_sam2_recon/run_mixed_20260728_203734_depth40`
- Camera: right eye
- World frame: first right camera, OpenCV RDF
- Camera compensation: `p_C0 = T_C0_from_Ct @ p_Ct`

The native 40-frame run is the accepted demo timeline. Frame-0 SAM3D meshes and
alignment are reused from the full run because frame 0 is pixel-identical.

## Pipeline

| Stage | Operation | Interaction |
|---|---|---|
| `00_prepare` | Start at video frame 0 and prepare right-eye RGB timeline | automatic |
| `01_vlm` | Detect rigid/articulated objects and interaction intervals | automatic |
| `02_hand_masks` | Click SAM2 positive/negative hand prompts and propagate | manual pause |
| `03_hand_removal` | Run DiffuEraser on the hand masks | automatic |
| `04_object_masks` | Click object prompts on hand-removed RGB and propagate | manual pause |
| `05_geometry` | Reconstruct frame-0 objects with SAM3D and align to C0 | automatic + QC |
| `06_depth_pose` | Use metric depth and refine camera poses on static background | automatic |
| `07_rigid` | CoTracker/FoundationPose rigid tracking; remove bottle rotation | automatic |
| `08_hands` | Reconstruct both EgoForce hands and transform Ct geometry to C0 | automatic |
| `09_articulation` | Particulate parts/axis, part SAM2, constrained CoTracker hinge | manual + automatic |
| `10_visualization` | Joint Viser for RGB-D, hands, bottle, body, and door | automatic |

## Accepted Demo Decisions

1. Tracking RGB is the DiffuEraser hand-removed sequence.
2. Native metric depth is authoritative; hand pixels are invalidated for depth
   sampling.
3. Refined camera poses are used for every Ct-to-C0 conversion.
4. The symmetric bottle keeps only FoundationPose mesh-center translation. Its
   orientation is fixed at frame 0 and its frame-17 pose is frozen afterward.
   The accepted refinement branch uses the FoundationPose mesh center as its
   initialization and aligns each frame to hand-masked object depth with
   translation-only bounded ICP. All output rotation deltas remain identity.
   It is indexed as `07_rigid/04_refine_foundationpose_masked_rgbd_icp.py`.
5. Independent per-frame FoundationPose for the microwave door is a rejected
   baseline because the planar door produces 90-180 degree pose ambiguities.
6. The accepted door result processes the `close` event from frame 19 through
   frame 27. The original VLM end is frame 25; frames 26-27 are an explicit
   caller extension. CoTracker queries are created at frame 19. Two tracks are retained after 2D
   confidence, visibility, anchor depth, and mesh-distance filtering.
7. Door points are lifted to C0 once at frame 19. Later hinge angles are fitted
   by 2D reprojection around the fixed Particulate axis. The door has exactly one
   DoF and no independent translation.
8. The tracked closing angles reach about `-43.7` degrees at frame 25. Because
   the two tracks become inconsistent with the fixed-axis model afterward,
   frames 26-27 follow the original frame-28 joint-limit interpolation rate and
   reach about `-49.4` and `-55.2` degrees. Frames 28-39 hold that truncated angle.

## Native 40-Frame Preparation

```bash
WORKSPACE=/code/vlm_sam2_recon/run_mixed_20260728_203734_depth40
SOURCE_WORKSPACE=/code/vlm_sam2_recon/run_mixed_20260728_203734
SPATIAL_EXPORT=/tmp/3DVideo_2026-07-28-20-37-34-217_spatialmp4_export

/opt/conda/envs/arthoi/bin/python optional_native40/00_prepare_native40.py \
  --workspace "$WORKSPACE" \
  --source-workspace "$SOURCE_WORKSPACE" \
  --spatial-export "$SPATIAL_EXPORT" \
  --frame-start 0 --frame-count 40 --tracking-splat-radius 2

/opt/conda/envs/arthoi/bin/python optional_native40/01_register_reused_artifacts.py \
  --workspace "$WORKSPACE" \
  --source-workspace "$SOURCE_WORKSPACE" \
  --objects bottle microwave
```

## Accepted Articulated Tracking Command

```bash
WORKSPACE=/code/vlm_sam2_recon/run_mixed_20260728_203734_depth40
SOURCE=/code/vlm_sam2_recon/run_mixed_20260728_203734

/opt/conda/envs/arthoi/bin/python 09_articulation/02_track_hinge_cotracker.py \
  --workspace "$WORKSPACE" \
  --object-id microwave \
  --part-id link_14 \
  --part-mesh "$SOURCE/outputs/12_particulate/microwave/parts_C0/part_14.obj" \
  --mask-dir "$WORKSPACE/outputs/04_object_masks/microwave/parts/link_14/objects/link_14" \
  --joint-json "$SOURCE/outputs/12_particulate/microwave/joint_axes_C0.json" \
  --joint-name joint_15_14 \
  --rgb-dir "$WORKSPACE/outputs/03_diffueraser/inpainted_frames_png" \
  --depth-dir /tmp/vlm_sam2_recon_cache/run_mixed_20260728_203734_depth40/native_depth/metric_depth_npy \
  --poses-path "$WORKSPACE/outputs/00_pose_refinement/poses_refined.npz" \
  --output-dir "$WORKSPACE/outputs/10_articulate_tracking_axis_cotracker_close" \
  --interaction-actions close \
  --interaction-end-frame 27 \
  --terminal-joint-limit \
  --terminal-joint-limit-frame 28 \
  --anchor-frames 19 \
  --queries-per-anchor 24 \
  --tracker-confidence 0.7 \
  --stable-track-count 2 \
  --min-stable-valid-ratio 0.8 \
  --min-stable-median-confidence 0.7 \
  --max-anchor-mesh-distance-m 0.08 \
  --min-angle-points 2 \
  --angle-sign -1 \
  --no-enable-raw-icp \
  --skip-stage-state-update
```

The Particulate child-mesh convention requires the recorded fitted angle sign
to be negated when applying it to `part_14.obj`. The accepted output manifest
records `angle_application_sign: -1`; the axis line itself is unchanged.

## Accepted Visualization

The active viewer command is recorded in `pipeline.json`. Its main inputs are:

- bottle: `outputs/08_foundationpose_pose_refined/bottle_translation_only`
- hands: `outputs/09_egoforce/dynamic_manifest.json`
- microwave body: source `parts_C0/part_15.obj`
- microwave door: source `parts_C0/part_14.obj`
- door motion: `outputs/10_articulate_tracking_axis_cotracker_close/link_14`
- background: native metric depth with refined camera poses

Run `python check_pipeline.py` to verify that all indexed scripts and accepted
demo artifacts still exist.
