# Mixed Pipeline Contract

| Stage | Required input | Required output |
| --- | --- | --- |
| 00 | raw stereo video + SpatialMP4 export | right-eye timeline, camera calibration, C0 pose mapping |
| 01 | all global right-eye frames | articulated/rigid event list and interacted objects with frame-0 boxes |
| 02 | original 15fps right-eye timeline | human-clicked SAM2 hand masks propagated over every frame |
| 03 | original 15fps right-eye video + propagated hand mask video | DiffuEraser hand-removed 15fps video and extracted hand-removed frames |
| 04 | hand-removed 15fps video/frames + VLM object routing | human-clicked SAM2 object masks propagated over every frame, plus frame-0 mesh prompt RGB/mask |
| 05 | hand-removed frame-0 RGB + confirmed frame-0 object masks | canonical/posed SAM3D mesh, pose, scale and projection QC |
| 06 | RGB timeline + true depth anchors | full-resolution metric depth, including frame 0 |
| 07 | SAM3D outputs + frame-0 mask/depth | aligned C0 mesh, observed object cloud, ICP/IoU report |
| 08 | aligned meshes + RGB-D | interactive Viser inspection |

The pipeline must not silently switch to an event-local coordinate origin or use an event-local
frame as the mesh prompt. All paths and model roots are explicit in `configs/mixed_recon_config.json`.
Downstream tracking must consume the human-confirmed Stage 04 sequence, preferably
`outputs/04_object_masks/<object_id>/objects/<object_id>`; compatibility mirrors may also appear
under `outputs/02_sam2_frame0_masks/propagated/objects/<object_id>`.
The frame-0 prompt remains the reconstruction prompt contract, not the tracking-mask sequence.
Every timeline index must have a mask file. Fully invisible frames use an empty mask and are marked
through the manifest's per-object visibility range and terminal-empty-frame QC fields.

### Stage 10 articulated contact invariant

For a closing articulated interval, the final motion must be collision-aware: after angle fitting, Stage 10 searches along the fixed hinge axis for the first lid/base contact angle using mesh clearance bisection, applies that contact angle, and holds it for later frames. It must not continue optimizing into lid/base penetration. Default contact threshold is 1 mm outside a 4 cm hinge exclusion region; `--no-enable-contact-angle` is reserved for diagnostic comparisons.

