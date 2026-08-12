# Accepted Mixed Hand-Object Reconstruction Pipeline Memory

This is the accepted pipeline for reconstructing one egocentric sequence containing hands,
an articulated object, and a rigid object. The accepted result combines semantic event routing,
full-sequence segmentation, metric RGB-D geometry, articulated part motion, rigid object motion,
pose-compensated hand geometry, object-object support, hand-object contact correction, and Viser
playback in one common frame-0 camera coordinate system.

## Accepted scope

The pipeline must reconstruct all of the following together:

- pose-compensated left/right hand and arm geometry;
- canonical and C0-aligned meshes for every manipulated object;
- articulated parts, joints, and per-frame joint motion;
- rigid-object per-frame motion;
- semantic hand-object event intervals from the full video;
- articulated-part collision limits;
- rigid-on-articulated support relationships;
- pre-event and post-event rigid-object stability;
- final combined Viser playback.

## Coordinate and timeline invariants

- The selected eye and destination coordinate frame come from
  `outputs/00_rgb_frames/stage00_manifest.json` and `camera.json`; downstream code must not infer
  the eye from legacy filenames such as `right_rgb_png`.
- This accepted run uses the left-eye native RGB-D timeline and
  `frame0_left_camera_opencv_rdf` as C0.
- All object, part, hand, arm, joint, point-cloud, contact, and visualization outputs must be
  transformed into the same C0 frame.
- The global modeling image is always global RGB frame 0. An event-local first frame must never
  replace frame 0 as the reconstruction prompt or coordinate origin.
- Native RGB-D timestamps and camera poses are geometry truth. DiffuEraser and learned dense depth
  are appearance or fallback aids, not replacements for available metric depth.
- Raw model/tracking outputs are preserved. Refinements write new pose files and manifests rather
  than silently overwriting the inputs.

## Accepted execution order

`00 native RGB-D timeline and C0 poses
-> 00 camera-pose refinement
-> 01 full-timeline Qwen3-VL event/object routing
-> 02 human-interactive SAM2 hand masks and propagation
-> 03 DiffuEraser hand removal
-> 04 human-interactive SAM2 whole-object and articulated-part masks
-> 05 global-frame-0 SAM3D object reconstruction
-> 06 native metric-depth projection
-> 07 frame-0 RGB-D mesh alignment
-> Particulate articulated part/joint prediction
-> articulated part mask confirmation
-> articulated fixed-axis tracking and lid/base contact limit
-> rigid FoundationPose plus masked RGB-D ICP tracking
-> rigid-on-articulated support refinement
-> VLM event-gated rigid-pose stabilization
-> EgoForce pose-compensated hand/arm reconstruction
-> hand-object contact correction
-> combined Viser inspection`.

Stage numbers are historical output labels, not a dependency order. In particular, Particulate
output under `outputs/12_particulate` is a prerequisite for articulated tracking under
`outputs/10_*`.

## Semantic event routing

- Qwen3-VL sees the complete selected-eye timeline before returning results.
- It identifies every manipulated object, classifies it as `articulated` or `rigid`, and records
  event start/end frames, actions, evidence frames, hand side, terminal-state evidence, and the
  global-frame-0 object box.
- VLM semantics route geometry and motion algorithms; VLM output does not directly determine
  metric pose.
- Spatial relations such as `mouse on top of closed laptop` become explicit support constraints:
  supported object, support object, support part, activation interval, and confidence.
- A rigid object is allowed to move only inside its accepted interaction interval. Frames before
  the event retain the initial pose; frames after the confirmed terminal state retain a robust
  terminal pose.

## Hand masks and hand removal

- Human-clicked SAM2 hand masks are propagated over every timeline frame.
- Every frame has an explicit mask; invisible frames use an empty mask rather than a missing file.
- DiffuEraser consumes the RGB video and propagated hand masks to generate hand-removed frames.
- Hand-removed frames are used for object segmentation and visual tracking, but never as geometry
  truth.

## Object and part masks

- Whole-object SAM2 starts only after hand removal.
- Every manipulated object receives a full-timeline mask sequence and a global-frame-0 mesh prompt.
- Articulated moving parts receive their own full-timeline masks before part tracking.
- The intended source of truth is `outputs/04_object_masks/<object_id>`; historical mirrors may
  exist under `outputs/02_sam2_frame0_masks`, and cache artifacts may exist under
  `/tmp/vlm_sam2_recon_cache`, but downstream manifests must record the exact consumed path.
- Intermediate empty masks are failures. Only a documented terminal empty suffix is acceptable.

## Frame-0 object reconstruction and alignment

- Every manipulated object receives a SAM3D canonical mesh, initial pose, scale, and projection QC
  from global frame 0.
- SAM3D pose and scale are initialization values, not metric truth.
- Frame-0 metric depth is backprojected inside the confirmed object mask.
- ICP, fixed-scale refinement, point-to-plane terms, silhouette IoU, and depth consistency align
  each object independently in C0.
- Accepted outputs include the aligned C0 mesh, observed C0 point cloud, overlays, transforms, and
  quantitative alignment report.

## Articulated-object branch

- Particulate runs after the whole articulated mesh has been aligned to C0 and before moving-part
  tracking.
- Particulate supplies part meshes, parent/child links, joint type, joint origin, joint axis, and
  optional URDF/MJCF outputs.
- For the accepted laptop result, `part_15/link_15` is the base and `part_14/link_14` is the moving
  lid/screen connected by a revolute joint.
- The moving part is tracked from hand-removed RGB, part masks, metric depth, and refined camera
  poses, then constrained to the fixed C0 revolute axis.
- Closing motion applies a lid/base geometric contact limit. The moving lid stops at first safe
  contact rather than entering the base. Particulate joint limits are not treated as metric truth.
- Final articulated outputs include per-frame part transforms, joint angles, track points,
  confidence/depth provenance, collision/contact fields, and a manifest.

## Rigid-object branch

- The rigid object begins from its Stage 07 aligned C0 mesh.
- FoundationPose supplies the temporal initialization; masked native RGB-D ICP refines translation
  while retaining the Stage 07 frame-0 anchor.
- Invalid or low-quality ICP frames fall back according to the tracking manifest and remain
  explicitly recorded.
- Raw rigid poses remain available even when support or event-gating refinements are accepted.

## Rigid-on-articulated support refinement

The accepted support rule is semantic plus geometric:

1. VLM identifies a relation such as `mouse on top of laptop` and the relevant rigid interaction.
2. The first geometric contact identifies which articulated part is touched.
3. At that first-contact frame, the articulated part is transformed to its current C0 pose.
4. The nearest outward-facing local surface of that current part defines the support plane and
   support normal.
5. During the placement interval, each rigid tracking pose remains the temporal observation and is
   corrected along the locked support normal to maintain a small positive clearance rather than
   penetrate the support surface.
6. First contact selects the support surface; it does not by itself mean the object has settled.
7. A separate event gate stabilizes the rigid object before interaction and after confirmed
   placement.

For the current mouse/laptop run:

- the first detected mouse/laptop contact is frame 23;
- the supporting articulated part is laptop `part_14`;
- the target support clearance is 1.5 mm;
- the support-refined poses are stored in
  `outputs/11_object_object_support_mouse_laptop/mouse_poses_support_refined.npy`;
- support selection and per-frame QC are stored in
  `outputs/11_object_object_support_mouse_laptop/support_manifest.json`.

## VLM event-gated rigid-pose stabilization

A rigid object must remain motionless outside the semantic manipulation interval.

- Before the event starts, all frames hold the initial frame-0-aligned rigid pose.
- Inside the event interval, tracking and support refinement are allowed to move the object.
- After the confirmed terminal state, all frames hold one robust terminal pose.
- The terminal pose is estimated from a stable post-placement window with translation outlier
  rejection; isolated FoundationPose/ICP jumps must not appear in the final playback.

For the current mouse result:

- frames 0-17 hold the initial mouse pose;
- frames 18-29 contain the accepted pickup/move/place motion;
- frames 30-35 hold a robust terminal pose;
- frame 32 is rejected as a terminal-window translation outlier;
- pre-event and post-event maximum pose differences are both exactly zero;
- the accepted playback pose file is
  `outputs/11_object_object_support_mouse_laptop/mouse_poses_support_event_gated.npy`;
- the gate report is
  `outputs/11_object_object_support_mouse_laptop/event_gate_manifest.json`.

## Hand and hand-object branch

- EgoForce reconstructs left/right hand and arm geometry in each current camera frame.
- Refined camera poses transform every EgoForce result into C0 using
  `p_C0 = T_C0_from_Ct[frame] @ p_Ct`.
- SAM2 consistency and per-frame QC decide valid hand candidates.
- Hand-object optimization may apply bounded hand translation and local hand-vertex offsets, but it
  must not silently move finalized object trajectories.
- Object-object support and event gating finalize rigid-object motion before a final hand-contact
  pass is accepted.

## Accepted current-run artifacts

Workspace:

`/code/vlm_sam2_recon/run_mixed_20260802_142359_native36_left`

Key accepted outputs:

- refined camera poses: `outputs/00_pose_refinement/poses_refined.npz`;
- semantic events: `outputs/01_vlm/mixed_interactions.json` and the complete terminal-state variant;
- hand masks: `outputs/02_hand_masks`;
- hand-removed RGB: `outputs/03_diffueraser`;
- SAM3D objects: `outputs/03_sam3d_frame0`;
- native metric depth: `outputs/06_dense_depth` and native cache paths recorded in manifests;
- aligned object meshes: `outputs/07_alignment`;
- rigid mouse tracking initialization: `outputs/08_foundationpose_translation_icp_final/mouse`;
- pose-compensated hands: `outputs/09_egoforce`;
- articulated laptop tracking and contact-limited angles: `outputs/10_*`;
- mouse/laptop support and event gate: `outputs/11_object_object_support_mouse_laptop`;
- hand-object correction: `outputs/11_hand_object_optimization_*`;
- Particulate parts/joints: `outputs/12_particulate/laptop`.

Current combined Viser playback uses:

- laptop base/lid from `outputs/12_particulate/laptop/parts_C0`;
- laptop angles from `outputs/10_articulate_tracking_iou_vda_contact_track0/link_14`;
- mouse mesh from `outputs/07_alignment/mouse/frame_000000/sam3d_aligned_C0.glb`;
- mouse poses from
  `outputs/11_object_object_support_mouse_laptop/mouse_poses_support_event_gated.npy`;
- hand geometry from `outputs/09_egoforce/dynamic_manifest.json` with available optimized hand
  geometry under `outputs/11_hand_object_optimization_dashscope/final/optimized_C0`.

## Required QC and memory updates

Every accepted run records:

- selected eye, frame count, native FPS, C0 coordinate convention, and pose source;
- VLM event intervals, terminal confirmations, object classes, and support relations;
- hand/object/part mask propagation coverage and rejected frames;
- SAM3D projection QC and Stage 07 metric alignment metrics;
- Particulate part assignments and C0 joint axes;
- articulated track/depth confidence and collision-limited angles;
- rigid tracking fallbacks, support-plane selection, contact clearance, and event-gate boundaries;
- pre-event/post-event pose constancy and terminal-window outliers;
- EgoForce detection coverage and hand-contact correction metrics;
- exact artifacts consumed by the final Viser command.
