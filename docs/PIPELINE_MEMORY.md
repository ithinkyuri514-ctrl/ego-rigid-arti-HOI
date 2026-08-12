# 手物交互 4D 重建 — Pipeline Memory

> 本文件是当前项目的工程记忆。后续改代码前先读这里，默认沿用本文件记录的主线，不要被早期失败尝试带偏。
> 最后更新：2026-07-13。

---

## 0. 当前目标

从头显 RGB-D 视频中重建一个长序列手物交互场景：

- 手与 articulated 物体交互：笔记本电脑 base 固定，screen 绕 hinge 轴闭合。
- 后续扩展到 rigid 物体：手机等刚体的 6DoF 轨迹。
- 所有结果统一到第一帧右目相机坐标系 `frame0_right_camera`，抵消头显相机运动。

当前已经接受的主线是：

```text
Qwen3VL 场景理解
  -> SAM2 得到目标 / base / screen mask
  -> Hunyuan3D 生成带纹理 laptop mesh
  -> Particulate 得到 part_14 base、part_15 screen、hinge joint
  -> base-first RGB-D 对齐到 frame0_right_camera
  -> EgoForce 在 15fps RGB 上重建手
  -> head pose 抵消相机运动
  -> Qwen3VL RGB-D 重叠窗口判断第一次 contact frame + contact fingers
  -> contact-driven screen hinge motion（默认不强制角度单调）
  -> textured Viser 动态可视化
```

---

## 1. 数据输入

主数据目录：

```text
/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export
```

关键文件/目录：

- `rgb_right_png/`
- `rgb_left_png/`
- `depth_meters_npy/`
- `depth_mm_png/`
- `frames.csv`
- `manifest.json`
- `pose/head_pose.csv`
- `pose/head_pose.jsonl`

约定：

- 当前固定世界系是第一帧右目相机坐标系：`frame0_right_camera`。
- 头显相机运动必须用 `frames.csv` / pose 抵消，不能把每帧相机系当作静止世界。
- 深度默认已经与 RGB 空间可投影对齐；项目当前使用 `camera_to_rig` depth/RGB convention。

---

## 2. 当前主线脚本

| 阶段 | 脚本 / 模块 | 当前状态 |
| --- | --- | --- |
| VLM 场景理解 | `scripts/analyze_qwen3vl_hand_interaction.py` | 已有 |
| VLM 接触/手指细化 | `scripts/analyze_qwen3vl_hand_interaction.py --contact-pass`, `vlm_sam2_recon/stages/vlm_contact_semantics.py` | 当前接入 |
| 统一 manifest | `scripts/build_unified_manifest.py` | 已有 |
| SAM2 交互分割 | `scripts/run_sam2_interactive_from_vlm.py` | 已有 |
| Hunyuan3D 重建 | `scripts/run_hunyuan3d_local.py` | 当前主用 |
| Hunyuan3D API 实现 | `vlm_sam2_recon/stages/hunyuan3d_local.py`, `vlm_sam2_recon/stages/hunyuan3d_client.py` | 当前主用 |
| Particulate 分 part/axis | `scripts/run_particulate_local.py`, `vlm_sam2_recon/stages/particulate_local.py` | 当前主用 |
| laptop 静态对齐 | `scripts/align_laptop_to_camera.py`, `vlm_sam2_recon/stages/camera_alignment.py` | 当前主用 |
| contact-driven 动态重建 | `scripts/run_contact_driven_laptop.py`, `vlm_sam2_recon/stages/contact_driven_screen.py` | 当前主用 |
| pose-only 手+laptop 基线 | `scripts/run_pose_compensated_hand_laptop.py`, `vlm_sam2_recon/stages/pose_compensated_scene.py` | 诊断基线 |
| textured laptop part 切分 | `scripts/build_textured_laptop_parts.py` | 当前主用 |
| 动态 Viser | `scripts/serve_dynamic_laptop_hand_viser.py` | 当前主用 |

---

## 3. 当前接受输出

### 3.1 Hunyuan3D 原始 mesh

```text
inputs/hunyuan3d_meshes/target_laptop/whole/target_laptop.glb
```

说明：

- 这是原始带 UV/texture/PBR material 的整机 mesh。
- 当前后续 Particulate 和 textured visualization 都围绕这个 Hunyuan mesh。
- TRELLIS2 是旧路线 / baseline，不再作为当前主线 laptop mesh。

### 3.2 Particulate 输出

```text
outputs/particulate_hunyuan/target_laptop_decimated_50000/
```

关键文件：

```text
eval/pred.npz
urdf_20260707_154830/model.urdf
urdf_20260707_154830/meshes/part_14.obj
urdf_20260707_154830/meshes/part_15.obj
```

语义约定：

```text
part_14 = laptop base
part_15 = laptop screen
joint_14_15 / joint_15_14 = revolute hinge
```

注意：

- Particulate 导出的 `part_*.obj` 没有 Hunyuan 原始 UV/texture。
- `eval/pred.npz` 里的 `face_part_ids` 才是把原始 Hunyuan 纹理 mesh 切回 textured part 的 label source。

### 3.3 当前接受静态对齐

```text
outputs/object_alignment_hunyuan_base_first_nohinge/target_laptop/frame_000000
```

关键输出：

```text
alignment_result.json
joint_camera.json
part_14_camera.obj
part_15_camera.obj
laptop_camera_aligned.glb
part_masks/
observed_base_pointcloud.ply
observed_screen_pointcloud.ply
```

当前对齐思路：

1. 使用第一帧 base mask。
2. 用真实深度反投影 base mask，得到 observed base point cloud。
3. 取 Particulate 的 base mesh：`part_14`。
4. base-first 对齐：PCA 初始化 + constrained ICP/visible surface + silhouette refine。
5. 把同一个 Sim3 刚体/尺度变换施加到 screen mesh。
6. 不再让 screen/base 做自由 6DoF 或自由 ICP。

这个版本是当前静态初始化主线。

### 3.4 EgoForce 手重建

当前 hand mesh 输入目录：

```text
outputs/egoforce_rgb_right_15fps
```

当前 contact pipeline 消费这里的 15fps hand/arm mesh。

### 3.5 Contact-driven laptop 动态输出

在进入几何 contact pipeline 前，先做两级 VLM 判断：

1. 原 scene/keyframe pass 给出粗略 `first_contact_frame`、hand side 和 contacted part。
2. `--contact-pass` 在粗帧附近生成三帧重叠 RGB-D 窗口；深度按时间戳取最近的 4 Hz 帧，并用标定外参投影到 15 Hz 右 RGB 图像。
3. 每个窗口输出逐帧左右手 contact、五指集合（thumb/index/middle/ring/pinky）、part 和 confidence。
4. 同一帧使用严格多数票；finger 组合使用 ArtHOI 风格众数，平票优先更少且更确定的手指。
5. `contact_driven_screen.py` 读取 `contact_fingers` 后，只在相应 MANO fingertip 中做几何距离搜索；旧 JSON 无手指字段时才回退全部五指。

2026-07-08 序列当前细化结果：

```text
run_new_laptop_20260708_122858/outputs/qwen3vl_contact_fingers_rgbd.json
visual first contact = frame 16
hand = right
contact fingers = [index]
primary finger = index
```

注意：15 Hz RGB 没有严格同步的 15 Hz depth；当前 4 Hz depth 最大时间差约 78 ms，因此 depth 只作为 VLM 辅助证据。`window` 模式把 frame 16 作为几何搜索起点，当前 EgoForce/index-tip 几何锁定帧是 frame 26。`force` 模式才会直接把视觉 frame 16 当作动态接触帧。

本次接入后的候选动态输出（等待视觉验收）：

```text
run_new_laptop_20260708_122858/outputs/contact_driven_laptop/
  hunyuan_base_first_hingerefined_pose15_vlmfinger_000000_000135_right
```

当前接受动态输出：

```text
outputs/contact_driven_laptop/hunyuan_base_first_fixed_frame0_tight_contact_000000_000057
```

关键文件：

```text
dynamic_manifest.json
contact_manifest.json
contact_optimization.csv
contact_points_frame_camera.npy
hand_refine_delta_m.npy
frame_000000/
...
frame_000057/
```

当前动态模型：

```text
base: 固定在 frame0_right_camera
joint axis: 固定在 frame0_right_camera
screen: 只能绕固定 hinge axis 旋转
hand: EgoForce mesh 先通过 head pose 转到 frame0_right_camera，再做小的 translation contact correction
```

当前 contact loss 约束：

- fingertip 到 screen contact point 的距离。
- fingertip 到 hinge axis 的半径保持。
- fingertip 沿 hinge axis 的坐标保持。
- hand translation prior。
- hand translation temporal smooth。
- hinge angle temporal smooth。
- hinge angular acceleration smooth。
- screen plane penetration proxy。

注意：

- contact frame 当前应优先来自 Qwen3VL 语义理解输出的 `first_contact_frame`，通过 `--vlm-contact-json` 传给 `scripts/run_contact_driven_laptop.py`。
- `--contact-force-frame` 仍保留为最高优先级手动覆盖；如果既没有手动 frame 也没有 VLM JSON，才退回旧的几何最近点搜索。
- 当前默认关闭 `enforce_monotonic_after_contact`，不再强制 screen angle 单调增长；角度可以随观测回落，避免为了视觉结果硬性“只关不回开”。
- 当前不是 MANO 参数优化，只是 mesh-level global translation correction。
- 不允许 screen 做自由 3D Kabsch/SVD 作为最终运动。
- 不允许 base 或 hinge axis 每帧动。

### 3.6 Textured laptop 可视化

Pose-only 诊断基线输出：

```text
run_new_laptop_20260708_122858/outputs/pose_compensated_hand_static_laptop/
  hunyuan_hingerefined_000000_000135_right
```

这个分支只把原始 EgoForce hand/arm 从每帧相机坐标系刚体变换到 `frame0_right_camera`。Laptop base、screen 和 hinge 全部静止，screen angle 恒为 0；不启用 contact、hand refinement 或 screen motion。动态 Viser 读取每帧 `camera_to_frame0_matrix` 后，也会把 RGB frustum 放到对应的世界系相机 pose，避免用单位位姿 RGB 错误比较世界系 mesh。

当前 textured part 输出：

```text
outputs/contact_driven_laptop/hunyuan_base_first_fixed_frame0_tight_contact_000000_000057/textured_laptop_parts
```

关键文件：

```text
part_14_camera_textured.glb
part_15_camera_textured.glb
laptop_camera_textured.glb
textured_laptop_manifest.json
```

生成脚本：

```bash
/opt/conda/envs/egoforce/bin/python scripts/build_textured_laptop_parts.py \
  --alignment-dir outputs/object_alignment_hunyuan_base_first_nohinge/target_laptop/frame_000000 \
  --dynamic-dir outputs/contact_driven_laptop/hunyuan_base_first_fixed_frame0_tight_contact_000000_000057 \
  --overwrite
```

实现逻辑：

- 从原始 Hunyuan GLB 保留 UV/material。
- 用 Particulate decimated face label 转回原始 500k faces。
- 切出 textured `part_14` 和 `part_15`。
- 施加当前 accepted static alignment 到 `frame0_right_camera`。

当前 Viser：

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

可视化约定：

- base 是静态 textured GLB，不逐帧刷新。
- screen 是同一个 textured mesh handle，通过 `wxyz/position` 绕固定 hinge axis 更新。
- 默认只显示 left hand，避免错误的另一只手干扰。

---

## 4. 坐标系与运动补偿

必须遵守：

```text
所有动态 laptop/hand 输出都在 frame0_right_camera。
```

实现位置：

```text
vlm_sam2_recon/stages/contact_driven_screen.py
```

重要逻辑：

- hand mesh 原始来自当前帧相机坐标。
- 使用当前帧头显 pose，把 hand mesh 转到第一帧右目相机坐标。
- base mesh 不跟随当前帧相机动。
- joint origin/axis 不跟随当前帧相机动。
- screen 只绕固定 axis 旋转。

如果发现 base 在动态可视化里抖动，优先检查是不是忘了从当前帧相机系变到 `frame0_right_camera`。

---

## 5. Legacy / Baseline，不作为当前主线

下面这些代码/结果可以保留用于对比，但不要默认当主线：

- TRELLIS2 laptop mesh 主线：`scripts/run_trellis2_local.py`
- Hunyuan RANSAC whole-mesh 对齐：`scripts/run_hunyuan3d_ransac_alignment.py`
- Hunyuan Particulate base alignment 旧实验：`scripts/run_hunyuan3d_particulate_base_alignment.py`
- CoTracker screen angle 估计：
  - `scripts/run_screen_cotracker_dynamic.py`
  - `scripts/run_screen_hinge_rgbd_stable.py`
  - `vlm_sam2_recon/stages/screen_hinge_tracking.py`
- 自由 Kabsch/SVD 估计 screen 3D rotation。
- 每帧自由 ICP 对齐 screen。
- 没有 head-pose compensation 的动态 laptop 可视化。
- MANO/global-rigid proxy 过强吸收 screen motion 的实验。

CoTracker 经验：

- 前半段可用，但遮挡、反光、出画后 track 会漂移/点减少。
- 当前接受结果不再靠 CoTracker 点直接估角，而是 contact-driven hinge 约束。

---

## 6. 当前已完成

- 项目骨架和 manifest/schema 已搭好。
- SAM2 mask 流程已跑通。
- TRELLIS2 旧 mesh 跑通过，但 laptop 主线已切到 Hunyuan3D。
- Hunyuan3D API 生成 laptop mesh 已跑通。
- Particulate 在 Hunyuan laptop 上跑通，得到 base/screen/joint。
- base-first RGB-D 静态对齐已接受。
- EgoForce 15fps hand mesh 已作为输入接入。
- head pose compensation 已接入 contact pipeline。
- contact-driven laptop screen dynamic 已接受。
- textured Hunyuan laptop part 可视化已接入 Viser。

---

## 7. 下一步建议

短期下一步：

1. 把 textured laptop 作为默认 visualizer 路径。
2. 检查 contact-driven hand 和 textured screen 的视觉接触是否更清楚。
3. 开始 rigid object：`target_phone` 的 6DoF 对齐和放置事件。

2026-07-15 新刚体序列已单独建 memory，不要把 laptop hinge/contact 假设带过去：

```text
docs/RIGID_PIPELINE_MEMORY.md
run_rigid_20260715_151803/
```

中期：

- 将 contact-driven 输出回写 `ProjectManifest`。
- 把 laptop event 和 phone event 做成统一多事件时序。
- 手部 refinement 从 mesh translation 升级到谨慎的 MANO-layer refinement，但不能让手的自由度偷走 screen motion。

---

## 8. 改代码前检查清单

- [ ] 当前任务是否属于主线？若是，优先读本文件。
- [ ] 是否使用 Hunyuan mesh，而不是 TRELLIS2 laptop mesh？
- [ ] 是否使用 Particulate Hunyuan 输出：`outputs/particulate_hunyuan/...`？
- [ ] laptop part 标签是否仍是 `14=base`, `15=screen`？
- [ ] 静态初始化是否使用 `object_alignment_hunyuan_base_first_nohinge`？
- [ ] 动态输出是否在 `frame0_right_camera`？
- [ ] 是否正确使用 head pose 抵消相机运动？
- [ ] screen 是否只绕固定 hinge axis 旋转？
- [ ] base / joint axis 是否保持固定？
- [ ] 可视化是否优先使用 textured laptop parts？
