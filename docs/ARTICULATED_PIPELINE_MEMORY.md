# Articulated Object Reconstruction Pipeline Memory

> 这是项目级、数据无关的 articulated pipeline 约定。具体数据的视频路径、事件帧号、
> VLM 结果、Particulate part id、hinge 轴数值和 QC 结果必须写入对应 run workspace，
> 不能成为默认参数。

## 当前主线

```text
全局 RGB/timestamp/pose/depth timeline
  -> VLM 事件切片（global frame index + timestamp）
  -> SAM2 hand mask -> DiffuEraser
  -> 交互式 SAM2 whole/root/child masks
  -> Hunyuan3D canonical whole mesh
  -> Particulate parts + revolute joint proposal
  -> Video Depth Anything + true-depth metric calibration
  -> whole-part metric ICP alignment to full-video C0
  -> pre-CoTracker Viser inspection
  -> CoTracker3 RGB-D tracks + head-pose compensation
  -> hinge-constrained articulated motion q(t)
  -> EgoForce hand in C0
  -> VLM per-finger contact candidates
  -> depth/occlusion/relative-motion contact fusion
  -> contact and collision optimization
  -> preloaded final Viser
```

## 不可破坏约定

- `C0` 永远是完整视频第一帧的选定相机坐标系，不是 VLM 子事件的局部第一帧。
- 所有事件窗口都保存 `global_rgb_index`、原始视频帧号和 timestamp。
- 开门、关门等多个事件共享同一个 articulated object instance、mesh、root、part 和 hinge。
- root/body 的 pose 和 metric scale 先独立验证；child link 不能做自由逐帧 SE(3)。
- Particulate 的 joint 只是初始化和先验；轴方向、origin 和运动范围必须用 RGB-D 轨迹、mask
  和投影检查。
- CoTracker 3D 点先反投影到当前相机，再使用高频 head/camera pose 转入 `C0`，然后拟合运动。
- door 的最终运动是固定 revolute joint 的 `q(t)`，不是自由 Kabsch/ICP pose。
- VLM 事件边界和接触只是候选，精确边界由 mask、depth、相对运动和遮挡证据修正。

## 当前实现停止点

当前新数据的第一阶段实现停在：

```text
mesh / parts / hinge
  + VDA metric depth
  + whole-part ICP to full-video C0
  + Viser inspection
```

确认对齐后，下一阶段才接入 CoTracker3 和 hinge-constrained motion。

## Lid/base first-contact rule

After hinge angle fitting and any RGB mask IoU refinement, every closing articulated interval must run a geometric lid/base contact check in `C0`. Rotate the moving lid about the fixed hinge axis, measure the non-hinge lid-to-base clearance, and use bisection to locate the first contact angle. Apply that angle to the first contact frame; all later frames in the same closing interval hold it and cannot optimize farther into the base. The Stage 10 implementation defaults to a 1 mm clearance threshold, excludes a 4 cm hinge neighborhood, and records the contact angle, last safe angle, incremental rotation, clearance, and contact frame in the part manifest.

