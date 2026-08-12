# Accepted Native36-Left Pipeline

本仓库只保留 `run_mixed_20260802_142359_native36_left` 最终 accepted 流程需要的代码。

## Execution Order

```text
SpatialMP4 native RGB-D timeline
-> refined C0 camera poses
-> Qwen3-VL event routing
-> interactive SAM2 hand masks
-> DiffuEraser hand removal
-> interactive SAM2 object/part masks
-> SAM3D frame-0 meshes
-> native metric-depth alignment
-> Particulate part/joint prediction
-> fixed-axis articulated tracking with lid/base contact limit
-> FoundationPose + masked RGB-D ICP rigid tracking
-> rigid-on-articulated support and event gating
-> EgoForce hand/arm reconstruction in C0
-> adaptive hand-object contact correction
-> full-scene Viser playback
```

## Source Boundary

- 精确脚本清单：`docs/REQUIRED_CODE_NATIVE36_LEFT.txt`
- 安装和运行：`docs/GITHUB_HANDOFF_NATIVE36_LEFT.md`
- 第三方依赖：`docs/DEPENDENCY_AUDIT_NATIVE36_LEFT.md`
- 安装自检：`python tools/check_native36_installation.py`

WiLoR、Hunyuan、TRELLIS、PhysX、native40、旧 demo 和诊断分支不属于本仓库 accepted 主线。
