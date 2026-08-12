# Mixed RGB-D Hand–Object Reconstruction交接文档

> 适用运行：`mixed_20260802_142359_native36_left`  
> 目标：将核心代码、环境、模型权重、数据目录和运行方法交给同事，用于内部 GitHub 仓库和本地复现。

## 1. 项目定位

本项目从第一视角 SpatialMP4 RGB-D 视频中重建：

- 手和手臂几何；
- 一个铰接物体及其部件运动；
- 一个刚体物体的 6DoF/平移轨迹；
- 刚体物体放到铰接物体表面的支撑关系；
- 手—物体接触修正；
- 统一 C0 坐标系下的动态 Viser 场景。

本交接版本的示例场景是：

```text
right hand closes a laptop lid
right hand moves a mouse onto the closed laptop lid
```

本次运行目录：

```text
/code/vlm_sam2_recon/run_mixed_20260802_142359_native36_left
```

注意：运行目录中的部分历史字段仍保留了 `left`/`right` 兼容命名。交接和复现时，应以 `outputs/00_rgb_frames/stage00_manifest.json`、`camera.json` 和 `timeline.csv` 的实际 `selected_eye`、坐标系和帧数为准，不要根据目录名猜测相机眼别。

## 2. 全局约定

```text
坐标系：frame0_right_camera_opencv_rdf
建模帧：global frame 0
时间轴：原生 SpatialMP4 RGB-D 时间轴
深度真值：原生 metric RGB-D depth
相机补偿：p_C0 = T_C0_from_Ct[frame] @ p_Ct
```

不可破坏的约定：

- 所有 mesh、pose、hand、arm、joint、point cloud 和 contact 输出进入同一个 C0 坐标系。
- SAM3D 的 pose/scale 只作为初始化；最终位置由 frame-0 RGB-D 对齐修正。
- VLM 只负责事件和语义路由，不直接决定 metric pose。
- 物体轨迹完成后，手物优化不得偷偷修改最终物体轨迹。
- 运行目录和模型权重不提交到普通 Git。

## 3. 推荐仓库结构

GitHub 仓库只放代码、配置模板、文档和小型示例：

```text
vlm-sam2-recon/
├── README.md
├── .gitignore
├── configs/
│   ├── mixed.example.json
│   └── paths.example.json
├── docs/
├── scripts/
├── vlm_sam2_recon/
├── environment/
├── examples/
│   └── native36_left/
└── tools/
```

不要把以下内容整体提交：

```text
run_*/
outputs/
原始 mp4
SpatialMP4 export
逐帧 RGB/depth/mask
模型 checkpoint
/tmp/vlm_sam2_recon_cache
第三方仓库源码副本
```

完整运行目录应放到公司共享盘、对象存储或内部 Release asset，并在 README 中提供下载地址和 SHA256。

## 4. Accepted 代码交付清单

GitHub 仅跟踪 `docs/REQUIRED_CODE_NATIVE36_LEFT.txt` 中列出的 accepted native36-left 主线代码。当前包括 34 个阶段入口/直接后端脚本、主线共享模块和 5 个核心测试；WiLoR、Hunyuan、TRELLIS、PhysX、native40、旧 demo、旧 Stage 11 接触链及诊断可视化不提交。

### 4.1 工作区、原生 RGB-D 和事件理解

```text
scripts/init_mixed_interaction_workspace.py
scripts/mixed_stage00_prepare.py
scripts/rigid_stage00_prepare.py
scripts/prepare_native_rgbd_workspace.py
scripts/refine_camera_poses_static_rgbd.py
scripts/analyze_mixed_interactions_qwen3vl.py
```

### 4.2 Mask、去手和 frame-0 SAM3D

```text
scripts/mixed_stage02_hand_masks.py
scripts/mixed_stage02_propagate_masks.py
scripts/mixed_stage02_sam2_frame0.py
scripts/rigid_stage02_hand_mask_server.py
scripts/rigid_stage03_diffueraser.py
scripts/mixed_stage03_sam3d_frame0.py
scripts/run_sam3d_objects_prompt.py
scripts/mixed_stage04_object_masks.py
scripts/mixed_stage04_articulate_part_masks.py
scripts/rigid_stage04_object_mask_server.py
scripts/rigid_stage04_object_masks.py
```

Stage 02 和 Stage 04 的 SAM2 点选仍需要人工确认；Hunyuan 按钮及兼容调用已从 accepted object-mask server 删除。

### 4.3 深度、对齐和运动跟踪

```text
scripts/mixed_stage06_frame0_depth.py
scripts/mixed_stage07_align_sam3d.py
scripts/mixed_stage12_particulate.py
scripts/mixed_stage10_track_articulate_parts.py
scripts/rigid_stage08_track_pose.py
scripts/mixed_stage08_track_rigid.py
scripts/rigid_stage08_foundationpose_independent.py
scripts/rigid_stage08_refine_foundationpose_icp.py
```

`prepare_native_rgbd_workspace.py` 生成全时间轴同步 native metric depth；`mixed_stage06_frame0_depth.py` 是 frame-0 对齐辅助步骤。

### 4.4 约束、EgoForce 和手物优化

```text
scripts/constrain_articulated_part_body_contact.py
scripts/constrain_rigid_pose_translation_only.py
scripts/refine_terminal_rigid_contact_clearance.py
scripts/refine_mouse_support_on_articulated_part.py
scripts/gate_rigid_pose_by_interaction_event.py
scripts/rigid_stage09_egoforce.py
scripts/export_egoforce_raw_all_c0.py
scripts/adaptive_contact_optimize.py
```

### 4.5 最终可视化

```text
scripts/serve_full_scene_viser.py
```

## 5. 外部仓库和 Conda 环境

建议同事在同一台 GPU 服务器上分别创建环境，不要把所有依赖强行合并到一个环境。

| 环境 | 用途 | 主要外部目录 |
|---|---|---|
| `qwen3vl` | Qwen3-VL 全时间轴事件理解 | `/code/Qwen3-VL` |
| `sam3d-objects` | SAM3D frame-0 mesh | `/code/sam-3d-objects` |
| `diffueraser` | 手部移除/inpainting | `/code/ArtHOI-4D-Reconstruction/third_party/diffueraser` |
| `arthoi` | CoTracker、部分 ArtHOI/几何工具 | `/code/ArtHOI-4D-Reconstruction` |
| `particulate` | Particulate 部件/关节预测 | `/code/particulate` |
| `egoforce` | EgoForce 手/手臂重建和 Viser | `/code/EgoForce` |
| `lingbot-depth` | 可选 learned depth fallback/QC | `/code/ArtHOI-4D-Reconstruction` |
| `base` | 文件检查、轻量 manifest 工具 | 项目脚本 |

### 5.1 基础检查

```bash
conda env list
nvidia-smi
python --version
```

建议版本基线：

```text
Python 3.10/3.11
PyTorch 与 NVIDIA driver/CUDA 版本匹配
GPU 显存按模型要求准备，SAM3D/Qwen3-VL/DiffuEraser 需要较大显存
```

### 5.2 环境导出

在原机器上导出：

```bash
for env in qwen3vl sam3d-objects diffueraser arthoi particulate egoforce lingbot-depth; do
  conda env export -n "$env" --from-history > "environment/${env}.yml"
  conda run -n "$env" pip freeze > "environment/${env}-pip.txt"
done
```

恢复环境：

```bash
conda env create -f environment/qwen3vl.yml
conda env create -f environment/sam3d-objects.yml
conda env create -f environment/diffueraser.yml
conda env create -f environment/arthoi.yml
conda env create -f environment/particulate.yml
conda env create -f environment/egoforce.yml
```

`--from-history` 只记录 Conda 明确安装的包，因此 pip freeze 文件也必须一并保留。不要把整个 `/opt/conda/envs` 目录上传 GitHub。

### 5.3 第三方仓库精确版本

复现实验时不要只记录仓库名，应 checkout 到当前验证过的 commit。机器可读锁文件见 `environment/third_party.lock.json`，完整审计见 `docs/DEPENDENCY_AUDIT_NATIVE36_LEFT.md`。

| 仓库 | commit |
|---|---|
| Qwen3-VL | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` |
| SAM2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| SAM3D Objects | `f91db411c50efee93d8db7aeb323885650f6f722` |
| DiffuEraser | `8e6f279ac7531e27ad1849c6f8dab5372a8597e7` |
| ArtHOI-4D-Reconstruction | `33cfbb6367afb6d076122d9aa2c2a5f9c467781c` |
| Particulate | `9cf7f6116ee5ca2e1bf54dbc798d220810bb47ca` |
| EgoForce | `79fb146499c93979e118eba136d6538df29685ff` |
| CoTracker | `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` |
| FoundationPose | `a1b694b83e633c2cb6115b9063d940a687759392` |
| SpatialMP4 | `da721695829a0e947930f9b7f1a254dd695fb794` |

### 5.4 系统和原生依赖

除了 Python/Conda 包，还需要 `git`、`ffmpeg`、`ffprobe`、`cmake`、`ninja`、C/C++ 编译器和兼容的 NVIDIA/CUDA 工具链。Stage 00 依赖 SpatialMP4 Python 原生扩展；当前回退路径为 `/code/SpatialMP4/build_spatialmp4_patched/python`。该扩展必须在目标机器按运行时 OpenCV/动态库重新构建，并用 `ldd spatialmp4*.so` 检查共享库。

公共 Python import 参考 `environment/requirements-common.txt`，但模型环境必须分别恢复；当前没有一个 Conda 环境覆盖全流程所有阶段。

### 5.5 安装自检

```bash
python tools/check_native36_installation.py
```

该脚本检查交付代码、系统工具、第三方仓库 commit、主要 checkpoint 和 SpatialMP4 扩展。加 `--strict-imports` 可严格检查当前 Conda 环境中的公共 Python 包。

## 6. 权重清单

以下是当前机器中确认存在、且本流程可能需要的权重。权重应从各项目官方发布页或内部模型仓库下载，不能未经许可直接上传 GitHub。

### 6.1 必需权重

| 模块 | 当前路径 | 权重 |
|---|---|---|
| Qwen3-VL | `/code/models/Qwen3-VL-8B-Instruct` | 4 个 `model-*.safetensors` 分片及 tokenizer/config |
| SAM2 | `/code/ArtHOI-4D-Reconstruction/third_party/sam2` | `checkpoints/sam2.1_hiera_large.pt` |
| SAM3D Objects | `/code/sam-3d-objects` | `checkpoints/hf/ss_generator.ckpt`、`slat_generator.ckpt`、decoder/encoder 文件 |
| DiffuEraser | `/code/ArtHOI-4D-Reconstruction/third_party/diffueraser` | Stable Diffusion、BrushNet、ProPainter、motion adapter 等其官方权重 |
| Particulate | `/code/particulate` | `PartField/model/model_objaverse.ckpt` 及 Particulate 官方权重 |
| EgoForce | `/code/EgoForce` | `_DATA/model_weights.pth` |
| CoTracker | `/code/ArtHOI-4D-Reconstruction/third_party/co-tracker` | `checkpoints/scaled_offline.pth` 或当前脚本要求的版本 |
| FoundationPose | `/code/ArtHOI-4D-Reconstruction/third_party/foundationpose` | `weights/2024-01-11-20-02-45/model_best.pth` |

### 6.2 可选权重

| 模块 | 用途 |
|---|---|
| Video-Depth-Anything | VDA depth 对照/某些旧分支；native36 的原生 metric depth 主线不应依赖它 |
| LingBot-Depth | learned depth 预览或 fallback，不替代可用原生 metric depth |
| WiLoR | 手部对照实验，不是当前 accepted 手部主线 |
| TRELLIS.2/Hunyuan3D | 其他 mesh baseline，不是 native36 当前必需主线 |

### 6.3 权重安装后的验证

```bash
test -f "$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt"
test -f "$SAM3D_ROOT/checkpoints/hf/ss_generator.ckpt"
test -f "$EGOFORCE_ROOT/_DATA/model_weights.pth"
test -f "$PARTICULATE_ROOT/PartField/model/model_objaverse.ckpt"
```

建议为每个权重保存 SHA256：

```bash
sha256sum /path/to/checkpoint > checkpoints.sha256
sha256sum -c checkpoints.sha256
```

## 7. 目录和环境变量

同事机器上不要使用 `/code` 绝对路径。建议设置：

```bash
export RECON_ROOT=/path/to/vlm-sam2-recon
export DATA_ROOT=/path/to/spatial_data
export MODEL_ROOT=/path/to/models
export QWEN_ROOT=/path/to/Qwen3-VL
export SAM2_ROOT=/path/to/sam2
export SAM3D_ROOT=/path/to/sam-3d-objects
export DIFFUERASER_ROOT=/path/to/diffueraser
export ARTHOI_ROOT=/path/to/ArtHOI-4D-Reconstruction
export PARTICULATE_ROOT=/path/to/particulate
export EGOFORCE_ROOT=/path/to/EgoForce
export WORKSPACE=$RECON_ROOT/run_mixed_<run_id>
```

变量名使用不含空格的标准形式：

```bash
export ARTHOI_ROOT=/path/to/ArtHOI-4D-Reconstruction
```

建议把这些变量写入本地未提交文件：

```text
.env.local
```

不要把真实服务器路径、API key 或模型下载 token 提交仓库。

## 8. 数据准备

输入需要包括：

```text
raw_video.mp4
SpatialMP4 export/
├── frames.csv
├── depth_meters_npy/
├── pose/head_pose.csv
└── pose/head_pose.jsonl
```

初始化工作区：

```bash
python scripts/init_mixed_interaction_workspace.py \
  --run-id mixed_<run_id> \
  --workspace-dir "$WORKSPACE" \
  --video-path "$DATA_ROOT/input.mp4" \
  --spatial-export-root "$DATA_ROOT/input_spatialmp4_export" \
  --tracker-fps 5.143
```

准备 native RGB-D 时间轴：

```bash
python scripts/mixed_stage00_prepare.py \
  --workspace "$WORKSPACE" \
  --video "$DATA_ROOT/input.mp4" \
  --spatial-export "$DATA_ROOT/input_spatialmp4_export" \
  --eye left \
  --target-fps 5.143
```

执行后必须检查：

```text
outputs/00_rgb_frames/stage00_manifest.json
outputs/00_rgb_frames/camera.json
outputs/00_rgb_frames/timeline.csv
```

重点确认：

- 选择的 eye；
- native frame count；
- RGB/depth 时间戳匹配；
- `frame0_*_camera_opencv_rdf` 坐标约定。

## 9. 推荐执行顺序

```text
00 native RGB-D
  -> 00 pose refinement
  -> 01 Qwen3-VL events
  -> 02 SAM2 hand masks
  -> 03 DiffuEraser hand removal
  -> 04 SAM2 object/part masks
  -> 05 SAM3D frame-0 meshes
  -> 06 native metric depth
  -> 07 frame-0 RGB-D alignment
  -> 08 rigid object tracking
  -> Particulate articulated parts/joint
  -> articulated fixed-axis tracking + lid/base contact
  -> rigid-on-articulated support refinement
  -> event-gated rigid pose
  -> EgoForce hand/arm C0 reconstruction
  -> hand-object contact optimization
  -> Viser playback
```

Stage 编号是历史标签，不应单独作为依赖顺序。Particulate 输出通常要先于 articulated tracking 使用。

## 10. 各阶段使用方法

### Stage 01：VLM 事件分析

让 Qwen3-VL 查看完整 selected-eye 时间轴，输出：

- 操作对象；
- rigid/articulated 类别；
- 事件区间；
- hand side；
- frame 0 全局 box；
- 终态确认信息。

输出：

```text
outputs/01_vlm/mixed_interactions.json
```

VLM 结果只做事件路由，不直接替代 RGB-D pose。

### Stage 02：SAM2 手部 mask

该阶段需要人工正/负点提示。流程是：

```text
frame 0 或 anchor frame 点选
  -> SAM2 传播
  -> 全时间轴 mask
  -> 人工/QC 检查
```

输出：

```text
outputs/02_hand_masks/
```

### Stage 03：DiffuEraser

输入 RGB 视频和手部 mask，输出去手视频。去手 RGB 只用于物体分割和视觉跟踪，不作为深度真值。

### Stage 04：物体/部件 SAM2 mask

在去手视频上分别生成：

```text
mouse whole-object mask
laptop whole-object mask
laptop screen/lid part mask
laptop base/body mask
```

Articulated part mask 必须在 Stage 10 前人工确认。

### Stage 05–07：mesh 和 frame-0 对齐

```text
frame 0 RGB + frame 0 mask
  -> SAM3D canonical mesh
  -> metric RGB-D ICP
  -> silhouette IoU refinement
  -> aligned mesh in C0
```

输出重点：

```text
outputs/03_sam3d_frame0/
outputs/07_alignment/<object_id>/frame_000000/
```

### Stage 08：鼠标刚体跟踪

鼠标使用 FoundationPose、RGB-D ICP 和事件门控前的跟踪结果。建议保留：

```text
Delta_C0_object_motion.npy
manifest.json
success.npy
icp_accepted.npy
```

### Particulate：部件和关节

以 articulated object 的全局 frame-0 canonical mesh 为输入：

```bash
conda run -n particulate python scripts/mixed_stage12_particulate.py \
  --workspace "$WORKSPACE" \
  --object-id laptop \
  --source-mesh "$WORKSPACE/outputs/03_sam3d_frame0/laptop/mesh_canonical.glb" \
  --alignment-report "$WORKSPACE/outputs/07_alignment/laptop/frame_000000/alignment_report.json" \
  --particulate-root "$PARTICULATE_ROOT" \
  --python-bin /opt/conda/envs/particulate/bin/python
```

输出：

```text
outputs/12_particulate/laptop/parts_C0/part_14.obj
outputs/12_particulate/laptop/parts_C0/part_15.obj
outputs/12_particulate/laptop/joint_axes_C0.json
```

### Stage 10：lid/base 接触角

屏幕被限制为绕固定铰轴的一维旋转。流程：

```text
RGB-D/CoTracker 角度初值
  -> 固定铰轴约束
  -> 排除铰链附近区域
  -> 计算 lid-base clearance
  -> 角度步进搜索首次接触
  -> 二分细化安全接触角
  -> 在安全区间内做 mask IoU 优化
  -> 接触后保持角度
```

本次 native36 运行的关闭事件是帧 `6–15`。关键输出目录：

```text
outputs/10_articulate_tracking_pipeline_contact_integrated/
```

### Stage 11：mouse-on-laptop 支撑约束

鼠标进入放置事件后，系统：

1. 逐帧计算鼠标与 laptop screen `part_14` 的最近距离；
2. 首次低于接触阈值时确定接触帧；
3. 在接触区域选择最近屏幕三角面；
4. 用三角面法线建立支撑平面；
5. 沿法线平移鼠标，使 clearance 达到 `1.5 mm`；
6. 不改变鼠标 mesh 旋转；
7. 事件前固定 frame 0，事件后固定鲁棒终端位姿。

本次输出：

```text
outputs/11_object_object_support_mouse_laptop/support_manifest.json
outputs/11_object_object_support_mouse_laptop/event_gate_manifest.json
outputs/11_object_object_support_mouse_laptop/mouse_poses_support_event_gated.npy
```

本次关键参数：

```text
placement_start_frame = 18
first_contact_frame = 23
contact_threshold = 18 mm
clearance = 1.5 mm
max_correction = 40 mm
```

### EgoForce：手和手臂

```bash
conda run -n egoforce python scripts/rigid_stage09_egoforce.py \
  --workspace "$WORKSPACE" \
  --egoforce-root "$EGOFORCE_ROOT" \
  --egoforce-python /opt/conda/envs/egoforce/bin/python \
  --poses-path "$WORKSPACE/outputs/00_pose_refinement/poses_refined.npz"
```

输出：

```text
outputs/09_egoforce/
outputs/09_egoforce/dynamic_manifest.json
```

核心变换：

```text
p_C0 = T_C0_from_Ct[frame] @ p_Ct
```

### 手物接触优化

EgoForce 输出之后，再使用已有 hand-object optimizer：

```text
EgoForce hand/arm C0 mesh
  -> object SDF / depth point cloud / SAM2 mask
  -> bounded global hand translation
  -> local vertex offsets
  -> contact QC
```

原则：手可以修正，最终鼠标和 laptop 轨迹不能被手物优化修改。

## 11. 最终 Viser

native36 的最终 combined scene 使用：

```text
laptop base/lid：outputs/12_particulate/laptop/parts_C0
laptop angle：outputs/10_articulate_tracking_iou_vda_contact_track0/link_14
mouse mesh：outputs/07_alignment/mouse/frame_000000/sam3d_aligned_C0.glb
mouse pose：outputs/11_object_object_support_mouse_laptop/mouse_poses_support_event_gated.npy
hands：outputs/09_egoforce 或最终 hand-object optimization manifest
```

启动前，先从最终 manifest 复制实际路径并确认存在，然后查看 accepted Viser 参数：

```bash
conda run -n egoforce python scripts/serve_full_scene_viser.py --help
```


## 12. 输出和复现检查

每个 accepted run 至少应保存：

```text
pipeline_state.json
configs/mixed_recon_config.json
configs/rigid_recon_config.json
outputs/00_rgb_frames/stage00_manifest.json
outputs/00_rgb_frames/camera.json
outputs/00_rgb_frames/timeline.csv
outputs/01_vlm/mixed_interactions.json
outputs/07_alignment/<object>/frame_000000/alignment_report.json
outputs/08_*/<object>/manifest.json
outputs/10_*/articulate_tracking_manifest.json
outputs/11_*/support_manifest.json 或 contact manifest
outputs/09_egoforce/dynamic_manifest.json
最终 Viser 使用的 hand/object/pose 路径
```

复现前检查：

```bash
python -m py_compile scripts/*.py
python scripts/<stage>.py --help
find "$WORKSPACE" -type f | sort | head
```

对最终关键文件执行：

```bash
test -f "$WORKSPACE/outputs/00_rgb_frames/camera.json"
test -f "$WORKSPACE/outputs/01_vlm/mixed_interactions.json"
test -f "$WORKSPACE/outputs/12_particulate/laptop/joint_axes_C0.json"
test -f "$WORKSPACE/outputs/09_egoforce/dynamic_manifest.json"
```

## 13. GitHub 交付边界

提交到 GitHub：

```text
scripts/
docs/
configs/*.example.json
environment/*.yml
tools/
environment/third_party.lock.json
environment/requirements-common.txt
docs/DEPENDENCY_AUDIT_NATIVE36_LEFT.md
README.md
.gitignore
```

不提交：

```text
原始视频
SpatialMP4 export
完整 run_* 目录
模型权重
大规模逐帧 RGB/depth/mask
/tmp 缓存
```

建议额外提供：

```text
examples/native36_left/manifest_snapshot/
examples/native36_left/preview.mp4
artifacts/native36_left.sha256
```

完整数据放到内部对象存储或共享盘，用下载脚本恢复到 `$DATA_ROOT` 和 `$WORKSPACE`。

## 14. 已知限制

- Stage 02、Stage 04 的 SAM2 prompt 仍需要人工确认。
- 外部模型仓库和 checkpoint 需要根据各自许可证单独下载。
- 当前脚本存在历史路径默认值，正式上传前应改为环境变量或配置文件。
- `pipeline_state.json` 是运行记录，不等价于全自动调度器；建议后续增加统一 `run_native36_pipeline.py`。
- native36 当前部分旧实验目录仍存在，交付时只保留 accepted 输出和必要 QC manifest。
