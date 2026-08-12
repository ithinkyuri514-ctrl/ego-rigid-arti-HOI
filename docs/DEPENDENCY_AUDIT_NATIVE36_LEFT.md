# native36-left 依赖审计

审计对象：`run_mixed_20260802_142359_native36_left` 对应的代码交接包。

## 结论

代码不能只复制 mixed 入口脚本。入口会继续调用 rigid 后端，并导入 `vlm_sam2_recon/` 中的共享模块。交付时严格按 `REQUIRED_CODE_NATIVE36_LEFT.txt` 保留 34 个 accepted 主线脚本，并保留 `vlm_sam2_recon/` 共享源码包（排除 `__pycache__`）。

即使代码齐全，GitHub 仓库本身也不能独立完成全流程：外部模型仓库、模型权重、SpatialMP4 原生扩展、FFmpeg 和多个阶段专用 Conda 环境仍需单独安装。

## 本次发现并补齐的直接后端

以下脚本此前未出现在最小清单中，但会被 mixed wrapper 直接导入或启动：

- `scripts/rigid_stage00_prepare.py`
- `scripts/rigid_stage04_object_mask_server.py`
- `scripts/rigid_stage04_object_masks.py`
- `scripts/rigid_stage08_track_pose.py`
- `scripts/run_sam3d_objects_prompt.py`

另外，交接文档中实际使用的 FoundationPose、EgoForce 导出、支撑/事件门控、adaptive hand-object contact 和最终 Viser 脚本也已统一加入清单。Hunyuan、TRELLIS、WiLoR、PhysX、旧 Stage 11 接触链及诊断可视化均已从 Git 跟踪中移除。

## 外部仓库版本锁定

| 仓库 | URL | commit |
|---|---|---|
| Qwen3-VL | `https://github.com/QwenLM/Qwen3-VL.git` | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` |
| SAM2 | `https://github.com/facebookresearch/sam2.git` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| SAM3D Objects | `https://github.com/facebookresearch/sam-3d-objects.git` | `f91db411c50efee93d8db7aeb323885650f6f722` |
| DiffuEraser | `https://github.com/lixiaowen-xw/diffueraser` | `8e6f279ac7531e27ad1849c6f8dab5372a8597e7` |
| ArtHOI-4D-Reconstruction | `https://github.com/hitcs-zikaiwang/ArtHOI-4D-Reconstruction.git` | `33cfbb6367afb6d076122d9aa2c2a5f9c467781c` |
| Particulate | `https://github.com/RuiningLi/particulate.git` | `9cf7f6116ee5ca2e1bf54dbc798d220810bb47ca` |
| EgoForce | `https://github.com/dfki-av/EgoForce.git` | `79fb146499c93979e118eba136d6538df29685ff` |
| CoTracker | `https://github.com/facebookresearch/co-tracker` | `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` |
| FoundationPose | `https://github.com/NVlabs/FoundationPose` | `a1b694b83e633c2cb6115b9063d940a687759392` |
| SpatialMP4 | `https://github.com/Pico-Developer/SpatialMP4` | `da721695829a0e947930f9b7f1a254dd695fb794` |

机器可读版本位于 `environment/third_party.lock.json`。SpatialMP4 另需本机编译，因此单独列在系统依赖中。

## Python 依赖

主流程源码静态扫描得到的公共包包括：

```text
numpy scipy opencv-python Pillow trimesh open3d
PyYAML yacs viser requests torch torchvision
```

模型仓库还会带入各自的专用依赖，例如 `transformers`、`accelerate`、`diffusers`、CUDA 扩展等。不要用一个 `requirements.txt` 强行合并全部模型环境；当前机器上的环境本来就是按阶段拆分的。

完整保留 `vlm_sam2_recon/` 时，静态扫描还会看到 `trellis2`、`app_local`、`o_voxel`、`nvdiffrast`、`pytorch3d` 等导入。这些属于 TRELLIS/Hunyuan/旧实验模块，并非 accepted native36 执行路径的必需依赖；只有运行对应可选分支时才安装。

| Conda 环境 | 主要阶段 | 说明 |
|---|---|---|
| `qwen3vl` | VLM 事件理解 | Qwen3-VL/Transformers 栈 |
| `sam3d-objects` | SAM2/SAM3D、常用几何 | 有 Torch、OpenCV、Open3D、trimesh 等 |
| `diffueraser` | 去手/inpainting | Diffusers/Accelerate 栈 |
| `arthoi` | CoTracker、FoundationPose、几何优化 | 几何依赖较全，并含 Viser |
| `particulate` | Particulate/PartField | 独立模型环境 |
| `egoforce` | 手和手臂重建 | EgoForce 及其可视化依赖 |
| `lingbot-depth` | 可选 learned-depth | 不是 native36 metric depth 主线 |

`environment/requirements-common.txt` 只是公共导入参考，不替代各第三方仓库官方安装步骤和导出的 Conda YAML/pip freeze。

## 系统与原生依赖

必需或强烈建议安装：

- NVIDIA 驱动，以及与各环境 PyTorch/CUDA 扩展兼容的 CUDA 工具链。
- `git`、`ffmpeg`、`ffprobe`。
- `cmake`、`ninja`、C/C++ 编译器，用于 SpatialMP4 和模型 CUDA/C++ 扩展。
- SpatialMP4 Python 扩展。当前机器路径为 `/code/SpatialMP4/build_spatialmp4_patched/python/spatialmp4*.so`。

SpatialMP4 必须针对运行环境中可用的 OpenCV/动态库重新编译。若导入失败，使用 `ldd spatialmp4*.so` 检查缺失共享库；不能只复制 `.so` 就假设另一台机器可用。

## 权重边界

代码仓库不包含权重。至少需要 Qwen3-VL、SAM2、SAM3D Objects、DiffuEraser、Particulate/PartField、EgoForce、CoTracker 和 FoundationPose 权重。路径与用途见主交接文档第 6 节。WiLoR、TRELLIS、Hunyuan3D 和 learned-depth baseline 不属于 accepted native36 主线。

## 自动检查

在项目根目录运行：

```bash
python tools/check_native36_installation.py
```

该命令检查交付代码、系统命令、第三方仓库 commit、主要 checkpoint 和 SpatialMP4 扩展。公共 Python import 默认只报告当前环境缺项，因为不同阶段使用不同环境；若要把当前环境当作公共几何环境严格检查：

```bash
python tools/check_native36_installation.py --strict-imports
```

自检通过只说明安装项存在，不等于模型 CUDA 扩展已在当前 GPU 上完成端到端推理验证。最终还需按主文档执行小样本 smoke test。
