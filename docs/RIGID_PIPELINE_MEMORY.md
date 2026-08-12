# Generic Rigid Hand–Object 4D Reconstruction Pipeline

> 本文只记录可复用于不同数据序列的刚体重建 pipeline、坐标约定和质量门槛。
> 任何具体视频路径、帧号、接触区间、接触手指、阈值拟合结果和 QC 数值，都必须保存在
> 对应 run workspace 中，不能成为本 pipeline 的默认参数。
>
> 最后更新：2026-07-18。

## 1. 任务定义

输入是一段带相机/head pose、稀疏 metric depth 的第一视角手物交互视频；目标是恢复：

- 单个刚体的 metric mesh；
- 刚体逐帧 6DoF pose；
- 手和手臂逐帧 mesh；
- 手物接触状态与深度约束下的接触优化结果；
- 统一参考系下的 RGB-D 点云、手和物体动态可视化。

主线为：

```text
RGB / timestamp / camera pose / sparse metric depth
  -> VLM 理解事件、目标物体和可用关键帧
  -> SAM2 手 mask -> DiffuEraser 去手
  -> 交互式 SAM2 物体 mask 及全帧传播
  -> Hunyuan3D canonical object mesh
  -> Video Depth Anything dense depth + sparse metric-depth calibration
  -> metric object observation -> Sim3/ICP mesh alignment
  -> CoTracker3 object tracks -> RGB-D 3D trajectories -> rigid SE(3)
  -> EgoForce hand/arm reconstruction
  -> depth-first per-finger contact inference and optimization
  -> Viser synchronized playback
```

## 2. 不可破坏的全局约定

1. 目标物体是一个刚体，每帧只有一个 SE(3)；不引入 articulated part、hinge 或形变。
2. 主时间轴由选定的单目 RGB 帧及其真实 timestamp 定义，不能靠猜测 FPS 对齐传感器。
3. 固定参考系为第一帧所用相机：`C0 = frame0_camera`，采用 OpenCV RDF（X 右、Y 下、Z 前）。
4. object mesh/pose、hand/arm、tracks 和所有 RGB-D 点云最终都必须显式转换到 `C0`。
5. 矩阵统一命名为 `T_<dst>_from_<src>`，点变换语义为
   `p_dst = T_dst_from_src @ p_src`。
6. VDA 输出是相对/仿射不确定深度，未经真实深度标定不得当作米制深度。
7. DiffuEraser 结果只用于补外观、减小遮挡，不作为几何真值。
8. 物体 tracking 和 pose 先独立完成；接触优化不得任意修改已验证的 object pose/scale。
9. 所有数据相关选择都由本次运行的观测和 QC 决定，不使用某个历史序列的固定帧号或区间。

## 3. 输入与运行目录契约

每个数据序列建立独立 workspace，至少保存：

- 原始视频或已解码的单目 RGB；
- RGB timestamp；
- 高频 head/camera pose 及外参；
- 稀疏真实 depth、depth timestamp、内参与 depth-to-RGB 外参；
- 本次运行配置、阶段状态、日志和 QC；
- 所有模型 checkpoint/API 配置的可追溯信息。

推荐阶段目录：

```text
outputs/00_rgb_frames/
outputs/01_vlm/
outputs/02_hand_masks/
outputs/03_diffueraser/
outputs/04_object_masks/
outputs/05_hunyuan_mesh/
outputs/06_dense_depth/
outputs/07_alignment/
outputs/08_tracking/
outputs/09_egoforce/
outputs/10_visualization/
outputs/11_contact_optimization/
```

每个 workspace 可以有自己的 `RUN_RECORD.md` 或等价记录，但其中内容只是该数据实例，
不能反向写成项目级默认参数。

## 4. 时间同步与相机运动抵消

对每个 RGB frame 保存真实 timestamp，并按 timestamp 关联：

- camera/head pose：平移线性插值，旋转 quaternion SLERP；
- sparse metric depth：使用真实 timestamp 的最近邻或受控时间窗口；
- VDA dense depth：与 RGB 一一对应；
- hand、object track、mask 和可视化：全部使用同一 RGB timeline。

若采集系统给出 `T_W_from_H(t)` 和相机外参 `T_H_from_C`，则：

```text
T_W_from_C(t)  = T_W_from_H(t) @ T_H_from_C
T_C0_from_C(t) = inverse(T_W_from_C(0)) @ T_W_from_C(t)
```

当前相机坐标中的几何统一转到 `C0`：

```text
p_C0 = T_C0_from_C(t) @ p_Ct
```

若 depth 使用独立相机 `D_t`，还需先应用标定外参：

```text
T_C_from_D = inverse(T_H_from_C) @ T_H_from_D
p_C0 = T_C0_from_C(t) @ T_C_from_D @ p_Dt
```

外参方向不能只靠字段名猜测；应以 RGB-D 投影覆盖率、边缘一致性和静态背景稳定性验证。
头部 pose 的抵消适用于 object tracks、object/depth point clouds、hand/arm 和场景 RGB-D，
不能只对其中一类几何做补偿。

## 5. Stage 00：单目 RGB、相机参数与统一时间轴

- 从双目输入中显式选定左目或右目，禁止把双目拼接图当作单相机图像。
- 按真实 timestamp 下采样到目标处理帧率。
- 输出 RGB、内参、畸变参数、逐帧 `T_C0_from_C(t)` 和统一 timeline。
- 保存原始采样索引和插值来源，保证后续结果可追溯。

质量门槛：

- frame index、timestamp 单调且一一对应；
- 图像尺寸与内参匹配；
- pose 不发生无记录的外推；
- `T_C0_from_C(0)` 数值上接近单位矩阵；
- 静态背景经 pose compensation 后在 `C0` 中近似静止。

## 6. Stage 01：VLM 事件、目标与关键帧理解

VLM 输出结构化信息：

- 交互事件描述；
- 与手交互的单个目标刚体；
- 目标是否满足 rigid 假设；
- 遮挡少、轮廓完整、适合分割/mesh/alignment 的候选关键帧；
- 接触分析阶段的逐手指语义候选。

VLM 是候选生成器，不是几何裁判。关键帧和接触判断必须再结合 mask、depth、运动和
投影一致性验证。Prompt 必须 object-agnostic，不硬编码物体类别、固定手、接触侧或抓取动作。

## 7. Stage 02–04：手移除与物体分割

### 手 mask 与 DiffuEraser

- 用 SAM2 或等价视频分割得到逐帧 hand mask；
- mask video 与输入视频的帧数、尺寸和 FPS/timestamp 对齐；
- 送入 DiffuEraser 得到无手视频；
- 保留原始 RGB 和 mask，后续几何验证仍回到真实观测。

### 交互式 SAM2 物体 mask

- 提供浏览器/交互入口，让用户在可靠帧上用正负点或 bbox 指定目标；
- 在去手视频上初始化并双向传播；
- 输出逐帧 binary mask、原 RGB overlay、crop 和质量统计；
- 遮挡区可利用传播和 inpainting 保持完整物体假设，但 visible-region 几何残差只使用真实可见区域。

质量门槛：无异常空 mask、无身份漂移、面积/质心跳变可解释，并人工抽查首/中/尾、
强遮挡和重新显露帧。

## 8. Stage 05：Hunyuan3D canonical mesh

- 使用经 SAM2 精确扣图的低遮挡 object crop/mask 作为 prompt；
- 可从多个候选关键帧生成并通过轮廓、外观和几何可用性选择 mesh；
- 保存原始 canonical mesh、纹理、生成参数和 prompt 图；
- 此阶段 mesh 没有可信 metric scale，也不默认位于相机坐标系。

不能用 VLM 粗抠图替代最终 SAM2 prompt；透明、反光或细长结构要人工检查轮廓完整性。

## 9. Stage 06：逐帧 dense metric depth

1. 用 Video Depth Anything 为每个 RGB frame 估计 dense depth。
2. 将稀疏真实 depth 通过外参投影到对应 RGB，并按 timestamp 建立 metric anchors。
3. 在有效、无遮挡且几何一致区域拟合 depth 或 inverse-depth 域的 scale/shift。
4. 对标定参数做稳健时间插值/平滑，得到每帧 metric dense depth。
5. 保存每个 anchor 的有效像素、拟合形式、scale/shift、残差及 held-out 验证。

示意模型：

```text
depth_metric(t) ~= scale(t) * depth_vda(t) + shift(t)
```

具体采用 depth 还是 inverse-depth 形式由本次数据验证决定。无效零深度、边界飞点、动态遮挡
和 RGB-depth 不同步区域不得进入标定。

## 10. Stage 07：mesh metric 对齐到 C0

在可靠关键帧中：

1. 用 object mask 与 metric depth 反投影 observed object point cloud；
2. 根据 mesh 与观测的主轴、轮廓或特征产生多个 Sim3 初值，避免正反面/轴向翻转；
3. 联合优化 metric scale、rotation 和 translation；
4. 依次使用 point-to-point、point-to-plane ICP，并结合 depth residual、silhouette IoU/
   boundary distance 和投影 coverage；
5. 若关键帧不是 frame 0，将结果通过相机 pose 转到 `C0`；
6. 输出 canonical-to-`C0` 变换、metric aligned mesh 和独立 QC。

不能只看 ICP 数值：对称物体可能在反向姿态下仍有较小 Chamfer。必须联合检查 RGB 投影、
深度前后关系、轮廓和可辨识语义方向。

## 11. Stage 08：CoTracker3 + RGB-D 刚体 pose

### 2D tracking

- 在一个或多个可靠 object mask 内均匀采样足量点，覆盖纹理、轮廓和不同深度区域；
- 用 CoTracker3 做长时点跟踪；
- 每帧用 object mask、visibility/confidence 和图像边界过滤观测；
- 在遮挡后重新出现时允许多锚点或分段 track，而不是依赖单一首帧轨迹。

### 3D trajectory filtering

对每个有效 2D track，使用该帧 metric depth 和内参反投影到 `C_t`，再用
`T_C0_from_C(t)` 转到 `C0`。剔除：

- 无效深度和局部深度不连续；
- 相邻帧深度突变或 3D 速度异常；
- 离开 object mask/轮廓过远；
- 长期不满足共同刚体距离约束的轨迹；
- 与多数轨迹估计的 SE(3) 残差持续过大的 outlier。

头部运动必须在 3D trajectory 和所有用于 ICP 的点云上显式抵消。图像中物体因头动而移动
是正常现象；在 `C0` 中静止物体仍明显漂移，才说明 pose、时间同步、外参或 depth 有问题。

### rigid pose estimation

- 用 3D-3D RANSAC/Kabsch 估计相邻或锚点间刚体变换；
- 可用首帧 metric 3D 到逐帧 2D PnP 稳定方向；
- 用 mask 内 depth point cloud 的受限 ICP 小幅细化，不允许 ICP 跳到背景或错误对称解；
- 输出 `T_C0_from_O(t)`、inlier 数、残差、mask coverage、depth residual 和置信度；
- 点不足或退化时显式标记低置信，不静默伪造成功。

接触与否不是 object tracking 的启动条件。物体 pose 主要由物体自身轨迹和 RGB-D 决定，
手物接触只在后续作为物理/接触约束，避免把手的错误运动直接传给物体。

## 12. Stage 09：EgoForce 手和手臂

- 在统一 RGB timeline 上运行 EgoForce；
- 原始手/臂 mesh、joints 和 translation 通常位于逐帧 `C_t`；
- 对每帧应用对应 `T_C0_from_C(t)`，统一转到 `C0`；
- 用 SAM2 hand mask、2D keypoint reprojection 和 metric hand depth 对检测及左右手身份做门控；
- 先做 depth-first 对齐，使手在正确的相机深度和可见投影位置，再进入接触优化。

深度对齐应结合：hand mask 内可见 depth、mesh silhouette/keypoint reprojection、时间平滑和
人体骨骼先验。不能为了接触而牺牲明显可靠的深度与 2D 观测。

## 13. Stage 11：通用逐手指接触融合与优化

### 接触状态推断

逐手指融合以下证据：

- object-agnostic VLM 语义候选（低权重）；
- metric finger-to-object surface gap；
- object-local relative finger speed；
- hand/object mask overlap；
- object motion evidence；
- object-over-hand 的遮挡顺序。

遮挡顺序可帮助判断前/后侧：若手指投影位于 object mask 内、但在 visible hand mask 中消失，
则支持手指位于物体后侧。该判断来自观测，不在 prompt 中显式指定“接触后面”。

firm-contact 应要求多根手指、连续多帧和 metric 几何共同支持；所需手指数、连续帧数和阈值
必须按数据质量配置并写入 run record，不能固化某次运行的结果。

### 优化策略

1. 从证据最可靠的 firm-contact anchor 附近建立短 seed window；
2. 每帧根据选定接触侧重新生成 metric object-surface targets；
3. 优化 hand depth/translation/允许的局部 pose，使相关手指区域至少有合理表面接触；
4. 从 seed 向前、向后逐帧传播，并使用时间平滑和观测置信度控制改变量；
5. object pose 和 scale 保持冻结；
6. 非接触帧不施加强制吸附，手松开后不能继续通过接触 loss 拖动物体或粘住手指。

使用“手指区域到同侧表面”的约束，避免把固定 hand vertex 永久钉在单一 object-local 点。
碰撞 proxy/SDF 仅用于近似防穿透；最终视觉 mesh 仍应做双向表面采样、signed distance 或
triangle-level collision QC，以发现 proxy 未覆盖的明显穿模。

## 14. Stage 10/最终 Viser 可视化

Viser 至少提供独立开关：

- object mesh；
- raw/depth-aligned/final hand/arm；
- RGB camera/frustum；
- scene RGB-D point cloud；
- object depth point cloud；
- raw/filtered tracks；
- contact targets、contact fingers 和 QC overlays。

所有动态帧数据在启动时预加载；播放时用原子化 visible 切换或只更新 transform，禁止逐帧
remove/recreate 导致闪烁。大型 object mesh 常驻一份，逐帧只更新 SE(3)，避免复制网格占用
大量内存。

## 15. 质量门槛

每阶段必须产出可检查的诊断，而不是只产出最终动画：

- Timeline：单目尺寸、timestamp、pose 覆盖和插值来源正确；
- Masks：无身份漂移和不可解释跳变，关键遮挡帧有人审查；
- Mesh：prompt mask 正确，mesh 可渲染，拓扑/方向满足后续对齐；
- Depth：稀疏 anchor 拟合与 held-out residual 可追踪；
- Alignment：metric scale、silhouette、depth、ICP 多指标共同通过；
- Tracking：轨迹剔除原因、SE(3) inliers/residual/confidence 完整；
- Coordinate：静态背景和静止物体在 `C0` 中稳定；
- Hand：深度、2D 投影、mask 和时间连续性共同验证；
- Contact：接触只出现在证据支持的区间，松开后无粘连，前后侧与深度/遮挡一致；
- Collision：同时报告 proxy QC 与 visual-mesh QC，不能把 proxy 无穿透等同于真实 mesh 无穿模；
- Visualization：全部元素在同一 timeline 和 `C0`，开关独立且播放不闪烁。

阈值和最终数值均属于 run-specific 配置/结果。项目级 pipeline 只规定需要检查什么，
不保存某条数据的“通过数值”。

## 16. 每次新数据的执行检查表

- [ ] 建立独立 workspace 和 run record，不复制历史数据的帧号/区间。
- [ ] 审计单/双目布局、真实 timestamp、depth/pose 数量和外参方向。
- [ ] 建立统一 RGB timeline，并生成逐帧 `T_C0_from_C(t)`。
- [ ] VLM 识别目标与关键帧，但用几何和 mask 复核。
- [ ] SAM2 手 mask、DiffuEraser、交互式 SAM2 object mask 完成并人工检查。
- [ ] Hunyuan3D prompt 使用精确 object mask/crop。
- [ ] VDA 每帧 depth 已由本序列真实 metric depth 标定。
- [ ] mesh 在 `C0` 中通过方向、尺度、深度和轮廓联合对齐。
- [ ] CoTracker3 3D 轨迹已过滤，且所有点云/轨迹完成 head-pose compensation。
- [ ] object SE(3) 有逐帧置信度与失败状态。
- [ ] EgoForce hand/arm 已通过 depth 和 2D 观测对齐并转到 `C0`。
- [ ] 接触由逐手指多证据自动推断，未硬编码物体、手指、接触侧或固定帧。
- [ ] 接触优化不改变已验证 object pose/scale，松开后不粘连。
- [ ] Viser 预加载动态数据，并能独立显示 mesh、RGB、RGB-D、tracks、hand 和 contact。
