# GeoReward 方案 A 执行计划：基于 DA3 的几何运动一致性 Reward

## 0. 文档定位

本文档给出方案 A 的完整工程规划和参考代码，目标是在不引入视觉质量 Reward 的前提下，使用 Depth Anything 3（DA3）输出的深度、相机位姿和置信度，评价 Wan2.2 生成视频的三维运动一致性。

本方案采用标准、完整的 Best-of-N 流程：

```text
输入首帧 + Prompt
        ↓
Wan2.2 完整生成 N 个候选视频
        ↓
每个候选完整解码为 81 帧
        ↓
均匀抽取关键帧并执行一次 DA3 推理
        ↓
计算几何运动一致性 Reward
        ↓
对 N 个完整候选排序
        ↓
保存全部候选并选出最高分视频
```

本文档不包含：

- 中途早停；
- 提前预测最终 Reward；
- Successive Halving；
- 动态分配去噪步数；
- 图像美学、清晰度或通用视觉质量评价；
- VLM、视频质量模型或学习型 Reward。

严重重影、模糊或漂移只有在它们破坏 DA3 几何预测时才会被间接惩罚。方案 A 的准确定位是：

> DA3-based geometric motion consistency reward for robotic manipulation videos。

---

## 1. 版本范围

方案分为 V1 和 V2 两部分。

### 1.1 V1：Scene + Motion

V1 实现两个核心分量：

```text
R_scene   静态背景和相机几何稳定性
R_motion  动态区域的三维运动与结构一致性
```

V1 的重点是解决当前 Reward 的主要偏差：

```text
全图最低 70% 几何误差由静态背景主导，
导致机械臂几乎不动的视频可能获得高分。
```

V1 使用以下设计：

1. 静态/动态粗分割使用“帧差 + DA3 同像素深度变化率”；
2. `R_scene` 只在静态区域计算；
3. 运动量仅作为 `motion_gate`，判断有没有动；
4. 平滑性只使用一阶位移/速度序列，不计算二阶加速度和三阶 jerk；
5. 结构保持是 `R_motion` 的主力分量；
6. confidence 只用于几何可靠性筛选，不作为独立正向 Reward。

V1 的核心公式为：

```python
R_motion = motion_gate * (
    0.7 * R_shape
    + 0.3 * R_smoothness
)
```

### 1.2 V2：Scene + Motion + Interaction

V2 在 V1 基础上增加：

```text
R_interaction  机器人末端与目标物体之间的相对几何一致性
```

V2 评价：

- 机器人是否接近目标；
- 是否形成稳定接触；
- 接触后物体是否随机械臂共同运动；
- 机器人与物体的相对距离是否突然失稳；
- 是否存在明显的相对深度错误或穿透迹象。

V2 不改变 V1 的静态/动态划分和 `R_motion` 设计。

---

## 2. 显式设计决策

### 2.1 静态/动态区域不再使用重投影误差划分

区域划分只使用：

```text
RGB 帧差
+
DA3 同一像素坐标的深度变化率
```

重投影误差只用于后续 `R_scene`，不参与 static/dynamic mask 的生成，从而避免：

```text
因为几何误差低而被划为静态，
随后又只在这些低误差像素上评价几何误差。
```

### 2.2 当前基线采用 OR 组合

本文严格采用显式规则：

```python
dynamic_mask = image_dynamic | depth_dynamic
static_mask = ~dynamic_mask
```

其含义是高召回：只要 RGB 或深度中的任一信号检测到变化，就先把该区域排除出静态背景。

需要明确：

```text
OR：任一信号变化就判为动态，高召回但可能增加误检。
AND：两个信号同时变化才判为动态，低误检但可能漏掉真实运动。
```

“图像变了但深度没变就排除、深度变了但图像没变就排除、只有两者都变才是运动”的文字逻辑对应 `AND`，不是本计划采用的 `OR`。当前实现以用户明确给出的 `OR` 代码为准；`AND` 仅作为后续消融实验，不作为 V1 默认行为。

### 2.3 运动量只做门控

运动量不进入加权平均，只判断视频是否有足够运动：

```python
motion_gate = min(1.0, motion_energy / min_motion_threshold)
```

完全静止的视频即使结构稳定、速度异常度低，也会得到接近 0 的 `R_motion`。

### 2.4 平滑性只使用一阶差分

不计算：

- 二阶差分加速度；
- 三阶差分 jerk。

只计算相邻关键帧之间的三维位移，并从速度序列中寻找极端异常值。

### 2.5 结构保持是主力指标

`R_shape` 暂定权重 0.7，`R_smoothness` 暂定权重 0.3：

```python
R_motion = motion_gate * (
    0.7 * R_shape
    + 0.3 * R_smoothness
)
```

### 2.6 DA3 confidence 不是视频质量分

confidence 只用于：

- 排除 DA3 不可靠像素；
- 给深度和重投影误差提供可靠性权重；
- 计算有效覆盖率。

不再使用：

```python
total += 0.15 * conf.mean()
```

---

## 3. 总体代码结构

建议保持项目直接、可读，不做过多抽象。

```text
geo_reward/
├── da3_reward.py          # DA3 调用、R_scene、总 Reward
├── region_masks.py        # 帧差 + 深度变化率 mask
├── motion_reward.py       # motion_gate、R_shape、R_smoothness
├── interaction_reward.py  # V2：R_interaction
├── bon_pipeline.py        # 完整 N 候选生成与排序，流程不变
└── utils.py               # 张量/PIL 转换、抽帧
```

如果希望减少文件数量，`region_masks.py` 和 `motion_reward.py` 可以先合并进 `da3_reward.py`。V2 的交互逻辑建议单独放置，避免主类继续膨胀。

建议增加轻量配置：

```python
from dataclasses import dataclass


@dataclass
class GeometryRewardConfig:
    # Region masks
    image_diff_threshold: float = 15.0
    image_vote_ratio: float = 0.30
    depth_change_threshold: float = 0.05
    min_component_area_ratio: float = 0.002
    morph_kernel_size: int = 3

    # Motion
    min_motion_threshold: float = 0.01
    smooth_quantile: float = 0.95
    tau_smooth: float = 2.0
    tau_shape: float = 0.10
    motion_shape_weight: float = 0.70
    motion_smooth_weight: float = 0.30

    # Scene
    tau_scene_proj: float = 0.05
    tau_scene_anchor: float = 0.05
    scene_proj_weight: float = 0.60
    scene_anchor_weight: float = 0.40

    # V1 total
    total_scene_weight: float = 0.30
    total_motion_weight: float = 0.70

    # V2 total
    total_interaction_weight: float = 0.30
```

这些默认值只是让代码可运行的初始值，正式数值需要通过实验标定。

---

# Part I：V1 Scene + Motion

## 4. V1 输入与 DA3 输出

### 4.1 输入帧

当前 `GeoRewardBoN` 会：

1. 将 Wan 输出 `(3, T, H, W)` 转换成 PIL 帧；
2. 从 81 帧均匀抽取最多 20 帧；
3. 把采样帧送入 `DA3GeoReward.compute_reward()`。

V1 保持这一流程。

### 4.2 使用 DA3 处理后的图像计算帧差

DA3 可能对输入图像 resize。为了保证 RGB 帧差和 DA3 深度严格对齐，应使用：

```python
pred.processed_images  # (N, H_da3, W_da3, 3), uint8
```

而不是直接使用原始 PIL 分辨率。

DA3 输出：

```python
images = pred.processed_images  # (N, H, W, 3)
depths = pred.depth             # (N, H, W)
conf = pred.conf                # (N, H, W)
extrinsics = pred.extrinsics    # (N, 4, 4)
intrinsics = pred.intrinsics    # (N, 3, 3)
```

如果 `processed_images` 不存在，则将 PIL 帧 resize 到 `depths.shape[-2:]`。

参考辅助函数：

```python
import numpy as np
from PIL import Image


def prepare_da3_aligned_images(frames_pil, pred):
    if pred.processed_images is not None:
        return pred.processed_images.astype(np.uint8)

    h, w = pred.depth.shape[-2:]
    images = []
    for frame in frames_pil:
        resized = frame.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        images.append(np.asarray(resized, dtype=np.uint8))
    return np.stack(images, axis=0)
```

---

## 5. V1 静态/动态区域粗分割

## 5.1 数据结构

同时保留两种 mask：

```python
global_static_mask   # (H, W)，整段视频始终稳定的背景，用于 R_scene
global_dynamic_mask  # (H, W)，整段视频出现过运动的扫掠区域
frame_dynamic_masks  # (N, H, W)，每帧局部动态区域，用于 R_motion
frame_static_masks   # (N, H, W)
```

为什么不能只使用一个全局动态 mask：

```text
机械臂从左向右运动后，全局动态 mask 会覆盖整条运动路径。
如果每一帧都使用这个大 mask 计算 3D 质心，
很多当前位置没有机械臂的背景像素也会进入质心，导致轨迹错误。
```

因此：

- `global_static_mask` 用于背景几何评价；
- `frame_dynamic_masks[t]` 用于当前帧的运动点云和形状。

## 5.2 RGB 帧差

灰度转换必须先转成浮点或有符号整数，避免 `uint8` 相减下溢：

```python
import cv2
import numpy as np


def rgb_to_gray(images):
    return np.stack(
        [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images],
        axis=0,
    ).astype(np.float32)


def compute_pair_image_dynamic(gray, threshold=15.0):
    pair_masks = []
    pair_diffs = []
    for t in range(len(gray) - 1):
        diff = np.abs(gray[t + 1] - gray[t])
        pair_diffs.append(diff)
        pair_masks.append(diff > threshold)
    return np.stack(pair_masks), np.stack(pair_diffs)
```

输出：

```python
pair_image_dynamic  # (N-1, H, W), bool
pair_image_diff     # (N-1, H, W), float32
```

跨帧累积证据：

```python
image_motion_ratio = pair_image_dynamic.mean(axis=0)
image_dynamic_global = image_motion_ratio > image_vote_ratio
```

默认：

```python
image_vote_ratio = 0.30
```

即某个像素在超过 30% 的相邻关键帧对中超过帧差阈值，就视为整段视频中的图像动态区域。

## 5.3 DA3 同像素深度变化率

同一像素坐标的相邻帧深度变化率：

\[
E^{depth}_t(u,v)=
\left|
\log
\frac{d_{t+1}(u,v)+\epsilon}
{d_t(u,v)+\epsilon}
\right|
\]

参考实现：

```python
def compute_pair_depth_dynamic(
    depths,
    threshold=0.05,
    eps=1e-6,
):
    pair_changes = []
    pair_masks = []

    for t in range(len(depths) - 1):
        d0 = np.maximum(depths[t].astype(np.float32), eps)
        d1 = np.maximum(depths[t + 1].astype(np.float32), eps)

        change = np.abs(np.log(d1 / d0))
        valid = np.isfinite(change) & (d0 > eps) & (d1 > eps)

        pair_changes.append(np.where(valid, change, np.nan))
        pair_masks.append(valid & (change > threshold))

    return np.stack(pair_masks), np.stack(pair_changes)
```

全局深度变化采用相邻帧对的中位数：

```python
depth_change_median = np.nanmedian(pair_depth_change, axis=0)
depth_dynamic_global = depth_change_median > depth_change_threshold
```

## 5.4 confidence 只作为深度可靠性 mask

当前 DA3 confidence 采用 `expp1`，不使用固定的 `conf > 0.3`。

建议每帧使用分位数：

```python
def build_confident_masks(conf, quantile=0.5):
    if conf is None:
        return None

    masks = []
    for c in conf:
        finite = np.isfinite(c)
        if not finite.any():
            masks.append(np.zeros_like(c, dtype=bool))
            continue
        threshold = np.quantile(c[finite], quantile)
        masks.append(finite & (c >= threshold))
    return np.stack(masks)
```

深度变化证据要求相邻两帧深度都可靠：

```python
pair_confident = confident[:-1] & confident[1:]
pair_depth_dynamic &= pair_confident
```

不可靠深度不应被视为“无变化”，也不应强制判成动态。需要单独记录：

```python
global_unknown_mask
frame_unknown_masks
```

## 5.5 OR 双门槛组合

全局动态区域：

```python
global_dynamic_mask = (
    image_dynamic_global
    | depth_dynamic_global
)
```

全局静态区域：

```python
global_static_mask = (
    ~global_dynamic_mask
    & ~global_unknown_mask
)
```

逐帧动态区域由当前帧前后相邻帧对的证据合并。

对于帧 `t`：

```python
dynamic_from_prev = pair_dynamic[t - 1]  # (t-1, t)
dynamic_from_next = pair_dynamic[t]      # (t, t+1)
frame_dynamic[t] = dynamic_from_prev | dynamic_from_next
```

完整参考实现：

```python
def pair_masks_to_frame_masks(pair_masks):
    # pair_masks: (N-1, H, W)
    n_pairs, h, w = pair_masks.shape
    n_frames = n_pairs + 1
    frame_masks = np.zeros((n_frames, h, w), dtype=bool)

    frame_masks[0] = pair_masks[0]
    frame_masks[-1] = pair_masks[-1]

    for t in range(1, n_frames - 1):
        frame_masks[t] = pair_masks[t - 1] | pair_masks[t]

    return frame_masks


def build_static_dynamic_masks(
    images,
    depths,
    conf=None,
    image_diff_threshold=15.0,
    image_vote_ratio=0.30,
    depth_change_threshold=0.05,
):
    gray = rgb_to_gray(images)

    pair_image_dynamic, pair_image_diff = compute_pair_image_dynamic(
        gray,
        threshold=image_diff_threshold,
    )
    pair_depth_dynamic, pair_depth_change = compute_pair_depth_dynamic(
        depths,
        threshold=depth_change_threshold,
    )

    confident = build_confident_masks(conf, quantile=0.5)
    if confident is not None:
        pair_confident = confident[:-1] & confident[1:]
        pair_depth_dynamic &= pair_confident
        pair_unknown = ~pair_confident
    else:
        pair_unknown = np.zeros_like(pair_depth_dynamic)

    # 用户指定的 OR 组合。
    pair_dynamic = pair_image_dynamic | pair_depth_dynamic

    image_motion_ratio = pair_image_dynamic.mean(axis=0)
    image_dynamic_global = image_motion_ratio > image_vote_ratio

    depth_change_median = np.nanmedian(pair_depth_change, axis=0)
    depth_dynamic_global = depth_change_median > depth_change_threshold

    global_dynamic = image_dynamic_global | depth_dynamic_global
    global_unknown = pair_unknown.mean(axis=0) > 0.5
    global_static = ~global_dynamic & ~global_unknown

    frame_dynamic = pair_masks_to_frame_masks(pair_dynamic)
    frame_unknown = pair_masks_to_frame_masks(pair_unknown)
    frame_static = ~frame_dynamic & ~frame_unknown

    return {
        "global_static": global_static,
        "global_dynamic": global_dynamic,
        "global_unknown": global_unknown,
        "frame_static": frame_static,
        "frame_dynamic": frame_dynamic,
        "frame_unknown": frame_unknown,
        "pair_image_diff": pair_image_diff,
        "pair_depth_change": pair_depth_change,
    }
```

## 5.6 空间后处理

粗 mask 会包含孤立噪点，需要轻量形态学处理：

```python
def clean_binary_mask(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    x = mask.astype(np.uint8)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)
    return x.astype(bool)
```

移除过小连通区域：

```python
def remove_small_components(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    output = np.zeros_like(mask, dtype=bool)
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = True
    return output
```

应用方式：

```python
min_area = int(H * W * min_component_area_ratio)

global_dynamic = clean_binary_mask(global_dynamic)
global_dynamic = remove_small_components(global_dynamic, min_area)

for t in range(N):
    frame_dynamic[t] = clean_binary_mask(frame_dynamic[t])
    frame_dynamic[t] = remove_small_components(frame_dynamic[t], min_area)
```

处理后重新计算静态区域：

```python
global_static = ~global_dynamic & ~global_unknown
frame_static = ~frame_dynamic & ~frame_unknown
```

## 5.7 覆盖率记录

必须记录：

```python
static_ratio = global_static.mean()
dynamic_ratio = global_dynamic.mean()
unknown_ratio = global_unknown.mean()
frame_dynamic_ratios = frame_dynamic.mean(axis=(1, 2))
```

如果静态区域或动态区域过小，相关分数应标记为不可靠，不能自动给高分。

---

## 6. V1 静态场景 Reward：R_scene

## 6.1 逐像素重投影误差

当前 `_project_and_compare()` 直接返回截断均值，需要先拆成：

```python
def _project_error_map(...):
    return {
        "error": error_map,          # (H, W)
        "valid": valid_map,          # (H, W)
        "projected_x": px_map,
        "projected_y": py_map,
    }
```

核心投影代码保持当前实现不变：

```python
K_src_inv = np.linalg.inv(intr_src)
rays = K_src_inv @ pixels_flat
pts_cam_src = rays * depth_src.reshape(-1)[None, :]

R_src = ext_src[:3, :3]
t_src = ext_src[:3, 3]
pts_world = R_src.T @ (pts_cam_src - t_src[:, None])

R_tgt = ext_tgt[:3, :3]
t_tgt = ext_tgt[:3, 3]
pts_cam_tgt = R_tgt @ pts_world + t_tgt[:, None]

proj = intr_tgt @ pts_cam_tgt
px = proj[0] / (proj[2] + 1e-8)
py = proj[1] / (proj[2] + 1e-8)
depth_projected = pts_cam_tgt[2]
```

深度误差也保留：

```python
depth_sampled = self._bilinear_sample(depth_tgt, px, py)

ratio = depth_projected[valid] / depth_sampled[valid]
scale = np.median(ratio)
aligned = depth_projected[valid] / scale

log_error = np.abs(
    np.log(aligned / (depth_sampled[valid] + 1e-8) + 1e-8)
)
```

## 6.2 只在 global_static_mask 上评价

源帧静态 mask：

```python
source_static = global_static_mask.reshape(-1)
```

目标位置的静态 mask：

```python
target_static = self._bilinear_sample(
    global_static_mask.astype(np.float32),
    px,
    py,
) > 0.5
```

最终：

```python
valid_static = valid & source_static & target_static
```

全图 70% 截断不再负责剔除动态区域；动态区域已经由 mask 排除。静态区域内部仍可保留温和 robust trimming，例如保留误差最低 90%，用于排除少量边界噪声：

```python
def robust_region_mean(values, keep_ratio=0.90):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    values = np.sort(values)
    n_keep = max(1, int(len(values) * keep_ratio))
    return float(values[:n_keep].mean())
```

## 6.3 静态 projection 分数

双向投影保持不变：

```text
t → s
s → t
```

得到正误差：

```python
E_scene_proj >= 0
```

映射为 0～1：

```python
R_scene_proj = np.exp(-E_scene_proj / tau_scene_proj)
```

## 6.4 静态 anchor 分数

首帧锚定仍然保留，但只在：

```python
global_static_mask
```

内计算。

得到：

```python
E_scene_anchor >= 0
R_scene_anchor = np.exp(-E_scene_anchor / tau_scene_anchor)
```

## 6.5 R_scene 组合

初始组合：

```python
R_scene = (
    0.60 * R_scene_proj
    + 0.40 * R_scene_anchor
)
```

V1 中 `R_scene` 的职责是防止：

- 背景漂移；
- 桌面和场景结构崩坏；
- 大范围空间几何不一致；
- 相机视角发生不合理变化。

---

## 7. V1 动态三维点云

## 7.1 每帧动态点云

使用 `frame_dynamic_masks[t]`，而不是全局运动扫掠 mask。

```python
def depth_to_world_points(depth, extrinsic, intrinsic):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack(
        [u.reshape(-1), v.reshape(-1), np.ones(h * w)],
        axis=0,
    )

    rays = np.linalg.inv(intrinsic) @ pixels
    points_camera = rays * depth.reshape(-1)[None, :]

    r = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    points_world = r.T @ (points_camera - t[:, None])

    return points_world.T.reshape(h, w, 3)
```

对所有帧：

```python
world_maps = []
dynamic_point_clouds = []

for t in range(N):
    world = depth_to_world_points(
        depths[t],
        extrinsics[t],
        intrinsics[t],
    )
    world_maps.append(world)

    valid = (
        frame_dynamic_masks[t]
        & np.isfinite(world).all(axis=-1)
        & (depths[t] > 1e-6)
    )
    dynamic_point_clouds.append(world[valid])
```

## 7.2 场景尺度归一化

三维位移和点对距离必须对场景尺度归一化，否则不同视频和 DA3 尺度不可比较。

使用首帧静态区域的深度中位数：

```python
scene_scale = np.median(
    depths[0][global_static_mask]
)
scene_scale = max(float(scene_scale), 1e-6)
```

所有运动量除以：

```python
scene_scale
```

---

## 8. V1 运动量门控：motion_gate

## 8.1 鲁棒三维质心

普通平均值容易受少数离群点影响，使用逐坐标中位数：

```python
def robust_centroid(points):
    if points is None or len(points) < 10:
        return None
    return np.median(points, axis=0)
```

得到：

```python
centroids = [
    robust_centroid(points)
    for points in dynamic_point_clouds
]
```

只对相邻两帧都有可靠质心的帧对计算三维位移向量：

```python
velocity_vectors = []
velocity_magnitudes = []

for t in range(N - 1):
    if centroids[t] is None or centroids[t + 1] is None:
        continue

    velocity = (
        centroids[t + 1] - centroids[t]
    ) / scene_scale

    velocity_vectors.append(velocity)
    velocity_magnitudes.append(np.linalg.norm(velocity))
```

## 8.2 motion_energy

运动能量使用一阶位移大小的中位数：

```python
motion_energy = np.median(velocity_magnitudes)
```

不使用总和，避免视频中单次跳变被误认为持续运动。

## 8.3 motion_gate

严格采用：

```python
motion_gate = min(
    1.0,
    motion_energy / (min_motion_threshold + 1e-8),
)
```

行为：

```text
完全静止             → motion_gate ≈ 0
有少量运动           → 0 < motion_gate < 1
达到最低运动要求     → motion_gate = 1
```

`motion_gate` 不进入加权平均，而是乘在 `R_motion` 最外层。

## 8.4 无可靠动态区域

如果有效质心数量不足：

```python
if len(velocity_magnitudes) < 2:
    motion_gate = 0.0
```

同时在日志中记录：

```python
motion_valid = False
```

不能把“无法计算运动”当成“运动非常平滑”。

---

## 9. V1 一阶速度异常：R_smoothness

## 9.1 速度异常度

只使用一阶位移大小序列：

```python
speeds = np.asarray(velocity_magnitudes, dtype=np.float32)
v_median = np.median(speeds)
v_high = np.quantile(speeds, 0.95)

E_smoothness = v_high / (v_median + 1e-8)
```

解释：

```text
正常匀速或缓慢变化：q95 / median 接近 1
单帧瞬移：q95 / median 明显增大
多帧抖动：高分位速度相对中位数升高
```

因为正常视频的 `E_smoothness` 基线约为 1（完美匀速时 q95/median = 1），直接映射 `exp(-1/tau)` 会导致匀速视频的 `R_smoothness` 永远不等于 1（例如 `tau=2` 时基线为 `exp(-0.5) ≈ 0.61`）。虽然 BoN 排序只关心相对大小不受基线偏移影响，但为了：

- 绝对分数可解释（匀速 = 满分 1.0）；
- 后续设阈值或跨实验对比时不需要额外减基线；

V1 默认使用 excess 版本：

```python
E_smoothness_excess = max(E_smoothness - 1.0, 0.0)
R_smoothness = np.exp(-E_smoothness_excess / tau_smooth)
```

行为：

```text
完美匀速：E_smoothness = 1.0 → excess = 0 → R_smoothness = 1.0
轻微波动：E_smoothness = 1.3 → excess = 0.3 → R_smoothness = exp(-0.15) ≈ 0.86
单帧瞬移：E_smoothness = 5.0 → excess = 4.0 → R_smoothness = exp(-2.0) ≈ 0.14
```

原始比值版本（`exp(-E_smoothness / tau_smooth)`）保留为消融对照。

## 9.2 运动方向突变诊断

方向突变可使用相邻一阶速度向量的夹角，不需要计算加速度：

```python
def velocity_turn_angles(vectors, eps=1e-8):
    angles = []
    for a, b in zip(vectors[:-1], vectors[1:]):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < eps or nb < eps:
            continue
        cosine = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        angles.append(np.arccos(cosine))
    return np.asarray(angles)
```

V1 第一版只记录方向异常，不进入 `R_smoothness`，避免同时引入过多未经标定的分量。后续可作为消融项。

---

## 10. V1 结构保持：R_shape

## 10.1 不使用无对应关系的固定点索引

相邻帧动态 mask 内的第 `i` 个点通常不是同一个物理点，不能直接写：

```python
distance(points_t[i], points_t[j])
vs.
distance(points_s[i], points_s[j])
```

否则点的排列变化会被误认为结构变化。

V1 使用不依赖点对应关系的“局部距离分布签名”。

## 10.2 点云采样

每帧从动态点云确定性采样最多 `K` 个点。

**关键：采样前必须先按 3D 坐标排序。** 原因是动态 mask 内像素的排列顺序取决于图像扫描（行优先），帧间 mask 形状的微小变化会导致相同线性索引对应不同物理位置。如果不排序就按线性索引均匀采样，签名的帧间波动可能来自采样位置漂移，而不是真实的结构变化。

按 3D 坐标字典序排序（先 x，再 y，再 z）使得采样点在空间中的覆盖模式与 mask 形状解耦，代价仅为一次 `O(N log N)` 排序。

```python
def uniform_subsample_points(points, max_points=256):
    if len(points) <= max_points:
        return points

    # 按 3D 坐标字典序排序，使采样位置与 mask 扫描顺序解耦
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = points[order]

    indices = np.linspace(
        0,
        len(points) - 1,
        max_points,
    ).round().astype(np.int64)
    return points[indices]
```

## 10.3 局部距离签名

计算采样点之间的两两距离，只保留每个点的局部近邻距离：

```python
def local_shape_signature(points, scene_scale, k_neighbors=8):
    if points is None or len(points) < k_neighbors + 1:
        return None

    points = uniform_subsample_points(points, max_points=256)
    points = points / scene_scale

    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    # 排除自身距离 0。
    distances[distances == 0] = np.inf

    nearest = np.partition(
        distances,
        kth=k_neighbors - 1,
        axis=1,
    )[:, :k_neighbors]

    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) < 10:
        return None

    return np.quantile(
        nearest,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    )
```

签名表达动态点云的局部尺度分布。使用局部近邻而不是全局任意点对，是因为机械臂存在合法关节运动，全局两端距离可能自然改变，而局部结构应该相对稳定。

## 10.4 相邻帧形状变化

```python
def shape_change_error(signature_a, signature_b, eps=1e-8):
    if signature_a is None or signature_b is None:
        return None

    ratio = (
        signature_b + eps
    ) / (
        signature_a + eps
    )
    return float(np.mean(np.abs(np.log(ratio))))
```

对所有相邻关键帧：

```python
shape_signatures = [
    local_shape_signature(points, scene_scale)
    for points in dynamic_point_clouds
]

shape_errors = []
for t in range(N - 1):
    error = shape_change_error(
        shape_signatures[t],
        shape_signatures[t + 1],
    )
    if error is not None:
        shape_errors.append(error)

E_shape = np.median(shape_errors)
```

## 10.5 R_shape

```python
R_shape = np.exp(-E_shape / tau_shape)
```

它主要检测：

- 动态主体局部拉伸；
- 机械臂局部结构突变；
- 手指融合；
- 物体明显变形；
- 动态 mask 对应的三维形状在某些帧突然崩坏。

限制：粗动态 mask 中混入背景会影响局部距离分布，因此必须同时记录 `dynamic_ratio` 和点云有效数量。

---

## 11. V1 R_motion 组合

严格采用：

```python
R_motion = motion_gate * (
    0.7 * R_shape
    + 0.3 * R_smoothness
)
```

参考函数：

```python
def compute_motion_reward(
    dynamic_point_clouds,
    scene_scale,
    min_motion_threshold,
    tau_smooth,
    tau_shape,
    shape_weight=0.7,
    smooth_weight=0.3,
):
    centroids = [
        robust_centroid(points)
        for points in dynamic_point_clouds
    ]

    vectors = []
    speeds = []
    for t in range(len(centroids) - 1):
        if centroids[t] is None or centroids[t + 1] is None:
            continue
        velocity = (
            centroids[t + 1] - centroids[t]
        ) / scene_scale
        vectors.append(velocity)
        speeds.append(np.linalg.norm(velocity))

    if len(speeds) < 2:
        return {
            "reward": 0.0,
            "gate": 0.0,
            "shape": 0.0,
            "smoothness": 0.0,
            "motion_energy": 0.0,
            "smoothness_error": None,
            "shape_error": None,
            "valid": False,
        }

    speeds = np.asarray(speeds, dtype=np.float32)
    motion_energy = float(np.median(speeds))
    motion_gate = min(
        1.0,
        motion_energy / (min_motion_threshold + 1e-8),
    )

    v_median = float(np.median(speeds))
    v_high = float(np.quantile(speeds, 0.95))
    smooth_error = v_high / (v_median + 1e-8)
    smooth_excess = max(smooth_error - 1.0, 0.0)
    r_smooth = float(np.exp(-smooth_excess / tau_smooth))

    signatures = [
        local_shape_signature(points, scene_scale)
        for points in dynamic_point_clouds
    ]
    shape_errors = []
    for a, b in zip(signatures[:-1], signatures[1:]):
        error = shape_change_error(a, b)
        if error is not None:
            shape_errors.append(error)

    if shape_errors:
        shape_error = float(np.median(shape_errors))
        r_shape = float(np.exp(-shape_error / tau_shape))
    else:
        shape_error = None
        r_shape = 0.0

    reward = motion_gate * (
        shape_weight * r_shape
        + smooth_weight * r_smooth
    )

    return {
        "reward": float(reward),
        "gate": float(motion_gate),
        "shape": float(r_shape),
        "smoothness": float(r_smooth),
        "motion_energy": motion_energy,
        "smoothness_error": smooth_error,
        "shape_error": shape_error,
        "valid": True,
    }
```

预期行为：

| 视频情况 | motion_gate | R_shape | R_smoothness | R_motion |
|---|---:|---:|---:|---:|
| 完全静止 | 接近 0 | 高或不可用 | 高或不可用 | 接近 0 |
| 正常平滑运动 | 1 | 高 | 高 | 高 |
| 运动但结构崩坏 | 1 | 低 | 可能高 | 中低 |
| 结构稳定但有瞬移 | 1 | 可能高 | 低 | 中等 |
| 瞬移且结构崩坏 | 1 | 低 | 低 | 低 |

---

## 12. V1 总 Reward

V1 初始组合：

```python
R_total_v1 = (
    0.30 * R_scene
    + 0.70 * R_motion
)
```

这样体现：

```text
R_scene  是全局背景约束；
R_motion 是主要排序依据。
```

如果希望严格避免背景分数补偿完全静止的视频，可以额外乘 motion gate：

```python
R_total_v1 = motion_gate * (
    0.30 * R_scene
    + 0.70 * (
        0.70 * R_shape
        + 0.30 * R_smoothness
    )
)
```

推荐 V1 默认使用第二种形式，因为所有当前机械臂 Prompt 都要求发生动作。此时：

```python
R_motion = motion_gate * motion_quality
R_total_v1 = motion_gate * (
    0.30 * R_scene
    + 0.70 * motion_quality
)
```

这能防止：

```text
motion_gate = 0，
但仅依靠 R_scene 仍获得 0.30 总分。
```

完整 V1 参考流程：

```python
def compute_reward_v1(self, frames_pil):
    pred = self.model.inference(
        frames_pil,
        process_res=self.process_res,
    )

    images = prepare_da3_aligned_images(frames_pil, pred)

    masks = build_static_dynamic_masks(
        images=images,
        depths=pred.depth,
        conf=pred.conf,
        image_diff_threshold=self.cfg.image_diff_threshold,
        image_vote_ratio=self.cfg.image_vote_ratio,
        depth_change_threshold=self.cfg.depth_change_threshold,
    )

    scene = self._compute_scene_reward(
        depths=pred.depth,
        extrinsics=pred.extrinsics,
        intrinsics=pred.intrinsics,
        conf=pred.conf,
        static_mask=masks["global_static"],
    )

    world_maps, point_clouds, scene_scale = self._build_dynamic_geometry(
        depths=pred.depth,
        extrinsics=pred.extrinsics,
        intrinsics=pred.intrinsics,
        frame_dynamic_masks=masks["frame_dynamic"],
        global_static_mask=masks["global_static"],
    )

    motion = compute_motion_reward(
        dynamic_point_clouds=point_clouds,
        scene_scale=scene_scale,
        min_motion_threshold=self.cfg.min_motion_threshold,
        tau_smooth=self.cfg.tau_smooth,
        tau_shape=self.cfg.tau_shape,
        shape_weight=self.cfg.motion_shape_weight,
        smooth_weight=self.cfg.motion_smooth_weight,
    )

    motion_quality = (
        self.cfg.motion_shape_weight * motion["shape"]
        + self.cfg.motion_smooth_weight * motion["smoothness"]
    )

    total = motion["gate"] * (
        self.cfg.total_scene_weight * scene["reward"]
        + self.cfg.total_motion_weight * motion_quality
    )

    return {
        "total": float(total),
        "scene": float(scene["reward"]),
        "motion": float(motion["reward"]),
        "motion_gate": float(motion["gate"]),
        "shape": float(motion["shape"]),
        "smoothness": float(motion["smoothness"]),
        "scene_error": scene["error"],
        "shape_error": motion["shape_error"],
        "smoothness_error": motion["smoothness_error"],
        "motion_energy": motion["motion_energy"],
        "static_ratio": float(masks["global_static"].mean()),
        "dynamic_ratio": float(masks["global_dynamic"].mean()),
        "unknown_ratio": float(masks["global_unknown"].mean()),
    }
```

---

# Part II：V2 Interaction

> **当前状态：暂不实施。** V2 依赖 robot/object 语义 mask，而 DA3 不提供语义信息，mask 传播（尤其接触融合阶段）的工程风险高于预期科研收益。优先完成 V1 全部实验和消融，若 V1 结果中发现明确的"运动正常但抓取关系错误，V1 无法区分"的 case，再按需引入 V2。以下内容保留作为设计参考，不进入当前开发计划。

## 13. V2 目标和前置条件

V2 在 V1 完成并验证后增加：

```text
R_interaction
```

V2 不修改：

- 帧差 + 深度变化率 mask；
- `R_scene`；
- motion gate；
- 一阶速度异常；
- `R_shape`；
- V1 的完整生成 N 个候选流程。

V2 必须区分：

```text
robot/end-effector
target object
```

DA3 只提供几何，不提供语义身份。因此 V2 需要一种区域来源。

建议优先级：

1. 为输入首帧提供一次人工 robot/object mask；
2. 使用已有分割模型产生首帧 mask，再传播到后续帧；
3. 使用动态连通区域和首帧 ROI 做启发式匹配；
4. 如果无法可靠区分，则该样本不计算 `R_interaction`，不能伪造接触分数。

V2 引入区域身份的目的是计算几何关系，不是引入视觉质量 Reward。

---

## 14. V2 任务元数据

建议新增：

```text
batch_task_specs.json
```

示例：

```json
{
  "test0001": {
    "interaction_required": false,
    "motion_required": true,
    "action": "rotate_and_lower"
  },
  "test0025": {
    "interaction_required": true,
    "motion_required": true,
    "action": "grasp_and_lift",
    "robot_mask": "masks/test0025_robot.png",
    "object_mask": "masks/test0025_object.png"
  }
}
```

对于：

- rotate wrist；
- open/close gripper；
- inspection motion；

不强制计算 object interaction。

对于：

- grasp；
- pick up；
- lift；
- carry；
- place；

启用 `R_interaction`。

---

## 15. V2 robot/object 区域传播

### 15.1 首帧 mask 对齐

首帧 mask resize 到 DA3 输出分辨率：

```python
def resize_mask(mask, height, width):
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0
```

### 15.2 基于动态连通区域的启发式传播

在不增加视频分割模型的 V2 初版中：

1. 对 `frame_dynamic_masks[t]` 提取连通区域；
2. 在第一帧根据与 robot/object 初始 mask 的重叠确定身份；
3. 后续帧根据 2D 质心距离、3D 质心距离、区域面积连续性进行匹配；
4. 如果匹配置信度过低，该帧标记为 interaction unknown。

参考接口：

```python
def extract_components(mask, min_area):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    components = []
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        components.append({
            "mask": labels == label,
            "area": int(area),
            "centroid_2d": centroids[label],
        })
    return components
```

匹配代价：

```python
cost = (
    w_2d * normalized_2d_distance
    + w_3d * normalized_3d_distance
    + w_area * abs(log(area_t / area_prev))
)
```

这一传播方法只适合主体分离较清楚的机械臂场景。若机器人和物体接触后连通区域融合，应允许 robot/object mask 在图像上接近或部分重叠，但仍通过历史质心维持身份。

---

## 16. V2 末端执行器和物体三维表示

### 16.1 物体三维中心

```python
object_points_t = world_map_t[object_mask_t]
object_center_t = np.median(object_points_t, axis=0)
```

### 16.2 末端执行器区域

如果没有单独的 end-effector mask，可以从 robot mask 中选取距离物体最近的一部分 3D 点：

```python
def nearest_robot_points_to_object(
    robot_points,
    object_points,
    keep_quantile=0.10,
):
    # 第一版可对点云先降采样，避免完整 O(NM)。
    robot = uniform_subsample_points(robot_points, 512)
    obj = uniform_subsample_points(object_points, 256)

    diff = robot[:, None, :] - obj[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    nearest = distances.min(axis=1)

    threshold = np.quantile(nearest, keep_quantile)
    return robot[nearest <= threshold]
```

末端中心：

```python
end_effector_center_t = np.median(
    nearest_robot_points,
    axis=0,
)
```

如果有精确 end-effector mask，应优先直接使用。

---

## 17. V2 接触距离和阶段识别

## 17.1 最近三维距离

```python
def minimum_cloud_distance(points_a, points_b):
    a = uniform_subsample_points(points_a, 512)
    b = uniform_subsample_points(points_b, 256)
    diff = a[:, None, :] - b[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return float(distances.min())
```

按场景尺度归一化：

```python
distance_t = minimum_cloud_distance(
    end_effector_points_t,
    object_points_t,
) / scene_scale
```

得到：

```text
d_0, d_1, ..., d_T
```

## 17.2 接触候选阶段

第一版使用相对距离阈值：

```python
contact_flags = distances < contact_distance_threshold
```

为了避免单帧偶然接近，要求至少连续 `K` 个关键帧：

```python
def keep_persistent_runs(flags, min_length=2):
    output = np.zeros_like(flags, dtype=bool)
    start = None
    for i, value in enumerate(flags):
        if value and start is None:
            start = i
        if (not value or i == len(flags) - 1) and start is not None:
            end = i if not value else i + 1
            if end - start >= min_length:
                output[start:end] = True
            start = None
    return output
```

---

## 18. V2 R_interaction 分量

## 18.1 接近趋势

在首次稳定接触之前，希望距离总体减小。

只使用一阶距离变化：

```python
distance_steps = np.diff(distances[:contact_start + 1])
approach_violation = np.maximum(distance_steps, 0.0)
E_approach = np.median(approach_violation)
```

允许局部小幅回退，不要求严格单调。

## 18.2 接触稳定性

接触阶段：

```python
contact_distances = distances[contact_flags]
E_contact_distance = np.median(contact_distances)
E_contact_stability = (
    np.quantile(contact_distances, 0.90)
    / (np.median(contact_distances) + 1e-8)
)
```

## 18.3 物体跟随耦合

只使用一阶位移：

```python
robot_steps = np.diff(end_effector_centers, axis=0)
object_steps = np.diff(object_centers, axis=0)
```

接触后的耦合误差：

```python
coupling_errors = np.linalg.norm(
    robot_steps - object_steps,
    axis=1,
) / scene_scale

E_coupling = np.median(
    coupling_errors[contact_pair_flags]
)
```

它惩罚：

- 机器人移动但物体不动；
- 物体突然独立飞走；
- 接触后两者运动方向和幅度严重不一致。

## 18.4 相对距离跳变

```python
relative_distance_steps = np.abs(np.diff(distances))
E_relative_jump = (
    np.quantile(relative_distance_steps, 0.95)
    / (np.median(relative_distance_steps) + 1e-8)
)
```

仍然只使用一阶差分，不引入二阶或三阶。

## 18.5 映射为分数

```python
R_approach = np.exp(-E_approach / tau_approach)
R_contact_stability = np.exp(
    -E_contact_stability / tau_contact_stability
)
R_coupling = np.exp(-E_coupling / tau_coupling)
R_relative = np.exp(-E_relative_jump / tau_relative)
```

初始组合：

```python
R_interaction = (
    0.15 * R_approach
    + 0.25 * R_contact_stability
    + 0.45 * R_coupling
    + 0.15 * R_relative
)
```

耦合项权重最高，因为对于抓取、搬运和放置任务，最关键的是接触后物体是否随机械臂共同运动。

## 18.6 interaction gate

如果任务要求接触，但整个视频没有形成持续接触：

```python
interaction_gate = 0.0
```

如果形成持续接触：

```python
interaction_gate = 1.0
```

平滑版本：

```python
interaction_gate = min(
    1.0,
    persistent_contact_frames / min_contact_frames,
)
```

最终：

```python
R_interaction *= interaction_gate
```

---

## 19. V2 总 Reward

对于不需要物体交互的 Prompt：

```python
R_total_v2 = R_total_v1
```

对于需要交互的 Prompt，初始方案：

```python
motion_quality = (
    0.7 * R_shape
    + 0.3 * R_smoothness
)

R_total_v2 = motion_gate * interaction_gate * (
    0.20 * R_scene
    + 0.50 * motion_quality
    + 0.30 * R_interaction
)
```

这样：

- 没有运动时总分归零；
- 明确要求交互但没有形成接触时总分归零；
- 背景稳定不能补偿完全未发生的动作；
- 接触正确也不能补偿严重的运动结构崩坏。

如果实验发现二值 interaction gate 过于严格，可以换成连续门控，但 V2 第一版建议保留清晰行为。

---

## 20. BoN 集成

完整候选生成流程保持不变。

`GeoRewardBoN.generate()` 仍然：

```python
for i in range(N):
    seed = seed_base + i

    video = self.wan.generate(
        input_prompt=prompt,
        img=image,
        frame_num=frame_num,
        seed=seed,
        **wan_kwargs,
    )

    frames = wan_output_to_da3_input(video)
    sampled_frames = [frames[idx] for idx in indices]

    reward = self.reward.compute_reward(
        sampled_frames,
        task_spec=task_spec,
    )

    candidates.append(video)
    rewards.append(reward)

best_idx = max(
    range(len(rewards)),
    key=lambda i: rewards[i]["total"],
)
```

所有 N 个视频都完整生成、完整解码和完整评分。

---

## 21. CLI 参数规划

V1 建议新增：

```text
--reward_version v1
--image_diff_threshold 15
--image_vote_ratio 0.30
--depth_change_threshold 0.05
--min_motion_threshold 0.01
--tau_smooth 2.0
--tau_shape 0.10
--motion_shape_weight 0.70
--motion_smooth_weight 0.30
```

V2 增加：

```text
--reward_version v2
--task_specs batch_task_specs.json
--contact_distance_threshold ...
--min_contact_frames 2
--tau_coupling ...
--tau_contact_stability ...
```

所有阈值写入 `rewards.json`，保证结果可复现。

---

## 22. 输出 JSON

V1 单候选建议记录：

```json
{
  "total": 0.72,
  "version": "v1",
  "scene": 0.81,
  "motion": 0.68,
  "motion_gate": 1.0,
  "shape": 0.74,
  "smoothness": 0.55,
  "motion_energy": 0.023,
  "shape_error": 0.037,
  "smoothness_error": 1.42,
  "static_ratio": 0.77,
  "dynamic_ratio": 0.18,
  "unknown_ratio": 0.05
}
```

V2 增加：

```json
{
  "interaction": 0.63,
  "interaction_gate": 1.0,
  "approach": 0.82,
  "contact_stability": 0.66,
  "coupling": 0.51,
  "relative_motion": 0.71,
  "persistent_contact_frames": 4
}
```

同时记录实际 seed。当前随机 `seed_base=None` 时必须把生成出的真实 seed 写入日志，保证候选可复现。

---

## 23. 调试可视化

虽然不评价视觉质量，但必须保存几何诊断图用于验证 Reward 是否计算正确。

V1 建议保存：

```text
debug_masks/
├── frame_00_static.png
├── frame_00_dynamic.png
├── frame_00_unknown.png
├── frame_00_overlay.png
├── depth_change_median.png
└── image_motion_ratio.png
```

overlay：

```text
绿色：static
红色：dynamic
黄色：unknown
```

V2 增加：

```text
蓝色：robot
紫色：object
白色：end-effector/contact points
```

调试图不参与 Reward，只用于确认：

- 背景是否被正确划为静态；
- 机械臂运动路径是否被划为动态；
- 光照变化是否产生大量误检；
- robot/object 身份是否在帧间保持；
- 接触阶段是否被正确识别。

---

## 24. 单元测试规划

### 24.1 帧差测试

构造：

- 全黑静态视频；
- 一个白色方块平移；
- 全局亮度变化；
- 单帧噪声。

验证：

- 静态视频 dynamic ratio 接近 0；
- 方块运动轨迹进入 global dynamic；
- per-frame mask 只覆盖方块当前/相邻位置；
- `uint8` 相减不会下溢。

### 24.2 深度变化测试

构造：

- 常量深度；
- 局部深度从 2.0 变为 1.0；
- 无效深度、NaN 和 0；
- 全局小尺度噪声。

验证 log-ratio 和 valid mask。

### 24.3 OR 组合测试

显式验证：

```text
image=True, depth=False  → dynamic=True
image=False, depth=True  → dynamic=True
image=True, depth=True   → dynamic=True
image=False, depth=False → dynamic=False
```

### 24.4 motion gate 测试

构造质心轨迹：

```text
完全静止
低幅连续运动
超过阈值的连续运动
单帧瞬移
```

验证 `motion_gate`。

### 24.5 一阶速度异常测试

```text
[1, 1, 1, 1]       → ratio 接近 1
[1, 1, 10, 1]      → ratio 明显增大
[0.1, 0.1, 0.1]    → ratio 接近 1，但 motion gate 低
```

### 24.6 shape 测试

构造点云：

- 刚体整体平移：形状签名应基本不变；
- 均匀旋转：形状签名应基本不变；
- 局部拉伸：`E_shape` 增大；
- 点云融合/坍缩：`E_shape` 增大。

### 24.7 interaction 测试

构造机器人和物体中心：

- 接近后共同移动；
- 机器人移动但物体静止；
- 物体突然飞走；
- 全程没有接触；
- 只有单帧接近。

验证 `interaction_gate` 和 `R_coupling`。

---

## 25. 实验规划

## 25.1 V1 消融

对同一批完整 N 候选比较：

```text
A0：当前 projection + anchor + confidence
A1：R_scene only
A2：R_scene + motion_gate
A3：R_scene + motion_gate + R_smoothness
A4：完整 V1：R_scene + R_shape + R_smoothness + motion_gate
```

重点观察：

- 完全静止候选是否还会被选中；
- 单帧瞬移候选排名是否下降；
- 结构崩坏候选排名是否下降；
- 正常运动候选是否稳定胜出。

## 25.2 mask 消融

比较：

```text
帧差 only
深度变化 only
OR：帧差 | 深度变化（默认）
AND：帧差 & 深度变化（消融）
```

报告：

- static/dynamic coverage；
- 人工 mask 的 IoU（如果有少量标注）；
- 最终候选排序变化；
- 光照变化下的误检；
- 与背景同色物体的漏检。

## 25.3 V2 消融

```text
V1
V1 + contact distance
V1 + coupling
完整 V2
```

重点观察：

- 抓取失败但运动平滑的视频是否下降；
- 机器人移动而物体不跟随的视频是否下降；
- 真正完成搬运的视频是否上升。

## 25.4 人工评价指标

每组候选人工标注：

```text
三维运动连续性：1～5
动态主体结构稳定性：1～5
机器人-物体交互合理性：1～5（适用时）
整体几何物理合理性：1～5
```

不标注视觉审美和清晰度，避免目标再次扩张。

统计：

- Spearman 相关系数；
- Kendall 相关系数；
- Reward Top-1 与人工 Top-1 一致率；
- 静止失败候选被选中的比例；
- 接触失败候选被选中的比例。

---

## 26. 实施顺序

### V1-1：区域 mask

产出：

- `region_masks.py`；
- 帧差累积；
- 深度变化中位数；
- OR 组合；
- global/per-frame mask；
- debug overlay。

### V1-2：静态 R_scene

产出：

- `_project_error_map()`；
- static-mask projection；
- static-mask anchor；
- confidence 分位数过滤；
- coverage 日志。

### V1-3：R_motion

产出：

- 动态点云；
- 鲁棒质心；
- `motion_gate`；
- 一阶速度异常；
- 局部距离分布 `R_shape`；
- V1 总 Reward。

### V1-4：完整 BoN 验证

产出：

- `run_bon.py` 新 Reward 参数；
- `run_bon_batch.py` 批量运行；
- 完整候选日志；
- V1 消融结果。

### V2-1：任务元数据和区域身份

产出：

- `batch_task_specs.json`；
- robot/object mask 接口；
- 连通区域匹配；
- interaction unknown 处理。

### V2-2：R_interaction

产出：

- 最近三维距离；
- 接触阶段；
- 一阶共同运动耦合；
- interaction gate；
- V2 总 Reward。

### V2-3：正式实验

产出：

- V1/V2 对比；
- 人工相关性；
- mask 和 Reward 消融；
- 参数标定结果。

---

## 27. 主要风险和处理

| 风险 | 影响 | 处理 |
|---|---|---|
| OR 将光照变化判为动态 | 静态区域减少 | 记录 OR/AND 消融；形态学去噪；后续可增加一致性条件 |
| 同像素深度变化受相机运动影响 | 大面积假动态 | 当前 Prompt 强制固定相机；另用 `R_scene` 检查相机漂移 |
| 动态 union mask 覆盖整条路径 | 质心混入背景 | R_motion 使用逐帧 mask，不使用 global union |
| 动态 mask 粗糙 | shape/centroid 不稳定 | 连通区域过滤、有效点数量门控、robust median |
| 质心不能代表关节运动 | 平滑性漏判 | 质心只做辅助 0.3；R_shape 权重 0.7 |
| 点云没有跨帧对应关系 | 固定点对不可比较 | 使用局部距离分布签名，不比较无对应索引 |
| DA3 confidence 未校准 | 固定阈值无效 | 使用帧内分位数，只做可靠性 mask |
| V2 无法区分 robot/object | interaction 无意义 | 要求首帧 mask/ROI；无法区分时不计算该分量 |
| 完全静止但 R_scene 高 | 静止候选胜出 | `motion_gate` 乘在 V1/V2 总分最外层 |

---

## 28. 完成判据

### V1 完成

- 可为任意完整视频输出 global/per-frame static/dynamic mask；
- OR 逻辑有单元测试；
- `R_scene` 只在 static mask 上计算；
- `motion_gate` 能让完全静止视频的总分接近 0；
- 一阶速度异常能惩罚单帧瞬移；
- `R_shape` 对整体刚体平移/旋转相对稳定，对局部拉伸明显下降；
- BoN 流程完整生成 N 个候选并按 V1 Reward 排序；
- 所有阈值、实际 seed 和分项指标写入 JSON。

### V2 完成

- 可根据 task spec 决定是否启用 interaction；
- 可获得 robot/object/end-effector 的逐帧几何区域；
- 能识别持续接触；
- 能区分“机器人移动、物体跟随”和“机器人移动、物体不动”；
- 需要交互但未形成接触时，interaction gate 生效；
- BoN 可按 V2 Reward 完整排序。

---

## 29. 最终推荐公式

### V1

```python
motion_gate = min(
    1.0,
    motion_energy / min_motion_threshold,
)

E_smoothness_excess = max(
    E_smoothness - 1.0, 0.0
)

R_smoothness = exp(
    -E_smoothness_excess / tau_smooth
)

R_shape = exp(
    -E_shape / tau_shape
)

motion_quality = (
    0.7 * R_shape
    + 0.3 * R_smoothness
)

R_total_v1 = motion_gate * (
    0.30 * R_scene
    + 0.70 * motion_quality
)
```

### V2

```python
R_total_v2 = motion_gate * interaction_gate * (
    0.20 * R_scene
    + 0.50 * motion_quality
    + 0.30 * R_interaction
)
```

对于不要求交互的 Prompt：

```python
R_total_v2 = R_total_v1
```

这套设计保持完整 Best-of-N 生成和评分流程不变，同时将 Reward 的核心从“全图静态几何一致性”转向：

```text
有没有运动
+
运动是否存在极端跳变
+
动态主体结构是否稳定
+
V2 中机器人和物体是否保持合理交互关系
```
