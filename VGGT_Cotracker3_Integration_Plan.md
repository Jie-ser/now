# VGGT-Omega + CoTracker3 集成方案

## 目标

在保留现有 4RC 全部代码不变的前提下，新建 `VGGT_Cotracker3/` 文件夹，实现基于 VGGT-Omega（几何+相机）+ CoTracker3（dense tracking）的替代 GeoReward pipeline，用于验证 GeoReward 设计的通用性（不绑定特定重建模型）。

## 约束

- 现有 `geo_reward/` 目录所有文件**不做任何修改**
- 所有新代码仅在 `VGGT_Cotracker3/` 文件夹内
- 不需要模型置信度（conf/conf_track）作为 mask —— 但使用 CoTracker3 的 **visibility**（物理遮挡信号）过滤不可见像素
- Reward 计算公式与现有 V2（`recon_reward.py`）**完全一致**，包括 `occlusion_margin`、`max_sample_pixels` 等全部参数
- 工作分辨率：统一在 VGGT-Omega 输出分辨率（约 304×512）下做逐像素 dense tracking，不做稀疏 grid 插值
- 包含 BoN 机制（简单全量生成+评分选优）
- 不包含：梯度引导、渐进淘汰、Tree Branching

### 分辨率策略

两个模型都有内部分辨率限制（VGGT-Omega 长边 512 + patch_size=16 对齐，CoTracker3 内部 384×512）。
"逐像素不降采样"的含义：**在工作分辨率 `(H_v, W_v)` 下，对每个像素建立独立的 tracking query，不做稀疏 grid + 插值**。

具体流程：
1. VGGT-Omega 预处理将原始帧 resize 到 `(H_v, W_v)`（如 304×512），输出 depth/camera 都在此分辨率
2. CoTracker3 输入也 resize 到 `(H_v, W_v)`，对该分辨率的 **全部 H_v×W_v 个像素** 建立 tracking query
3. 最终所有输出（pts, track, extrinsic, intrinsic）都在 `(N, H_v, W_v, ...)` 分辨率下
4. 两个模型内部的下采样/上采样是模型架构固有行为，不属于 pipeline 的降采样

### Visibility 处理策略

CoTracker3 输出的 `visibility (T, H, W)` 标记的是物理遮挡（该像素在该帧是否可见），**不同于** 4RC 的 `conf/conf_track`（模型对自身预测的置信度）。

- 不可见像素的 2D track 位置不可靠，反投影到 3D 会产生错误的 displacement
- 对于不可见像素：`track[t, h, w]` 设为 0，R_dynamic 计算时通过 visibility mask 排除
- R_static 不受影响（使用 pts 直接重投影，不依赖 track）
- 这是物理合理的过滤，不是模型置信度过滤

---

## 文件结构

```
VGGT_Cotracker3/
├── __init__.py                  # 导出主要类
├── vggt_omega_adapter.py        # VGGT-Omega 模型加载 + 推理 → depth, extrinsic, intrinsic, pts
├── cotracker3_adapter.py        # CoTracker3 模型加载 + 逐像素 dense tracking
├── combo_adapter.py             # 组合两个模型输出 → 4RC 兼容格式 (pts, track, extrinsic, intrinsic)
├── recon_reward_vggt.py         # 无 conf mask 的 ReconstructionReward（复用计算逻辑）
├── bon_pipeline_vggt.py         # 简单 BoN pipeline（全量生成+评分选优）
├── run_bon_vggt.py              # 单条 BoN CLI 入口
└── run_bon_batch_vggt.py        # 批量 BoN CLI 入口
```

---

## 各文件详细设计

### 1. `vggt_omega_adapter.py`

**职责**：封装 VGGT-Omega 模型的加载和推理，将输出转换为标准格式。

**依赖路径**：`VGGTomega/vggt-omega-main/`

#### 关键函数

```python
def load_vggt_omega(checkpoint_path, device="cuda"):
    """
    加载 VGGT-Omega 模型。
    
    Args:
        checkpoint_path: 权重路径（HuggingFace 下载的 .pth）
        device: 设备
    
    Returns:
        model: VGGTOmega 实例（eval 模式）
    """
    # from vggt_omega.models import VGGTOmega
    # model = VGGTOmega(patch_size=16, embed_dim=1024, 
    #                    enable_camera=True, enable_depth=True, enable_alignment=False)
    # model.load_state_dict(...)
    # return model.eval().to(device)


def preprocess_frames(frames_pil, image_resolution=512, patch_size=16):
    """
    将 PIL 帧列表转为 VGGT-Omega 输入格式。
    
    Args:
        frames_pil: List[PIL.Image]
        image_resolution: 目标分辨率（长边）
        patch_size: 对齐的 patch 大小
    
    Returns:
        images: (N, 3, H, W) tensor, 值域 [0, 1]
    """
    # 参考 vggt_omega/utils/load_fn.py 的 load_and_preprocess_images()
    # 1. resize 使长边 = image_resolution
    # 2. center crop 到 patch_size 的倍数
    # 3. ToTensor (归一化到 [0, 1])


def run_vggt_omega_inference(model, images, device="cuda"):
    """
    运行 VGGT-Omega 推理。
    
    Args:
        model: VGGTOmega 模型
        images: (N, 3, H, W) tensor, 值域 [0, 1]
    
    Returns:
        dict:
          - depth: (N, H, W) float tensor
          - extrinsic_c2w: (N, 4, 4) camera-to-world SE(3)
          - intrinsic: (N, 3, 3) 内参矩阵
          - pts: (N, H, W, 3) 世界坐标 3D 点
    """
    # 1. predictions = model(images.unsqueeze(0).to(device))
    # 2. depth = predictions["depth"].squeeze(0).squeeze(-1)  # (N, H, W)
    # 3. pose_enc = predictions["pose_enc"]  # (1, N, 9)
    # 4. from vggt_omega.utils.pose_enc import encoding_to_camera
    #    extrinsic_c_from_w, intrinsic = encoding_to_camera(pose_enc, (H, W))
    #    extrinsic_c_from_w: (1, N, 3, 4)  camera-from-world
    # 5. 转为 camera-to-world (4, 4):
    #    使用 closed_form_inverse_se3() 或手动 R^T, -R^T @ T
    #    补齐为 (N, 4, 4)
    # 6. 反投影 depth → pts (N, H, W, 3):
    #    参考 demo_gradio.py 的 unproject_depth_map_to_point_map()
    #    或使用上面讨论的标准反投影代码
```

#### 坐标约定转换

VGGT-Omega 输出 `extrinsic` 是 **camera-from-world** `(N, 3, 4)` 格式 `[R | T]`。
我们的 reward 需要 **camera-to-world** `(N, 4, 4)` 格式。

转换方法：
```python
# camera-from-world: [R | T], shape (N, 3, 4)
R = extrinsic_cfw[:, :3, :3]   # (N, 3, 3)
T = extrinsic_cfw[:, :3, 3:]   # (N, 3, 1)
# camera-to-world: [R^T | -R^T @ T]
R_inv = R.transpose(-1, -2)
T_inv = -R_inv @ T
# 拼接为 (N, 4, 4)
c2w = torch.eye(4).unsqueeze(0).expand(N, -1, -1).clone()
c2w[:, :3, :3] = R_inv
c2w[:, :3, 3:] = T_inv
```

---

### 2. `cotracker3_adapter.py`

**职责**：封装 CoTracker3 的加载和逐像素 dense tracking。

**依赖路径**：`co-tracker-main/co-tracker-main/`

#### 关键函数

```python
def load_cotracker3(checkpoint_path=None, device="cuda"):
    """
    加载 CoTracker3 offline 模型。
    
    Args:
        checkpoint_path: 本地权重路径。None 则用 torch.hub 自动下载。
    
    Returns:
        model: CoTrackerPredictor 实例
    """
    # Option A: torch.hub
    #   model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    # Option B: 本地
    #   from cotracker.predictor import CoTrackerPredictor
    #   model = CoTrackerPredictor(checkpoint=checkpoint_path, offline=True)
    # return model.to(device)


def frames_to_video_tensor(frames_pil, target_size=None):
    """
    将 PIL 帧列表转为 CoTracker3 输入格式。
    
    Args:
        frames_pil: List[PIL.Image]
        target_size: (H, W) 如果需要 resize（与 VGGT-Omega 输出分辨率对齐）
    
    Returns:
        video: (1, T, 3, H, W) float tensor, 值域 [0, 255]
    """


def run_dense_tracking(model, video, query_frame=0, chunk_size=4096, device="cuda"):
    """
    逐像素 dense tracking：对 query_frame 的每个像素追踪到所有帧。
    
    Args:
        model: CoTrackerPredictor
        video: (1, T, 3, H, W) tensor, 值域 [0, 255]
        query_frame: 起始帧索引（默认 0）
        chunk_size: 每批处理的查询点数（控制显存）
        device: 设备
    
    Returns:
        tracks_2d: (T, H, W, 2) float tensor — 每个像素在每帧的 (x, y) 坐标
        visibility: (T, H, W) bool tensor — 每个像素在每帧的可见性
    """
```

#### 逐像素 dense tracking 实现方案

CoTracker3 的 predictor 支持自定义 `queries (B, N, 3)` 参数（每个 query 为 `[frame_idx, x, y]`）。对于 H×W 的帧，生成所有像素的 query：

```python
H, W = video.shape[3], video.shape[4]
total_pixels = H * W  # 例如 480*832 = 399,360

# 生成所有像素 query: (1, H*W, 3)
ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
queries_all = torch.stack([
    torch.full((H*W,), query_frame, dtype=torch.float32),
    xs.reshape(-1).float(),
    ys.reshape(-1).float(),
], dim=-1).unsqueeze(0)  # (1, H*W, 3)

# 分 chunk 处理（避免显存溢出）
tracks_chunks = []
vis_chunks = []
for start in range(0, total_pixels, chunk_size):
    end = min(start + chunk_size, total_pixels)
    queries_chunk = queries_all[:, start:end, :]  # (1, chunk, 3)
    
    tracks_chunk, vis_chunk = model(video, queries=queries_chunk)
    # tracks_chunk: (1, T, chunk, 2)
    # vis_chunk: (1, T, chunk)
    
    tracks_chunks.append(tracks_chunk)
    vis_chunks.append(vis_chunk)

# 拼接 → reshape
tracks_all = torch.cat(tracks_chunks, dim=2)  # (1, T, H*W, 2)
vis_all = torch.cat(vis_chunks, dim=2)          # (1, T, H*W)

tracks_2d = tracks_all.squeeze(0).reshape(T, H, W, 2)  # (T, H, W, 2)
visibility = vis_all.squeeze(0).reshape(T, H, W)         # (T, H, W)
```

**注意事项**：
- 每个 chunk 都会重新编码视频（encoder 重复跑），如果 CoTracker3 内部不支持缓存视频特征
- 可优化：拆解 predictor 内部逻辑，先跑一次 encoder 缓存 fmaps，再只跑 decoder（需读 predictor 源码确认可行性）
- chunk_size 推荐 2048~8192，根据 GPU 显存调整
- 预计 480×832 分辨率需要 ~50-100 个 chunk

---

### 3. `combo_adapter.py`

**职责**：组合 VGGT-Omega 和 CoTracker3 的输出，生成 4RC 兼容格式。

```python
def run_combo_inference(
    vggt_model, cotracker_model, frames_pil,
    image_resolution=512, chunk_size=4096, device="cuda"
):
    """
    组合 VGGT-Omega + CoTracker3 推理，输出 4RC 兼容格式。
    
    Args:
        vggt_model: VGGTOmega 实例
        cotracker_model: CoTrackerPredictor 实例
        frames_pil: List[PIL.Image]
        image_resolution: VGGT-Omega 输入分辨率
        chunk_size: CoTracker3 chunk 大小
    
    Returns:
        dict:
          - pts: (N, H, W, 3)       世界坐标 3D 点
          - track: (N, H, W, 3)     逐像素 3D 位移（相对帧 0），不可见像素为 0
          - extrinsic: (N, 4, 4)    camera-to-world
          - intrinsic: (N, 3, 3)    内参
          - visibility: (N, H, W)   逐帧逐像素可见性（来自 CoTracker3，用于 R_dynamic 过滤）
    """
```

#### 核心流程

```
步骤 1: VGGT-Omega 推理
  frames_pil → preprocess → model.forward()
  → depth (N,H,W), extrinsic_c2w (N,4,4), intrinsic (N,3,3), pts (N,H,W,3)

步骤 2: 分辨率对齐
  VGGT-Omega 输出分辨率 (H_v, W_v) 可能与原始帧 (H_o, W_o) 不同
  需要记录缩放比例，确保 CoTracker3 的 2D tracks 与 depth map 对齐
  方案：将 CoTracker3 的输入 resize 到与 VGGT-Omega 相同的 (H_v, W_v)
  这样两者坐标系一致，无需额外映射

步骤 3: CoTracker3 dense tracking
  frames_pil → resize 到 (H_v, W_v) → video tensor [0,255]
  → run_dense_tracking() → tracks_2d (T,H_v,W_v,2), visibility (T,H_v,W_v)

步骤 4: 2D tracks → 3D displacement (构造 track)
  对于帧 t 中每个像素 (h, w):
    (x_t, y_t) = tracks_2d[t, h, w]         # 该像素在帧 t 的 2D 追踪位置
    depth_t = bilinear_sample(depth[t], x_t, y_t)  # 在帧 t depth map 上插值取深度
    pts_3d_t = unproject(x_t, y_t, depth_t, intrinsic[t], extrinsic_c2w[t])  # → 世界坐标
    pts_3d_0 = pts[0, h, w]                  # 该像素在帧 0 的世界坐标
    track[t, h, w] = pts_3d_t - pts_3d_0     # 3D 位移

步骤 5: 返回
  return { "pts": pts, "track": track, "extrinsic": extrinsic_c2w, "intrinsic": intrinsic, "visibility": visibility }
```

#### 2D→3D 反投影辅助函数

```python
def bilinear_sample_depth(depth, x, y):
    """
    在 depth map 上双线性插值采样。
    
    Args:
        depth: (H, W) 单帧 depth map
        x, y: (H, W) float — 采样坐标（可能是亚像素）
    
    Returns:
        sampled_depth: (H, W)
    """
    # 使用 F.grid_sample 实现
    # 注意坐标归一化到 [-1, 1]


def unproject_2d_to_world(x, y, depth, intrinsic, extrinsic_c2w):
    """
    将 2D 像素坐标 + depth 反投影到世界坐标。
    
    Args:
        x, y: (H, W) 像素坐标
        depth: (H, W) 深度
        intrinsic: (3, 3) 内参
        extrinsic_c2w: (4, 4) camera-to-world
    
    Returns:
        world_pts: (H, W, 3)
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 相机坐标
    x_cam = (x - cx) / fx * depth
    y_cam = (y - cy) / fy * depth
    z_cam = depth
    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)
    
    # 世界坐标
    R = extrinsic_c2w[:3, :3]
    t = extrinsic_c2w[:3, 3]
    pts_world = pts_cam @ R.T + t
    return pts_world
```

---

### 4. `recon_reward_vggt.py`

**职责**：基于 VGGT+CoTracker3 输出的 ReconstructionReward，去掉 conf mask。

#### 设计原则

- 从 `geo_reward/recon_reward.py` 复制核心计算逻辑
- **删除 `valid_geo`（conf-based）相关代码**：R_static 全像素参与，不做置信度过滤
- **`valid_track` 替换为 CoTracker3 的 `visibility`**：R_dynamic 用 `visibility & dynamic_mask` 过滤遮挡像素
- **删除 `conf`、`conf_track` 相关参数和输入**
- 保持所有 reward 公式不变（R_static, R_dynamic, R_motion, G_anchor）
- 保持所有温度参数和权重默认值不变
- 调用 combo_adapter 获取几何输出，而非 fourrc_adapter

#### 类结构

```python
class VGGTReconRewardConfig:
    """
    配置类，与 ReconRewardConfig 保持参数完全一致（仅去掉 conf_valid_quantile 和梯度引导参数）。
    """
    # Weights（与 recon_reward.py 一致）
    static_weight: float = 0.50
    dynamic_weight: float = 0.30
    motion_weight: float = 0.20

    # Dynamic mask
    dynamic_threshold_ratio: float = 0.01

    # R_static（与 recon_reward.py 一致）
    tau_reproj: float = 0.05
    occlusion_margin: float = 1.05        # ← 补齐：重投影遮挡判断的深度余量

    # R_dynamic（与 recon_reward.py 一致）
    tau_accel: float = 0.02
    tau_speed: float = 1.5
    max_sample_pixels: int = 1000         # ← 补齐：R_dynamic 轨迹采样上限

    # R_motion（与 recon_reward.py 一致）
    tau_cam: float = 0.02
    tau_rot: float = 0.05
    min_motion: float = 0.005
    tau_motion: float = 0.02

    # Frame sampling
    max_frames: int = 20
    image_size: int = 512  # VGGT-Omega 输入分辨率（长边）

    # CoTracker3
    chunk_size: int = 4096  # 每批追踪的像素数


class VGGTReconstructionReward:
    def __init__(self, config=None):
        self.cfg = config or VGGTReconRewardConfig()

    def compute_reward(self, frames_pil, vggt_model=None, cotracker_model=None):
        """
        计算 GeoReward。
        
        Args:
            frames_pil: List[PIL.Image]
            vggt_model: VGGTOmega 模型实例
            cotracker_model: CoTrackerPredictor 模型实例
        
        Returns:
            dict:
              - total: float          最终 reward [0, 1]
              - version: "vggt_v2"
              - R_static: float
              - R_dynamic: float
              - R_motion: float
              - G_anchor: float
              - scene_scale: float
              - dynamic_ratio: float
        """
        # 1. combo_adapter.run_combo_inference(vggt_model, cotracker_model, frames_pil)
        #    → pts, track, extrinsic, intrinsic
        # 2. compute_scene_scale(pts, extrinsic_frame0=extrinsic[0])
        #    → 复用 fourrc_adapter 相同逻辑（但在本文件内重新实现）
        # 3. compute_dynamic_mask(track, threshold_ratio, scene_scale)
        #    → 复用相同逻辑
        # 4. _compute_R_static(pts, extrinsic, intrinsic, static_mask)
        #    → 与现有相同，但不用 valid_geo 过滤
        # 5. _compute_R_dynamic(track, dynamic_mask)
        #    → 与现有相同，但不用 valid_track 过滤
        # 6. _compute_R_motion(extrinsic, track, dynamic_mask)
        #    → 与现有相同
        # 7. _compute_anchor_gate(pts, extrinsic)
        #    → 与现有相同
        # 8. R_total = G_anchor * (w_s * R_static + w_d * R_dynamic + w_m * R_motion)

    # --- 以下私有方法从 recon_reward.py 复制并移除 valid mask 逻辑 ---
    def _compute_R_static(self, pts, extrinsic, intrinsic, static_mask, scene_scale): ...
    def _compute_R_dynamic(self, track, dynamic_mask, scene_scale): ...
    def _compute_R_motion(self, extrinsic, track, dynamic_mask, scene_scale): ...
    def _compute_anchor_gate(self, pts, extrinsic): ...
```

#### 与现有 `recon_reward.py` 的具体差异

| 原代码 | 新代码 | 变化 |
|--------|--------|------|
| `compute_valid_mask(conf, conf_track, quantile)` | 删除 | 不需要模型置信度 mask |
| `valid_geo & static_mask` 过滤 R_static 像素 | 仅用 `static_mask` | 去掉 valid_geo（所有像素参与 R_static） |
| `valid_track & dynamic_mask` 过滤 R_dynamic 像素 | `visibility & dynamic_mask` | 用 CoTracker3 visibility 替代 valid_track（物理遮挡过滤） |
| `run_4rc_inference(model, views)` | `run_combo_inference(vggt, cotracker, frames)` | 几何后端切换 |
| `conf_valid_quantile` 参数 | 删除 | 不需要 |
| `occlusion_margin` 参数 | **保留，值相同** | R_static 重投影遮挡判断一致 |
| `max_sample_pixels` 参数 | **保留，值相同** | R_dynamic 轨迹采样逻辑一致 |
| 返回 `valid_geo_ratio`, `valid_track_ratio` | 替换为 `visibility_ratio` | 反映实际过滤比例 |
| 梯度引导相关参数 | 删除 | 本期不实现 |

---

### 5. `bon_pipeline_vggt.py`

**职责**：简单 BoN pipeline（全量生成 N 个候选 + 全部评分 + 选最优）。

#### 设计

不使用渐进淘汰，所有 N 个候选完整生成后统一评分。

```python
class GeoRewardBoNVGGT:
    """
    简单 Best-of-N pipeline（VGGT-Omega + CoTracker3 后端）。
    
    流程：
    1. Wan2.2 生成 N 个候选视频（完整 40 步去噪）
    2. VAE decode 所有候选
    3. 对每个候选：均匀抽帧 → VGGT+CoTracker3 推理 → 计算 reward
    4. 选 reward 最高的候选
    """
    
    def __init__(
        self,
        wan_i2v,                    # Wan2.2 I2V 模型封装
        vggt_model_path,            # VGGT-Omega 权重路径
        cotracker_model_path=None,  # CoTracker3 权重路径（None 用 torch.hub）
        N=8,                        # 候选数
        num_frames_for_reward=20,   # reward 评分用的帧数
        reward_config=None,         # VGGTReconRewardConfig
        chunk_size=4096,            # CoTracker3 per-chunk 点数
        device="cuda",
    ):
        self.wan = wan_i2v
        self.N = N
        self.num_frames_for_reward = num_frames_for_reward
        self.chunk_size = chunk_size
        self.device = device
        
        # 模型延迟加载（与 DiT 交替占用 GPU）
        self.vggt_model_path = vggt_model_path
        self.cotracker_model_path = cotracker_model_path
        self.reward = VGGTReconstructionReward(reward_config or VGGTReconRewardConfig())

    def generate(self, image, prompt, **kwargs):
        """
        执行 BoN 生成。
        
        Args:
            image: 首帧图片路径或 PIL.Image
            prompt: 动作指令文本
            **kwargs: 传给 Wan2.2 的其他参数
        
        Returns:
            dict:
              - best_video: 最优视频 tensor
              - best_score: 最优分数
              - all_scores: 所有候选分数
              - all_details: 所有候选的详细 reward 分解
        """
        # 步骤 1: 生成 N 个候选（Wan2.2 完整去噪）
        # 步骤 2: VAE decode
        # 步骤 3: 卸载 DiT+VAE，加载 VGGT-Omega + CoTracker3
        # 步骤 4: 对每个候选评分
        # 步骤 5: 卸载 VGGT+CoTracker3
        # 步骤 6: 返回最优候选
```

#### 显存管理策略

```
阶段 1 — 生成:
  GPU: DiT (Wan2.2) + VAE
  CPU: VGGT-Omega, CoTracker3

阶段 2 — 评分:
  GPU: VGGT-Omega + CoTracker3（交替或同时，取决于显存）
  CPU: DiT, VAE

切换方式: .to("cpu") / .to("cuda")
```

注意：VGGT-Omega (~1B 参数) + CoTracker3 (~40M 参数) 总显存需求远小于 DiT (14B)，可以同时放在 GPU 上。

---

### 6. `run_bon_vggt.py`（单条 CLI）

```python
"""
单条 Best-of-N CLI（VGGT-Omega + CoTracker3 后端）。

用法：
python VGGT_Cotracker3/run_bon_vggt.py \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --vggt_model /path/to/vggt_omega_checkpoint.pth \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832 --sample_shift 5.0 --t5_cpu
"""
```

#### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt_dir` | 必填 | Wan2.2 权重路径 |
| `--vggt_model` | 必填 | VGGT-Omega 权重路径 |
| `--cotracker_model` | None（torch.hub） | CoTracker3 权重路径 |
| `--image` | 必填 | 首帧图片 |
| `--prompt` | 必填 | 动作指令 |
| `--N` | 8 | 候选数 |
| `--size` | 480*832 | 视频分辨率 |
| `--num_frames_for_reward` | 20 | 评分帧数 |
| `--chunk_size` | 4096 | CoTracker3 chunk 大小 |
| `--sample_shift` | 5.0 | Wan2.2 采样 shift |
| `--t5_cpu` | False | T5 放 CPU |
| `--output_dir` | ./output_vggt | 输出目录 |
| Reward 参数 | 同 ReconRewardConfig | --static_weight, --dynamic_weight, 等 |

### 7. `run_bon_batch_vggt.py`（批量 CLI）

与现有 `run_bon_batch_v2.py` 类似，支持 `--start/--end` 批量处理，模型只加载一次。

---

## 关键实现细节

### 1. 分辨率对齐

VGGT-Omega 和 CoTracker3 的内部分辨率不同：
- VGGT-Omega: 输入 resize 到长边 512，patch_size=16 对齐 → 实际如 `(336, 512)` 或 `(512, 336)`
- CoTracker3: 内部 resize 到 `(384, 512)` 然后输出 rescale 回原始分辨率

**统一方案**：
1. VGGT-Omega 预处理后记录实际输出分辨率 `(H_v, W_v)`
2. 将原始帧 resize 到 `(H_v, W_v)` 后再给 CoTracker3
3. 这样 CoTracker3 输出的 2D tracks 坐标与 VGGT-Omega 的 depth map 严格对齐
4. 最终所有输出都在 `(H_v, W_v)` 分辨率下

### 2. 逐像素 tracking 的性能

对于 `(336, 512)` 分辨率（示例）：
- 总像素数: 172,032
- chunk_size=4096: ~42 个 chunk
- 每个 chunk 含一次 CoTracker3 前向传播
- CoTracker3 单次推理约 0.5-2 秒（取决于帧数和 GPU）
- 总计 ~21-84 秒 per candidate

优化方向（后续可选，本期不实现）：
- 拆解 CoTracker3 predictor，缓存视频特征 fmaps，chunk 只跑 decoder
- 降低推理分辨率

### 3. track 构造中的数值稳定性

- depth 在 tracked 位置可能为负或 NaN（遮挡/帧外区域）→ clamp depth > 0.01
- bilinear_sample 可能越界（tracked 到帧外）→ clamp 坐标到 [0, H-1] × [0, W-1]
- **不可见像素处理**：CoTracker3 的 visibility=False 的像素，`track` 设为 0，并返回 visibility mask
- visibility mask 传入 reward 计算，在 R_dynamic 中替代原来的 valid_track：
  ```python
  # 原代码: combined_mask = dynamic_mask.unsqueeze(0) & valid_track
  # 新代码: combined_mask = dynamic_mask.unsqueeze(0) & visibility
  ```
- R_static 不使用 visibility（直接用 pts 重投影，不依赖 track）

### 4. 与现有 reward 公式的一致性

以下公式保持不变（仅去掉 valid mask）：

```
R_total = G_anchor × (0.50 × R_static + 0.30 × R_dynamic + 0.20 × R_motion)

R_static = exp(-E_reproj / τ_reproj) 的加权
R_dynamic = coverage + accel_penalty + speed_smoothness
R_motion = cam_accel_penalty + rot_accel_penalty + motion_gate + teleport_penalty
G_anchor = exp(-anchor_error / τ)

dynamic_mask: max_t ||track[t]|| > threshold × scene_scale
scene_scale: 首帧静态区域 depth 中位数
```

---

## 依赖关系

```
run_bon_vggt.py / run_bon_batch_vggt.py
  └── bon_pipeline_vggt.py (GeoRewardBoNVGGT)
        ├── recon_reward_vggt.py (VGGTReconstructionReward)
        │     └── combo_adapter.py (run_combo_inference)
        │           ├── vggt_omega_adapter.py (VGGT-Omega 推理)
        │           └── cotracker3_adapter.py (CoTracker3 dense tracking)
        └── Wan2.2 (视频生成，复用现有代码)
```

外部依赖：
- `VGGTomega/vggt-omega-main/` — VGGT-Omega 模型代码
- `co-tracker-main/co-tracker-main/` — CoTracker3 模型代码
- `Wan2.2/` — 视频生成模型（复用现有）

---

## 验证计划

### 单元测试
1. `vggt_omega_adapter.py`: 输入 5 张 PIL 图 → 输出 depth, extrinsic, intrinsic, pts 形状正确
2. `cotracker3_adapter.py`: 输入 5 帧视频 → 输出 dense tracks (T, H, W, 2) 形状正确
3. `combo_adapter.py`: 组合输出格式与 4RC 一致（形状、值域）
4. `recon_reward_vggt.py`: 对同一视频分别用 4RC reward 和 VGGT reward 评分，检查分数合理

### 集成测试
5. `run_bon_vggt.py`: 单条端到端，生成 2 个候选并评分选优
6. 对同一批视频，比较 4RC pipeline 和 VGGT pipeline 的排序一致性（Spearman ρ）

---

## 运行命令示例

```bash
# 单条 BoN
python VGGT_Cotracker3/run_bon_vggt.py \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --vggt_model /path/to/vggt_omega_checkpoint.pth \
  --image /pfs/mayuema/spj/now/inputs/inputs_real/001.png \
  --prompt "robot arm picks up the red block" \
  --N 8 --size 480*832 --sample_shift 5.0 --t5_cpu

# 批量 BoN
python VGGT_Cotracker3/run_bon_batch_vggt.py \
  --start 1 --end 24 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --vggt_model /path/to/vggt_omega_checkpoint.pth \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json \
  --name_prefix test_vggt \
  --N 8 --t5_cpu --sample_shift 5.0
```
