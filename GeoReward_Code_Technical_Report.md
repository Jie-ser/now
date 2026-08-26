# GeoReward 代码技术报告

> 基于代码实际实现撰写，2026-08-14
> 更新：2026-08-27 — 新增 Tree Branching + Guidance 联合模式

---

## 1. 项目概述

GeoReward 利用 4RC 4D 重建模型的显式几何信息（3D 点云、追踪轨迹、相机位姿、置信度）作为 Reward 信号，在 Wan2.2 I2V（Image-to-Video）视频生成模型的推理阶段提升生成视频的物理一致性。应用场景为机械臂操作。

系统包含三个核心组件：
1. **Reward 信号**：基于 4RC 输出的显式几何一致性评分（`ReconstructionReward`）
2. **BoN 采样策略**：渐进淘汰 + Tree Branching 加速选优 + Tree Branching + Guidance 联合模式
3. **梯度引导**：可微几何 loss 驱动去噪方向优化（`GeometricGuidance`），支持独立使用或与 Tree Branching 联合

---

## 2. Reward 设计：4RC 显式几何一致性

### 2.1 整体架构

```
输入: PIL 帧序列 (均匀抽帧, 默认 20 帧)
  │
  ├── 4RC 推理 → pts(N,H,W,3), track(N,H,W,3), extrinsics(N,4,4),
  │              intrinsics(N,3,3), conf(N,H,W), conf_track(N,H,W)
  │
  ├── 后处理
  │     ├── valid_mask: conf/conf_track 分位数过滤 (保留 top 80%)
  │     ├── scene_scale: 首帧静态区域相机坐标系深度中位数
  │     └── dynamic_mask: track 最大位移法后验推断
  │
  ├── G_anchor: 首帧深度合理性门控
  │
  ├── R_static: 静态区域跨帧重投影 + 有效比例
  ├── R_dynamic: 动态区域覆盖率 + 轨迹加速度 + 速度平滑度
  ├── R_motion: 相机运动平滑度 + motion gate + 瞬移惩罚
  │
  └── R_total = G_anchor × (0.40×R_static + 0.40×R_dynamic + 0.20×R_motion)
```

**实现类：** `ReconstructionReward`（`geo_reward/recon_reward.py`）

### 2.2 4RC 适配器（`geo_reward/fourrc_adapter.py`）

将 PIL 帧转换为 4RC 模型可接受的输入格式：

1. **尺寸调整**：最长边缩放至 518px，中心裁剪至 patch_size=14 对齐
2. **归一化**：`(x - 0.5) / 0.5` → [-1, 1]
3. **视图字典**：`{img: (1,3,H,W), true_shape: [[H,W]], idx: i, instance: str(i)}`

**推理输出提取（`run_4rc_inference`）：**
- `pts (N, H, W, 3)`：世界坐标系 3D 点
- `track (N, H, W, 3)`：相对于 query frame 的位移（`track_abs - pts[query_idx]`）
- `conf (N, H, W)`：几何置信度
- `conf_track (N, H, W)`：追踪置信度
- `extrinsic (N, 4, 4)`：camera-to-world
- `intrinsic (N, 3, 3)`

### 2.3 Valid Mask 与 Dynamic Mask

**Valid Mask（`compute_valid_mask`）：**
- 分别对 conf 和 conf_track 计算全视频 Q20 分位数
- 保留 >= 该阈值的像素（即保留 top 80%）
- 输出：`valid_geo (N,H,W)` 和 `valid_track (N,H,W)`

**Dynamic Mask（`compute_dynamic_mask`，逐帧最大位移法）：**
```python
max_displacement = max_t ||track[t]||
dynamic = max_displacement > threshold_ratio × scene_scale
```
- `threshold_ratio = 0.01`（默认）
- 输出：`(H, W)` 的 static/dynamic 二值 mask
- 内置安全检查：如果 `track[0]` 均值位移 > 0.1，警告可能 query frame 错误

**Scene Scale（`compute_scene_scale`）：**
- 首帧世界坐标点变换到相机坐标系（通过 extrinsic_frame0），z 即深度
- 取静态区域中 finite 且 > 0.01 的深度中位数
- 兜底链：所有深度 → abs 中位数 → 1.0
- 下限 clamp >= 1e-6

### 2.4 G_anchor：首帧深度合理性门控

- 将首帧世界坐标 3D 点变换到相机坐标系（z = depth）
- 计算静态区域中深度在合理范围 (0.01, 100.0) 且 finite 的比例
- 门控：`sigmoid((validity - 0.8) / 0.05)`
- 当首帧深度大面积异常时（validity < 0.8），整体分数被压制

### 2.5 R_static：静态区域重投影一致性

**帧对选择：** strides = [1, 3, 5]，生成所有 (i, i+stride) 对

**计算流程：**
1. 提取帧 i 的静态 3D 点（需同时满足 static_mask 和 valid_geo）
2. 变换到帧 j 的相机坐标系（world → cam_j，通过求逆 extrinsic_j）
3. 投影到帧 j 的图像平面（intrinsic_j @ pts_cam）
4. 用 `grid_sample` 双线性采样帧 j 的实际深度
5. **遮挡过滤**：`proj_depth < sampled_depth × 1.05`（被遮挡点不参与）
6. 误差：`|log(proj_depth) - log(sampled_depth)|`，取每对中位数
7. 同时计算有效重投影比例 `V_ratio`（有效点数 / 总静态点数）

**最终公式：**
```python
R_static = exp(-E_reproj / 0.10) × sigmoid((V_ratio - 0.3) / 0.1)
```
- 有效比例项确保不会因大量点被遮挡过滤而虚高
- **兜底**：无有效帧对时返回 0.5（中性分数）

### 2.6 R_dynamic：动态区域运动质量

**覆盖率（Coverage）：** 动态区域中跨帧均有有效追踪的像素比例

**加速度惩罚（E_accel）：**
- 采样至多 1000 条动态轨迹（valid_track 全帧有效的点）
- 二阶差分：`accel = traj[t+2] - 2×traj[t+1] + traj[t]`
- 中位数加速度 / scene_scale
- 要求 N >= 3 帧

**速度极端值惩罚（E_speed）：**
- 帧间位移 → speed 序列
- `excess = (Q95 / median) - 1`，clamp >= 0

**最终公式：**
```python
R_dynamic = coverage^0.5 × exp(-E_accel / 0.05) × exp(-E_speed / 3.0)
```
- coverage 开方：避免覆盖率过度主导
- **兜底**：dynamic 总点数 < 50 或 N < 3 时返回 0.5 或 `coverage^0.5`

### 2.7 R_motion：相机运动与全局运动

**相机平移加速度（E_cam）：**
- 从 extrinsics 提取相机位置序列（camera-to-world 的 translation 列）
- 二阶差分 → 加速度 → 中位数 / scene_scale

**相机旋转加速度（E_rot）：**
- 逐帧相对旋转矩阵 → 旋转角（arccos trace 法）→ 角速度序列
- 二阶差分 → 中位数

**Motion Gate（平滑 sigmoid）：**
```python
dynamic_motion = median(max_frame_displacement of dynamic pixels)
gate = sigmoid((dynamic_motion - 0.005) / 0.005)
```

**瞬移惩罚（E_teleport）：**
- 逐帧动态像素位移 > `0.5 × scene_scale` 的比例

**最终公式：**
```python
R_motion = gate × exp(-E_cam / 0.02) × exp(-E_rot / 0.05) × (1 - E_teleport)
```

### 2.8 ReconRewardConfig 实际默认值

| 参数 | 值 | 用途 |
|------|-----|------|
| `static_weight` | 0.40 | R_static 权重 |
| `dynamic_weight` | 0.40 | R_dynamic 权重 |
| `motion_weight` | 0.20 | R_motion 权重 |
| `dynamic_threshold_ratio` | 0.01 | 动态 mask 阈值比 |
| `tau_reproj` | 0.10 | 重投影误差温度 |
| `occlusion_margin` | 1.05 | 遮挡过滤余量 |
| `tau_accel` | 0.05 | 加速度温度 |
| `tau_speed` | 3.0 | 极端速度温度 |
| `max_sample_pixels` | 1000 | 动态轨迹采样上限 |
| `tau_cam` | 0.02 | 相机平移加速度温度 |
| `tau_rot` | 0.05 | 相机旋转加速度温度 |
| `min_motion` | 0.005 | 最低运动量 |
| `tau_motion` | 0.005 | motion gate sigmoid 温度 |
| `conf_valid_quantile` | 0.20 | 置信度过滤分位数 |
| `max_frames` | 20 | 最大采帧数 |
| `image_size` | 518 | 4RC 输入分辨率 |

### 2.9 设计特点总结

- **几何来源**：4RC 原生 4D 输出（点+轨迹），直接支持动态场景
- **动态分割**：track 最大位移后验推断，无需手工帧差/深度变化率规则
- **conf 角色**：仅作为 valid mask（过滤不可靠区域），不参与分数计算
- **所有 checkpoint 统一公式**：早期/中期/最终 Reward 使用完全相同的 `compute_reward()`，区别仅在输入帧数
- **兜底策略**：异常情况返回 0.5（中性分数），避免极端值影响选优
- **可微性**：4RC forward 支持梯度流，可用于 Phase 3 梯度引导

---

## 3. Best-of-N 采样策略

### 3.1 顺序 BoN（`GeoRewardBoN`）

最基础的策略：生成 N 个候选视频，逐个评分，选最高分。

```
for i in range(N):
    video = wan.generate(seed=seed_base+i)
    frames = wan_output_to_pil(video) → sample_frames(max=20)
    reward = recon_reward.compute_reward(frames)
best = argmax(rewards["total"])
```

**计算量：** N×40 DiT 步 + N 次 4RC Reward 推理

### 3.2 渐进淘汰 BoN（`GeoRewardBoNProgressiveV2`）

核心思想：在去噪中间 checkpoint 提前评分淘汰弱候选，节省后续计算。

**流程：**

```
N=8 候选并行去噪
    │
    ├── σ≈0.83 (Early) → 评分 → 淘汰 50% → 4 存活
    ├── σ≈0.63 (Mid)   → 评分 → 淘汰 50% → 2 存活
    └── Step 40 (Final) → 评分 → 选出 best
```

**实现细节：**

1. **σ→Step 映射：** `wan.find_step_for_sigma(state, sigma_target)` 动态确定 checkpoint 对应的步数（适应不同 shift 参数）
2. **Pred_x0 提取：** 中间 checkpoint 通过 `wan.extract_pred_x0()` 获取预测清晰 latent（不中断去噪）
3. **VAE 解码：** `wan.decode_latent()` 全量解码（Wan2.2 VAE 使用 3D 时间卷积，latent 帧间有依赖，不可部分解码）

**淘汰规则（`_eliminate` 方法）：**

1. 若存活数 <= `min_survivors`(2)：全部保留
2. 按 total score 降序排列（NaN 视为 -∞）
3. `keep_count = max(min_survivors, len - int(len × 0.5))`
4. **Epsilon 保底**：末位存活者与首位被淘汰者分差 < 0.02 时多保留 1 个
5. 返回存活/淘汰列表

**三阶段评分模式（DiT/4RC 交替加载）：**
```
Phase 1 (Decode): 逐候选 VAE decode → tensor 立即移 CPU（释放 GPU）
Phase 2 (Score):  DiT+VAE → CPU, 4RC → GPU, 逐候选评分
Phase 3 (Swap):   4RC → CPU, DiT+VAE → GPU（恢复去噪）
```

**模型 offload 实现：**
- `_offload_dit()`：`low_noise_model` + `high_noise_model` → CPU
- `_offload_vae()`：`vae.model` 或 `vae` → CPU
- `_load_vae()`：VAE → CUDA
- `_offload_4rc()`：`recon_reward.model` → CPU
- `_load_4rc()`：`recon_reward.model` → CUDA

**计算量对比：**

| 方案 | DiT 步数 | Reward 推理次数 |
|------|----------|----------------|
| 朴素 BoN (N=8) | 320 | 8 |
| 渐进淘汰 | ~190 | 14 |

**默认参数：**
```python
DEFAULT_SIGMA_CHECKPOINTS = [0.83, 0.63]
DEFAULT_ELIMINATION_RATIO = 0.5
DEFAULT_MIN_SURVIVORS = 2
DEFAULT_SCORE_EPSILON = 0.02
DEFAULT_EARLY_MAX_FRAMES = 12
```

### 3.3 Tree Branching 加速（`GeoRewardBoNTreeBranching`）

核心思想：高噪声阶段不同候选结构趋同，共享 K 条主干去噪后分叉扩展为 N 条候选，再进入渐进淘汰。

**流程（默认 K=2, branches_per_trunk=4, branch_sigma=0.90, eta=0.10）：**

```
Step 0 → branch_step:  K=2 条主干并行去噪          [2×branch_step 步]
branch_step:            每条主干分叉为 4 条 → 共 8 条  [分叉点]
branch_step → end:      8 条进入渐进淘汰             [正常淘汰流程]
```

**分叉噪声注入公式（ODE-to-SDE）：**
```python
z_branch = sqrt(1 - η²) × z_trunk + η × σ_t × ε,    ε ~ N(0, I)
```

- **方差保持**：`sqrt(1-η²)` 确保扰动后 latent 仍在正确噪声水平 manifold
- **σ_t 缩放**：扰动大小与当前噪声水平成正比
- **理论基础**：Kim et al. (2025) ODE-to-SDE 边缘分布一致性
- η=0.10：适度多样性，避免偏离训练分布

**σ-based 分叉点：** 通过 `wan.find_step_for_sigma(state, branch_sigma)` 动态确定分叉步数，适应不同 shift/总步数配置。

**约束条件：**
- `branch_sigma > max(sigma_checkpoints)`：分叉必须在第一个淘汰 checkpoint 之前
- `N == num_trunks × branches_per_trunk`
- η 推荐 0.05~0.15，> 0.20 可能产生 artifacts
- `branch_sigma` 建议 ≤ 0.93（避免分叉后仍有 high_noise_model 步）

**实现流程（`_generate_tree` 方法）：**
1. **Phase 1 (Trunk)**：`wan.denoise_candidates(state, trunk_indices, 0, branch_step)` — 去噪 K 条主干
2. **Phase 2 (Branch)**：`wan.branch_candidates(state, trunk_indices, branches_per_trunk, eta, branch_seeds)` — ODE-to-SDE 噪声注入创建 N 条候选
3. **Phase 3 (Progressive)**：从 branch_step 开始执行正常渐进淘汰（仅使用步数 > branch_step 的 sigma checkpoints）

**计算量对比（shift=5.0, 40 步）：**

| 方案 | DiT 步数 | 节省比例 |
|------|----------|----------|
| 朴素 BoN (N=8) | 320 | — |
| 渐进淘汰 | ~190 | 40.6% |
| Tree Branching | ~134 | 58.1% |

**默认参数：**
```python
DEFAULT_NUM_TRUNKS = 2
DEFAULT_BRANCHES_PER_TRUNK = 4
DEFAULT_BRANCH_SIGMA = 0.90
DEFAULT_BRANCH_ETA = 0.10
```

### 3.4 离线评分（`GeoRewardBoNOffline`）

对已生成的 .pt 视频文件进行后评分，不涉及生成过程。用于实验分析和对比。

### 3.5 Tree Branching + Guidance 联合模式（`GeoRewardBoNTreeBranchingGuided`）

核心思想：在 Tree Branching 的首轮淘汰后（σ < 0.83），对存活的少量候选施加梯度引导，以最优投入产出比提升几何一致性。

**引导时间线：**

```
σ=1.0 → 0.90:  2 条主干去噪              [无 guidance — pred_x0 噪声大]
σ=0.90:         分叉 → 8 条候选
σ=0.90 → 0.83:  8 条独立去噪             [无 guidance — 代价高且即将淘汰]
σ=0.83:         评分淘汰 → 4 条
σ=0.83 → 0.63:  4 条去噪 + guidance(freq=3) [~1-2 次引导]
σ=0.63:         评分淘汰 → 2 条
σ=0.63 → 0.08:  2 条去噪 + guidance(freq=3) [~2-3 次引导]
Final:          评分选 best
```

**实现方式：**
- 继承 `GeoRewardBoNTreeBranching`
- 重写 `generate()` 在 `prepare_progressive()` 之后调用 `_vae_setup_fn()` 拆分 VAE 到 guidance 设备，结束后 `_vae_restore_fn()` 恢复
- 重写 `_progressive_elimination()` 使用 `denoise_candidates_with_guidance()` 替代 `denoise_candidates()`
- `guidance_offload_dit=None, guidance_reload_dit=None`（4-GPU 常驻各卡，无需搬运）
- VAE decode 时将 latent 转移到 `vae_device`（decoder 在 cuda:1/2）

**Guidance 窗口控制：**
- 通过设置 `guidance_sigma_max = 0.83`，`should_guide()` 自动跳过 σ > 0.83 的所有步
- 无需手工区分阶段——sigma window 机制自然实现"首轮淘汰后才引导"

**4-GPU 常驻显存布局（无 offload 开销）：**
```
cuda:0  DiT (low_noise_model + high_noise_model)
cuda:1  VAE decoder front half (conv2 + encoder + middle + stage0 + stage1)
cuda:2  VAE decoder back half (stage2 + stage3 + head)
cuda:3  4RC model
```

- 去噪步：DiT 在 cuda:0 no_grad 前向
- Guidance 步：latent 从 cuda:0 → VAE decode_differentiable on cuda:1/2 → frames → 4RC on cuda:3 → loss → grad → 回传修改 v_pred
- 评分 checkpoint：latent 移到 cuda:1 做 VAE decode，4RC 已在 cuda:3 直接评分
- offload_models=False（多卡常驻时不做任何模型搬运）

**VAE 生命周期管理：**
1. Case 开始：VAE 整体在 cuda:0（`prepare_progressive` 需要 encoder 做 latent 编码）
2. Encode 完成后：`_vae_setup_fn()` 调用 `split_decoder_to_devices(cuda:1, cuda:2)` 拆分
3. 去噪 + guidance + 评分：decoder 常驻 cuda:1/2
4. Case 结束：`_vae_restore_fn()` 恢复 VAE 到 cuda:0（清除 split 状态）

**异常恢复鲁棒性：**
- `_vae_setup_fn()` 在 try 内执行，任何异常触发 finally
- finally 中嵌套 try：`cleanup_progressive()` 和 `_vae_restore_fn()` 相互独立，一方抛错不影响另一方

**默认参数（Tree Branching + Guidance 联合模式）：**
```python
guidance_scale = 0.001
guidance_frequency = 3          # CLI 默认在 tree+guidance 时自动从 5 调为 3
guidance_sigma_min = 0.08
guidance_sigma_max = 0.83       # CLI 默认在 tree+guidance 时自动从 0.90 调为 0.83
guidance_frames = 8
```

**计算量对比（shift=5.0, 40 步, N=8）：**

| 方案 | DiT 步数 | Reward 推理 | Guidance 次数 |
|------|----------|-------------|---------------|
| 朴素 BoN | 320 | 8 | 0 |
| 渐进淘汰 | ~190 | 14 | 0 |
| Tree Branching | ~134 | 14 | 0 |
| **Tree Branching + Guidance** | **~134** | **14** | **~3-6** |

---

## 4. 梯度引导（Gradient Guidance）

### 4.1 设计思路

在去噪循环中，对中间预测 x0 做可微的几何 loss 计算，通过梯度修改速度预测（v_pred），将 latent 推向几何一致性更好的方向。

**实现类：** `GeometricGuidance`（`geo_reward/guidance.py`）

### 4.2 引导条件

```python
def should_guide(sigma_t, step_idx):
    return sigma_min < sigma_t < sigma_max and step_idx % frequency == 0
```

- `sigma_min = 0.08`：低噪声阶段不引导（避免破坏细节）
- `sigma_max = 0.90`：高噪声阶段不引导（pred_x0 不可靠）
- `frequency = 5`：每 5 步引导一次（平衡效果与计算量）

### 4.3 引导公式（`guided_v_pred` 方法）

```python
# Flow matching: x0_hat = latent - sigma_t × v_pred
x0_hat = (latent - sigma_t * v_pred).detach().requires_grad_(True)

# Forward: VAE decode → 4RC → geometric loss
loss = L_reproj + 0.5 × L_track_smoothness + 0.3 × L_anchor
grad = torch.autograd.grad(loss, x0_hat)

# WMReward-style normalization
scaling_t = 1.0 - sigma_t²
norm_ratio = ||v_pred||₂ / (||grad||₂ + 1e-8)
v_guided = v_pred + scale × norm_ratio × scaling_t × grad
```

**归一化设计要点：**
- `norm_ratio`：梯度幅度对齐到 v_pred 幅度（避免过大/过小修改）
- `scaling_t = 1 - σ²`：高噪声时衰减（σ 大 → scaling 小 → 修改弱）
- `scale = 0.001`：全局缩放因子

### 4.4 可微 Loss 组成

**L_reproj（重投影误差，权重 1.0）：**
- 连续帧对（最多 3 对）的静态区域 log-depth 重投影误差
- 与 R_static 逻辑一致，但保持梯度流
- 实现：`_differentiable_reproj_loss(pts, extrinsics, intrinsics, static_mask, valid_mask, scene_scale)`

**L_track_smoothness（轨迹平滑度，权重 0.5）：**
- 动态区域（最多 500 点）3D 轨迹加速度范数均值 / scene_scale
- 要求 N >= 3 帧
- 实现：`_differentiable_track_smoothness(track, dynamic_mask, valid_mask, scene_scale)`

**L_anchor（首帧深度锚定，权重 0.3）：**
- 首帧静态区域深度 log 偏差于中位数
- 中位数 `.detach()` 不参与梯度（仅偏差产生梯度）
- 实现：`_differentiable_anchor_loss(pts, static_mask, extrinsic_frame0)`

### 4.5 梯度计算完整流水线（`_compute_guidance_gradient`）

```
x0_hat → [transfer to vae_device]
       → VAE.decode_differentiable() → video (3, T, H, W)
       → 均匀抽帧 (guidance_frames=8 帧, torch.linspace)
       → _prepare_views() (可微 F.interpolate resize + center crop + normalize)
       → [transfer to fourrc_device]
       → 4RC loss_of_one_batch() (绕过 inference() 的 @torch.no_grad)
       → 提取 pts, track, extrinsics, intrinsics, conf, conf_track
       → compute_valid_mask() (detached, 不参与梯度)
       → compute_dynamic_mask() (detached, 不参与梯度)
       → recon_reward.compute_differentiable_loss()
       → autograd.grad(loss, x0_hat)
       → [transfer grad back to source device]
```

**关键设计：** mask 计算全部 detached，梯度仅通过 pts 和 track 的几何误差传回。

### 4.6 多 GPU 显存管理

| GPU 数量 | 分配方案 | 备注 |
|----------|----------|------|
| 4 GPU | GPU0=DiT, GPU1=VAE 前半(stage2 ResBlocks), GPU2=VAE 后半(upsample+stage3), GPU3=4RC | **目前唯一可用方案** |
| 3 GPU | GPU0=DiT, GPU1=VAE, GPU2=4RC | Pro6000 96GB 显存 OOM |
| 2 GPU | GPU0=DiT, GPU1=VAE+4RC | Pro6000 96GB 显存 OOM |
| 1 GPU | 回调式 offload（DiT↔4RC 交替） | Pro6000 96GB 显存 OOM |

> **注意：** 目前梯度引导仅 4-GPU 方案可正常运行，3/2/1-GPU 方案在 Pro6000（96GB）上均因显存不足而 OOM。

**单 GPU 引导步显存切换：**
```
DiT forward → v_pred (no_grad) → offload DiT to CPU
→ load 4RC + VAE on GPU → VAE decode_differentiable(x0_hat)
→ 4RC forward → loss → grad → offload 4RC+VAE
→ modify v_pred → reload DiT → scheduler.step
```

**4-GPU VAE 拆分（`split_decoder_to_devices`）：**
- VAE decoder 中 stage2 ResBlocks 放 cuda:1
- upsample + stage3 放 cuda:2
- 中间 feature map 跨设备传输

### 4.7 使用模式

- `--guidance` 可与 `--tree_branching` 联合使用（`GeoRewardBoNTreeBranchingGuided`，首轮淘汰后引导，4-GPU 常驻）
- `--guidance --no_progressive` 独立使用（顺序 BoN + 引导，N 个候选逐个引导生成后评分选优）
- `--guidance` 与纯渐进淘汰（`--tree_branching` 未开启）互斥（CLI 自动禁用并 warning）
- Tree Branching + Guidance 联合模式下，CLI 自动调整默认值：`guidance_frequency` 5→3，`guidance_sigma_max` 0.90→0.83（用户显式传参可覆盖）
- 代码中 `GeoRewardBoNProgressiveV2Guided` 类（渐进淘汰 + 引导）存在但 CLI 未开放

---

## 5. 4RC 模型集成

### 5.1 模型加载

```python
from arc.dust3r.model import AsymmetricCroCo3DStereo
model = AsymmetricCroCo3DStereo.from_pretrained(model_path)
```

路径自动发现：`fourrc_adapter.py` 中 `_ensure_4rc_importable()` 会自动搜索 `../4RC-main/4RC-main` 和 `../4RC-main` 添加到 sys.path。

### 5.2 推理模式

**标准推理（Reward 评分用）：**
```python
from arc.dust3r.inference_multiview import inference
result = inference(views, model, device, batch_size=1, dtype="bf16-mixed")
```
- 被 `@torch.inference_mode()` 包裹，无梯度
- 用于 `compute_reward()` 中的常规评分

**可微推理（Guidance 用）：**
```python
from arc.dust3r.inference_multiview import loss_of_one_batch
from arc.dust3r.utils.device import collate_with_cat
batch = collate_with_cat([tuple(views)])
result = loss_of_one_batch(batch, model, None, device, "bf16-mixed")
```
- 直接调用底层函数，保持梯度流
- 绕过 `inference()` 的 `@torch.no_grad` 装饰器
- 用于 `GeometricGuidance._compute_guidance_gradient()` 中

### 5.3 输出后处理

- `track` 转为相对位移：`track_relative = track_abs - pts[query_idx]`
- `query_idx` 从 4RC 输出中自动提取（支持 tensor/list/scalar 格式）
- `extrinsic` 为 camera-to-world 变换矩阵（求逆后为 world-to-camera，用于重投影）

---

## 6. 工具函数与 CLI

### 6.1 帧格式转换（`geo_reward/utils.py`）

- `wan_output_to_pil(tensor)`：(3,T,H,W) [-1,1] → list[PIL.Image]
- `sample_frames(total=81, max=20)`：首帧必选 + 均匀间隔，去重

### 6.2 坐标变换

- `transform_to_camera(pts_world, w2c)`：世界坐标 → 相机坐标（齐次变换）

### 6.3 CLI 工具

| 脚本 | 功能 |
|------|------|
| `run_bon_v2.py` | 单条 BoN 生成（支持渐进淘汰/Tree Branching/梯度引导） |
| `run_bon_batch_v2.py` | 批量 BoN 生成（模型只加载一次） |
| `score_video_v2.py` | 独立视频评分（仅加载 4RC） |

**关键 CLI 参数：**
- `--tree_branching`：启用 Tree Branching 加速
- `--guidance`：启用梯度引导（可与 `--tree_branching` 联合，或独立配合 `--no_progressive`）
- `--tree_branching --guidance`：Tree Branching + Guidance 联合模式（4-GPU 常驻，首轮淘汰后引导）
- `--no_model_offload`：禁用 DiT/4RC 交替加载
- `--no_progressive`：禁用渐进淘汰，改为顺序生成
- `--save_intermediate`：保存中间 checkpoint 视频
- 多 GPU 自动检测与分配（guidance 模式下自动根据 GPU 数量选择最优方案）

---

## 7. 代码与文档不一致之处

| 项目 | 文档描述 | 代码实际 |
|------|----------|----------|
| conf_valid_quantile | "Q20，保留 top 80%" | 代码实现正确：阈值取 Q20，>= 阈值的保留（约 80%） |
| 早期 Reward | 文档称 "统一公式" | 代码确认：所有 checkpoint 调用相同 `compute_reward()`，无简化版 |
| 渐进淘汰计算量 | "8×15 + 4×10 + 2×15 = 190" | 实际步数取决于 shift 参数和 σ→step 动态映射，190 为 shift=5.0 时的典型值 |
| R_static 兜底 | 文档未提及 | 代码中无有效帧对时返回 0.5（中性分数） |
| forward-backward cycle | 文档提及 | 代码中 R_static 仅做单向重投影（非双向） |
| VAE split_decoder_to_devices | 文档未详细描述 | 4-GPU 模式下 VAE decoder 被手动拆分：stage2 ResBlocks on cuda:1, upsample+stage3 on cuda:2 |
| guidance + progressive | 文档称 "暂不支持" | `GeoRewardBoNProgressiveV2Guided` 类已存在（CLI 未开放）；`GeoRewardBoNTreeBranchingGuided` 支持 Tree Branching + Guidance 联合（CLI 已开放） |

---

## 8. 总结

GeoReward 构建了一套从 Reward 信号设计到推理加速的完整系统：

1. **Reward 信号**：基于 4RC 4D 重建模型的显式几何一致性评分，利用 pts/track/extrinsics 计算重投影误差、轨迹平滑度、相机运动质量三维度分数，conf 仅做 valid mask 不参与评分
2. **BoN 策略**：渐进淘汰（σ-based checkpoint + 固定比例淘汰 + epsilon 保底）→ Tree Branching（共享主干 + ODE-to-SDE 分叉）→ Tree Branching + Guidance（首轮淘汰后梯度引导），计算量从 320 步降至 ~134 步（58% 节省），同时通过梯度引导进一步提升几何一致性
3. **梯度引导**：WMReward 风格归一化 + 可微 4RC forward + 三组分 geometric loss，支持独立使用或与 Tree Branching 联合（sigma window 自动控制引导时机）
4. **工程实现**：完善的显存管理（DiT/4RC 交替加载、4-GPU 常驻模式、VAE decoder 拆分与生命周期管理），嵌套 try/finally 保证异常恢复鲁棒性
