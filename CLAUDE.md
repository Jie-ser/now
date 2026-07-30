# GeoReward

利用 Depth Anything 3 (DA3) 的显式几何信息（深度、相机位姿、置信度）作为 Reward 信号，在 Wan2.2 I2V 视频生成模型的推理阶段提升生成视频的物理一致性。场景：机械臂操作。

## 项目结构

```
now/
├── geo_reward/                  # 核心模块
│   ├── __init__.py              # 导出 DA3GeoReward, GeometryRewardConfig, GeoRewardBoN, GeoRewardBoNProgressive
│   ├── da3_reward.py            # DA3GeoReward 主类 + GeometryRewardConfig + compute_reward_early
│   ├── region_masks.py          # 静态/动态区域分割（帧差+深度变化率+OR组合）
│   ├── motion_reward.py         # 运动质量评分（motion_gate + R_shape + R_smoothness）
│   ├── bon_pipeline.py          # GeoRewardBoN / GeoRewardBoNProgressive（两阶段固定比例淘汰）/ GeoRewardBoNOffline
│   └── utils.py                 # 格式转换（wan→PIL）、均匀抽帧
├── run_bon.py                   # 单条 BoN CLI（--mode bon / score）
├── run_bon_batch.py             # 批量 BoN CLI（--start/--end，模型只加载一次）
├── score_video.py               # 独立视频评分 CLI（支持 .mp4/.pt）
├── batch_prompts.json           # 机械臂测试 prompt 集合
├── batch_prompts_real.json      # 真实场景 prompt（第一批）
├── batch_prompts_real_2.json    # 真实场景 prompt（第二批）
├── batch_prompts_inputs_2.json  # 新输入批次 prompt
├── batch_prompts_inputs_real_3.json
├── batch_prompts_inputs_real_4.json
├── Progressive_Elimination_BoN_Technical_Report.md  # 渐进淘汰实验报告
├── GeoReward_SchemeA_Geometric_Motion_Execution_Plan.md  # V1 执行方案文档
├── Wan2.2/                      # 阿里视频生成模型（14B DiT, Flow Matching）
├── Depth-Anything-3-main/       # ByteDance 深度估计模型（DA3）
├── VGGTomega/                   # VGGT-Omega 参考（几何基础模型）
├── WMReward-main/               # Meta 参考实现（VJEPA-2 reward）
└── requirements.txt
```

## 工作流

- 本地（Windows）编辑代码 → git push → 远程服务器 git pull → 运行
- 远程服务器有GPU，本地不跑模型

## Reward V1 设计（方案 A：Scene + Motion）

### 核心思路

显式分割静态/动态区域，分别评价：

```
R_total = motion_gate × (0.30 × R_scene + 0.70 × motion_quality)
```

- `motion_gate`：动态区域 3D 质心位移中位数，完全静止 → 总分归零
- `R_scene`：只在静态区域计算投影一致性（60%）+ 首帧锚定（40%）
- `motion_quality = 0.70 × R_shape + 0.30 × R_smoothness`
  - `R_shape`：动态点云局部 kNN 距离分布签名的帧间稳定性
  - `R_smoothness`：速度序列 q95/median excess，惩罚单帧瞬移
- 所有分数通过 `exp(-E/τ)` 映射到 [0, 1]

### 简化 Reward（Early Checkpoint 用）

早期阶段 pred_x0 噪声大，R_shape/R_smoothness 不稳定，使用简化公式：

```
early_score = motion_gate × (0.4 × R_scene + 0.6 × motion_gate)
```

对应 `DA3GeoReward.compute_reward_early()` 方法。

### 区域分割

- RGB 帧差 + DA3 同像素深度 log-ratio 变化率
- OR 组合：任一信号检测到变化 → 动态（高召回）
- DA3 confidence 按帧内分位数做可靠性 mask，不作为正向 Reward
- 形态学开闭 + 连通区域过滤去噪点
- 同时维护 global mask（用于 R_scene）和 per-frame mask（用于 R_motion）

## Best-of-N 渐进淘汰设计（两阶段固定比例）

### 流程

```
N=8 候选全部同步去噪
    │
    ├── σ ≈ 0.83 (Step ~15/40) — Early Checkpoint
    │   ├── 对 8 个候选提取 pred_x0
    │   ├── VAE decode → DA3(12帧) → 简化 reward（R_scene + motion_gate）
    │   ├── 按分数排序，淘汰 bottom 50%（固定砍 4 个）
    │   ├── 保底：如果分数差距 < ε(0.02)，多留 1 个
    │   └── 4 个存活候选继续去噪
    │
    ├── σ ≈ 0.63 (Step ~25/40) — Mid Checkpoint
    │   ├── 对 4 个候选提取 pred_x0
    │   ├── VAE decode → DA3(20帧) → 完整 reward
    │   ├── 淘汰 bottom 50%（固定砍 2 个）
    │   └── 2 个存活候选继续去噪
    │
    └── Step 40/40 — Final
        ├── 完整 decode + 完整 reward
        └── 在 2 个中选出 best
```

### 计算量分析

```
DiT 步数: 8×15 + 4×10 + 2×15 = 190 步 (对比原版 8×40 = 320 步，节省 40.6%)
DA3 次数: 8 + 4 + 2 = 14 次
早期 DA3 采帧: 12 帧（vs 标准 20 帧，节省 40% DA3 推理）
```

### 淘汰规则

- 固定比例淘汰（`elimination_ratio = 0.5`）：每个 checkpoint 砍掉 bottom 50%
- Epsilon 保底（`score_epsilon = 0.02`）：最后一个存活者和第一个被淘汰者分数差 < ε 时多留 1 个
- 硬底线（`min_survivors = 2`）：无论如何保证至少 2 个存活
- σ-based 触发：用 σ 定义 checkpoint 位置，适应不同总步数

### 关键设计决策

- **为什么 σ=0.83（~Step 15）而不是更早？** Step 10 (σ≈0.90) 还在 high-noise 阶段，pred_x0 噪声极大；Step 15 刚进入 low-noise 模型约 5 步，全局布局已初步形成
- **为什么早期用简化 reward？** R_shape/R_smoothness 依赖 kNN 距离和速度序列，在噪声较大的 pred_x0 上不够稳定
- **为什么不用 mean-std 淘汰？** 实测发现 mean-std 方法对分数分布敏感，经常淘汰不足或过度淘汰；固定 50% 更可预测
- **VAE 全量 decode：** Wan2.2 VAE 使用 3D 时间卷积，latent 帧间有依赖，不能部分 decode

### GeoRewardBoNProgressive 默认参数

```python
DEFAULT_SIGMA_CHECKPOINTS = [0.83, 0.63]  # σ 阈值
DEFAULT_ELIMINATION_RATIO = 0.5            # 每 checkpoint 淘汰比例
DEFAULT_MIN_SURVIVORS = 2                  # 最终保底存活数
DEFAULT_SCORE_EPSILON = 0.02               # 分数不可区分阈值
DEFAULT_EARLY_MAX_FRAMES = 12              # 早期 DA3 采帧数
```

## 运行命令

```bash
# 安装
pip install -r requirements.txt
pip install -e Depth-Anything-3-main/
export PYTHONPATH=$PYTHONPATH:$(pwd)/Wan2.2

# Best-of-N 渐进淘汰（单条，默认）
python run_bon.py \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832 --sample_shift 3.0 --t5_cpu

# Best-of-N 渐进淘汰（批量）
python run_bon_batch.py \
  --start 1 --end 10 \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --da3_model /path/to/DA3 \
  --t5_cpu

# 自定义淘汰参数
python run_bon_batch.py \
  --start 1 --end 10 \
  --sigma_checkpoints 0.83 0.63 \
  --elimination_ratio 0.5 \
  --min_survivors 2 \
  --score_epsilon 0.02 \
  --early_max_frames 12 \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --da3_model /path/to/DA3 \
  --t5_cpu

# 禁用渐进淘汰（原版顺序 BoN）
python run_bon.py --no_progressive \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832

# 独立视频评分
python score_video.py --video path/to/video.mp4
python score_video.py --video path/to/video.pt --output_dir results/

# V1 Reward 参数可通过 CLI 调整，例如：
python score_video.py --video path/to/video.mp4 \
  --tau_shape 0.10 --tau_smooth 2.0 \
  --min_motion_threshold 0.01 \
  --image_diff_threshold 15.0
```

## 关键技术细节

- DA3 默认模型：`depth-anything/DA3NESTED-GIANT-LARGE-1.1`（1.4B）
- DA3 `inference()` 有 `@torch.inference_mode()`，V1 直接用高层 API
- DA3 输出：depth `(N,H,W)`, extrinsics `(N,4,4)` world-to-cam, intrinsics `(N,3,3)`, conf `(N,H,W)`
- 帧差使用 DA3 `processed_images` 分辨率，保证 RGB 与深度严格对齐
- Wan2.2 输出：`(3, T, H, W)` 值域 `[-1, 1]`，81帧，VAE 时间压缩 4x 空间 8x8
- 动态点云采样前按 3D 坐标字典序排序，避免 mask 形状变化导致签名漂移
- 场景尺度归一化：首帧静态区域深度中位数
- 渐进淘汰通过 `wan.prepare_progressive()` 共享文本 embedding 和噪声初始化
- 中间 checkpoint 通过 `extract_pred_x0()` 提取预测清晰 latent，不中断去噪

## GeometryRewardConfig 默认值

```python
image_diff_threshold = 15.0    # RGB 帧差阈值
image_vote_ratio = 0.30        # 全局动态投票比例
depth_change_threshold = 0.05  # 深度 log-ratio 阈值
min_motion_threshold = 0.01    # motion_gate 最低运动量
tau_smooth = 2.0               # R_smoothness 温度
tau_shape = 0.10               # R_shape 温度
motion_shape_weight = 0.70     # R_shape 权重
motion_smooth_weight = 0.30    # R_smoothness 权重
tau_scene_proj = 0.05          # R_scene 投影温度
tau_scene_anchor = 0.05        # R_scene 锚定温度
scene_proj_weight = 0.60       # 投影在 R_scene 中的权重
scene_anchor_weight = 0.40     # 锚定在 R_scene 中的权重
total_scene_weight = 0.30      # R_scene 在总分中的权重
total_motion_weight = 0.70     # R_motion 在总分中的权重
```

这些是初始可运行值，正式数值需要通过实验标定。

## 开发计划

- **V1**（当前）：Scene + Motion，显式区域分割 + 运动质量评分 + 两阶段渐进淘汰 BoN
- **V2**（后续，依赖 V1 实验结果）：增加 R_interaction（机器人-物体交互一致性）
- **Part 2**（更后续）：可微 DA3 + 梯度引导去噪 + 与 BoN 组合

## 代码规范

- 用中文交流，代码和文件名用英文
- 不做多余抽象，保持代码直接可读
