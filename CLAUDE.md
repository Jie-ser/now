# GeoReward

利用 Depth Anything 3 (DA3) 的显式几何信息（深度、相机位姿、置信度）作为 Reward 信号，在 Wan2.2 I2V 视频生成模型的推理阶段提升生成视频的物理一致性。场景：机械臂操作。

## 项目结构

```
now/
├── geo_reward/                  # 核心模块
│   ├── __init__.py              # 导出 DA3GeoReward, GeometryRewardConfig, ReconstructionReward, ReconRewardConfig, GeoRewardBoN, GeoRewardBoNProgressive, GeoRewardBoNProgressiveV2, GeoRewardBoNTreeBranching, GeometricGuidance
│   ├── da3_reward.py            # DA3GeoReward 主类 + GeometryRewardConfig + compute_reward_early（V1 保留）
│   ├── recon_reward.py          # 【V2 新增】ReconstructionReward + ReconRewardConfig（4RC 显式几何一致性）
│   ├── fourrc_adapter.py        # 【V2 新增】4RC 接口适配（PIL→view、valid_mask、dynamic_mask）
│   ├── guidance.py              # 【V2 新增】梯度引导模块（Phase 3，GeometricGuidance）
│   ├── region_masks.py          # 静态/动态区域分割（帧差+深度变化率+OR组合）（V1 保留）
│   ├── motion_reward.py         # 运动质量评分（motion_gate + R_shape + R_smoothness）（V1 保留）
│   ├── bon_pipeline.py          # GeoRewardBoN / GeoRewardBoNProgressive（V1）/ GeoRewardBoNProgressiveV2（V2+模型交替加载）/ GeoRewardBoNTreeBranching（Tree Branching 加速）/ GeoRewardBoNOffline
│   └── utils.py                 # 格式转换（wan→PIL）、均匀抽帧、坐标变换
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
├── GeoReward_V2_Reconstruction_Quality_Reward_Proposal.md  # V2 方案文档
├── GeoReward_SchemeA_Geometric_Motion_Execution_Plan.md  # V1 执行方案文档
├── Wan2.2/                      # 阿里视频生成模型（14B DiT, Flow Matching）
├── 4RC-main/                    # 4RC 4D 几何基础模型（重建+追踪）
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

## Reward V2 设计（4RC 显式几何一致性）

### 核心思路

用 4RC 的显式 3D 输出（pts、track、extrinsics）计算几何一致性。conf/conf_track 仅作为 valid mask，不参与分数计算。单模型、一次推理、后验区分 static/dynamic。

```
R_total = G_anchor × (0.40 × R_static + 0.40 × R_dynamic + 0.20 × R_motion)
```

- `G_anchor`：首帧静态区域深度合理性门控
- `R_static`：静态区域跨帧深度重投影误差 + forward-backward cycle + 有效重投影比例
- `R_dynamic`：动态区域覆盖率 + 3D 轨迹加速度 penalty + 速度平滑度
- `R_motion`：相机加速度 penalty + 旋转加速度 penalty + motion gate + 瞬移 penalty
- 所有 checkpoint（early/mid/final）使用完全相同的 compute_reward 函数，区别只是输入帧数

### 与 V1 的区别

- 几何来源：DA3（3D，静态假设）→ 4RC（4D，原生动态支持）
- 动态分割：帧差+深度变化率（手工）→ 4RC track 位移后验推断
- 早期 Reward：简化公式 → 统一公式（帧数不同而已）
- conf 角色：无 → 仅作为 valid mask（不作为分数）
- 可微性：不可微 → 4RC forward 支持梯度（Phase 3）

### Dynamic Mask（逐帧最大位移法）

从 track 的逐帧最大位移推断 static/dynamic：`max_t ||track[t]|| > threshold × scene_scale`

### 显存管理：模型交替加载

```python
# 去噪阶段：DiT on GPU, 4RC on CPU
# Checkpoint 评分阶段：DiT on CPU, 4RC on GPU
```

`GeoRewardBoNProgressiveV2` 自动处理模型交替加载。

### ReconRewardConfig 默认值

```python
static_weight = 0.40          # R_static 权重
dynamic_weight = 0.40         # R_dynamic 权重
motion_weight = 0.20          # R_motion 权重
dynamic_threshold_ratio = 0.01  # 动态 mask 相对 scene_scale 的阈值
tau_reproj = 0.10             # 重投影误差温度
tau_accel = 0.05              # 加速度 penalty 温度
tau_speed = 3.0               # 极端速度 penalty 温度
tau_cam = 0.02                # 相机加速度温度
tau_rot = 0.05                # 旋转加速度温度
min_motion = 0.005            # 最低运动量
conf_valid_quantile = 0.20    # conf 有效阈值分位数（Q20，保留 top 80%）
image_size = 518              # 4RC 输入分辨率
```

### 梯度引导（Phase 3，待验证后启用）

```python
L_guidance = L_reproj + 0.5 × L_track_smoothness + 0.3 × L_anchor
```

- 对显式几何 loss 求梯度，不对 conf 求梯度
- WMReward 风格归一化：`v_guided = v_pred + scale * norm_ratio * scaling_t * grad`
- `GeometricGuidance` 类封装，通过 `should_guide(sigma_t)` 控制引导窗口

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

## Tree Branching BoN 加速

### 核心思想

将渐进淘汰 BoN 的前半段从「8 条独立轨迹」改为「K 条共享主干 + 分叉扩展」，利用高噪声阶段的结构趋同性共享计算。相比渐进淘汰再省 ~29% DiT 步数。

### 流程（shift=5.0, branch_sigma=0.90）

```
Step 0 → 16:   K=2 条主干轨迹并行去噪          [2×16 = 32 步]
Step 16:        每条主干分叉为 4 条 → 共 8 条    [分叉点]
Step 16 → 22:  8 条候选独立去噪                 [8×6 = 48 步]
Step 22:        评分，淘汰 50% → 4 条           [Early Checkpoint]
Step 22 → 31:  4 条候选去噪                     [4×9 = 36 步]
Step 31:        评分，淘汰 50% → 2 条           [Mid Checkpoint]
Step 31 → 40:  2 条候选去噪                     [2×9 = 18 步]
Step 40:        最终评分，选出 best              [Final]
总计：32 + 48 + 36 + 18 = 134 步（vs 渐进 190 步，朴素 320 步）
```

### 分叉噪声注入公式

```python
z_branch = sqrt(1 - η²) * z_trunk + η * σ_t * ε,    ε ~ N(0, I)
```

- 方差保持：扰动后 latent 仍在正确噪声水平 manifold 上
- σ_t 缩放：扰动大小与当前噪声水平成正比
- 理论保证：Kim et al. (2025) ODE-to-SDE 边缘分布一致性

### σ-based 分叉点（动态映射步数）

通过 `find_step_for_sigma(state, branch_sigma)` 动态确定分叉步，适应不同 shift/步数。

### 默认参数

```python
DEFAULT_NUM_TRUNKS = 2              # 主干数量
DEFAULT_BRANCHES_PER_TRUNK = 4      # 每主干分叉数（总候选 N=8）
DEFAULT_BRANCH_SIGMA = 0.90         # 分叉点 σ（动态映射步数）
DEFAULT_BRANCH_ETA = 0.10           # 多样性超参数 η
```

### 约束

- `branch_sigma` 必须 > `max(sigma_checkpoints)`，否则分叉后无法正常渐进淘汰
- η 推荐 0.05~0.15；> 0.20 可能偏离训练分布产生 artifacts
- `branch_sigma` 建议 ≤ 0.93，否则分叉后仍有 high_noise_model 步（增加模型搬运开销）

### 计算量对比

| 方案 | DiT 步数 | VAE 解码 | Reward 推理 |
|------|----------|----------|-------------|
| 朴素 BoN (N=8) | 320 | 8 | 8 |
| 渐进淘汰 V2 | 190 | 14 | 14 |
| **Tree Branching** | **134** | **14** | **14** |

## 远程服务器路径

```
项目目录:  /pfs/mayuema/spj/now
Wan2.2 权重: /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B
DA3 模型:   /pfs/mayuema/spj/DA3/models/DA3NESTED-GIANT-LARGE-1.1
4RC 模型:   /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC
输入图片:   /pfs/mayuema/spj/now/inputs/<子目录>/
```

## 运行命令

```bash
# 安装
pip install -r requirements.txt
pip install -e Depth-Anything-3-main/
export PYTHONPATH=$PYTHONPATH:$(pwd)/Wan2.2

# Best-of-N 渐进淘汰（单条，默认）
python run_bon.py \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832 --sample_shift 3.0 --t5_cpu

# Best-of-N 渐进淘汰（批量）
python run_bon_batch.py \
  --start 1 --end 10 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --da3_model /pfs/mayuema/spj/DA3/models/DA3NESTED-GIANT-LARGE-1.1 \
  --t5_cpu

# 自定义淘汰参数
python run_bon_batch.py \
  --start 1 --end 10 \
  --sigma_checkpoints 0.83 0.63 \
  --elimination_ratio 0.5 \
  --min_survivors 2 \
  --score_epsilon 0.02 \
  --early_max_frames 12 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --da3_model /pfs/mayuema/spj/DA3/models/DA3NESTED-GIANT-LARGE-1.1 \
  --t5_cpu

# 禁用渐进淘汰（原版顺序 BoN）
python run_bon.py --no_progressive \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
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

# ===== V2（4RC）运行命令 =====

# V2 Best-of-N 渐进淘汰（单条，默认）
python run_bon_v2.py \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832 --sample_shift 3.0 --t5_cpu

# V2 禁用渐进淘汰（顺序 BoN，8 个全部跑完）
python run_bon_v2.py --no_progressive \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832 --sample_shift 3.0 --t5_cpu

# V2 批量 BoN（渐进淘汰）
python run_bon_batch_v2.py \
  --start 1 --end 10 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json \
  --name_prefix test_real \
  --t5_cpu --sample_shift 3.0

# V2 批量 BoN（禁用渐进，全量生成 + 评分）
python run_bon_batch_v2.py \
  --start 1 --end 24 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json \
  --name_prefix test_real \
  --no_progressive \
  --N 8 --t5_cpu --sample_shift 3.0

# V2 自定义淘汰参数 + reward 参数
python run_bon_batch_v2.py \
  --start 1 --end 10 \
  --sigma_checkpoints 0.83 0.63 \
  --elimination_ratio 0.5 \
  --min_survivors 2 \
  --score_epsilon 0.02 \
  --early_max_frames 12 \
  --static_weight 0.40 --dynamic_weight 0.40 --motion_weight 0.20 \
  --tau_reproj 0.10 --tau_accel 0.05 --tau_speed 3.0 \
  --tau_cam 0.02 --tau_rot 0.05 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json \
  --name_prefix test_real \
  --t5_cpu --sample_shift 3.0

# V2 独立视频评分
python score_video_v2.py \
  --video path/to/video.mp4 \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC

python score_video_v2.py \
  --video path/to/video.pt \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --output_dir results/

# V2 Reward 参数可通过 CLI 调整，例如：
python score_video_v2.py --video path/to/video.mp4 \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --static_weight 0.40 --dynamic_weight 0.40 --motion_weight 0.20 \
  --tau_reproj 0.10 --tau_accel 0.05 \
  --dynamic_threshold_ratio 0.01 \
  --conf_valid_quantile 0.20

# V2 多卡并行批量（示例：3 卡各跑 24 个）
CUDA_VISIBLE_DEVICES=1 python run_bon_batch_v2.py --start 1 --end 24 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json --name_prefix test_real \
  --no_progressive --N 8 --t5_cpu --sample_shift 3.0

CUDA_VISIBLE_DEVICES=2 python run_bon_batch_v2.py --start 25 --end 48 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json --name_prefix test_real \
  --no_progressive --N 8 --t5_cpu --sample_shift 3.0

CUDA_VISIBLE_DEVICES=3 python run_bon_batch_v2.py --start 49 --end 72 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json --name_prefix test_real \
  --no_progressive --N 8 --t5_cpu --sample_shift 3.0

# ===== Tree Branching 加速 =====

# V2 Tree Branching（单条，推荐配置）
python run_bon_v2.py \
  --tree_branching \
  --num_trunks 2 --branches_per_trunk 4 \
  --branch_sigma 0.90 --branch_eta 0.10 \
  --sigma_checkpoints 0.83 0.63 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --size 480*832 --sample_shift 5.0 --t5_cpu

# V2 Tree Branching（批量）
python run_bon_batch_v2.py \
  --start 1 --end 24 \
  --tree_branching \
  --num_trunks 2 --branches_per_trunk 4 \
  --branch_sigma 0.90 --branch_eta 0.10 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json \
  --name_prefix test_tree \
  --t5_cpu --sample_shift 5.0

# V2 Tree Branching 多卡并行
CUDA_VISIBLE_DEVICES=1 python run_bon_batch_v2.py --start 1 --end 24 \
  --tree_branching --branch_sigma 0.90 --branch_eta 0.10 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json --name_prefix test_tree \
  --t5_cpu --sample_shift 5.0

CUDA_VISIBLE_DEVICES=2 python run_bon_batch_v2.py --start 25 --end 48 \
  --tree_branching --branch_sigma 0.90 --branch_eta 0.10 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --input_dir /pfs/mayuema/spj/now/inputs/inputs_real \
  --prompts batch_prompts_real.json --name_prefix test_tree \
  --t5_cpu --sample_shift 5.0
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
- Tree Branching 通过 `wan.branch_candidates()` 在分叉点用 ODE-to-SDE 噪声注入创建分支
- 分叉点由 `find_step_for_sigma(state, branch_sigma)` 动态确定（σ-based，适应不同 shift）
- Scheduler 深拷贝保留 UniPC/DPM++ 内部状态（model_outputs, step_index 等）

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

- **V1**（已完成代码）：DA3 Scene + Motion，显式区域分割 + 运动质量评分 + 两阶段渐进淘汰 BoN
- **V2**（当前，代码已完成）：4RC 显式几何一致性 Reward（ReconstructionReward）
  - Phase 0：Reward 信号验证（扰动退化实验 + 候选间方差分析 + V1 交叉验证）
  - Phase 1：完整 BoN（4RC reward 选优 vs random vs V1）
  - Phase 2：Progressive BoN（验证 early score 与 final score 相关性）
  - Phase 3：梯度引导 + BoN（GeometricGuidance 模块已就绪，待 Phase 1/2 验证）
- **Tree Branching**（代码已完成）：共享主干 + 分叉加速（GeoRewardBoNTreeBranching）
  - 相比渐进淘汰再省 ~29% DiT 步数（134 vs 190）
  - 待验证：η=0 退化测试、不同 η 多样性对比、加速比实测、质量对比
- **Part 2**（更后续）：梯度引导去噪 + BoN 组合 + 下游机械臂任务验证

## 代码规范

- 用中文交流，代码和文件名用英文
- 不做多余抽象，保持代码直接可读
