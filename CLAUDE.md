# GeoReward

利用 Depth Anything 3 (DA3) 的显式几何信息（深度、相机位姿、置信度）作为 Reward 信号，在 Wan2.2 I2V 视频生成模型的推理阶段提升生成视频的物理一致性。场景：机械臂操作。

## 项目结构

```
now/
├── geo_reward/                  # 核心模块
│   ├── __init__.py              # 导出 DA3GeoReward, GeometryRewardConfig, GeoRewardBoN
│   ├── da3_reward.py            # DA3GeoReward 主类 + GeometryRewardConfig
│   ├── region_masks.py          # 静态/动态区域分割（帧差+深度变化率+OR组合）
│   ├── motion_reward.py         # 运动质量评分（motion_gate + R_shape + R_smoothness）
│   ├── bon_pipeline.py          # GeoRewardBoN / GeoRewardBoNOffline
│   └── utils.py                 # 格式转换（wan→PIL）、均匀抽帧
├── run_bon.py                   # 单条 BoN CLI（--mode bon / score）
├── run_bon_batch.py             # 批量 BoN CLI（--start/--end，模型只加载一次）
├── score_video.py               # 独立视频评分 CLI（支持 .mp4/.pt）
├── batch_prompts.json           # 机械臂测试 prompt 集合
├── GeoReward_SchemeA_Geometric_Motion_Execution_Plan.md  # 当前执行方案（V1 Scene+Motion）
├── Wan2.2/                      # 阿里视频生成模型（14B DiT, Flow Matching）
├── Depth-Anything-3-main/       # ByteDance 深度估计模型（DA3）
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

### 区域分割

- RGB 帧差 + DA3 同像素深度 log-ratio 变化率
- OR 组合：任一信号检测到变化 → 动态（高召回）
- DA3 confidence 按帧内分位数做可靠性 mask，不作为正向 Reward
- 形态学开闭 + 连通区域过滤去噪点
- 同时维护 global mask（用于 R_scene）和 per-frame mask（用于 R_motion）

### 关键改进（相对旧版本）

- 旧版：全图 70% 截断隐式排除运动区域 → 静止视频也能拿高分
- 新版：显式分割后，静止视频 motion_gate ≈ 0，总分直接归零
- confidence 不再是独立 Reward 分量（旧版占 15%），只做深度可靠性 mask
- 投影误差从负值改为 [0, 1] 映射，可解释性更好

## 运行命令

```bash
# 安装
pip install -r requirements.txt
pip install -e Depth-Anything-3-main/
export PYTHONPATH=$PYTHONPATH:$(pwd)/Wan2.2

# Best-of-N（单条）
python run_bon.py \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832 --sample_shift 3.0 --t5_cpu

# Best-of-N（批量）
python run_bon_batch.py \
  --start 1 --end 10 \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --da3_model /path/to/DA3 \
  --t5_cpu

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

- **V1**（当前）：Scene + Motion，显式区域分割 + 运动质量评分 + Best-of-N
- **V2**（后续，依赖 V1 实验结果）：增加 R_interaction（机器人-物体交互一致性）
- **Part 2**（更后续）：可微 DA3 + 梯度引导去噪 + 与 BoN 组合

## 代码规范

- 用中文交流，代码和文件名用英文
- 不做多余抽象，保持代码直接可读
