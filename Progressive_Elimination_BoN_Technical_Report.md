# Progressive Elimination Best-of-N 技术文档

## 1. 背景与动机

### 1.1 传统 Best-of-N 的计算瓶颈

传统 Best-of-N（BoN）策略在视频生成领域的标准做法是：

```
for i in range(N):
    video_i = 完整生成（40步去噪 + VAE decode）
    reward_i = 计算 reward
best = argmax(reward_i)
```

对于 Wan2.2-I2V-A14B（14B 参数 DiT），单个候选的生成时间约 4-6 分钟（A100），N=8 时总耗时约 32-48 分钟。其中：

- **去噪占比 ~98%**：每步需 2 次 14B DiT forward pass（CFG），40 步 = 80 次 forward
- **VAE decode 占比 ~1-2%**：约 2-5 秒/次
- **DA3 reward 计算占比 ~1%**：约 3-8 秒/次

核心矛盾：计算资源的 N 倍线性增长 vs. 大部分"明显差"的候选早期就已经确定差了。

### 1.2 核心洞察

1. **Flow matching 的去噪过程具有"先结构后细节"的特性**——全局几何布局（物体位置、运动方向、深度分布）在去噪中前期就已确定，后期主要补充纹理和边缘细节。

2. **GeoReward V1 恰好主要依赖几何信息**——R_scene 评估投影一致性，R_motion 评估 3D 轨迹平滑度和形状稳定性。这些信号不需要精细纹理即可计算。

3. **VAE decode 的开销相对 DiT 可忽略**——14B DiT 一步 ≈ 100-300 倍 VAE decode 成本，因此在中间步骤插入"decode → 评分 → 淘汰"的开销极低。

---

## 2. 方案设计

### 2.1 整体架构：σ-based Checkpoint + 软淘汰

```
N=8 candidates 全部同步开始去噪
    │
    ├── Checkpoint 1（σ ≈ 0.65，约 step 24-25/40）
    │   ├── 对所有 N 个存活候选计算 pred_x0
    │   ├── VAE decode → DA3 → GeoReward V1
    │   ├── 软淘汰：只淘汰 total < mean - 1.5×std 的统计离群值
    │   ├── 保存所有候选的 checkpoint 视频到磁盘
    │   └── 剩余候选继续去噪
    │
    ├── Checkpoint 2（σ ≈ 0.45，约 step 28-30/40）
    │   ├── 同上流程
    │   ├── 此时几何结构更加确定，淘汰更有信心
    │   └── 剩余候选继续去噪
    │
    └── Final（σ = 0，step 40/40）
        ├── 完整 clean latent decode + 完整 Reward
        ├── 保存最终视频
        └── 选出 best

所有中间和最终视频均保存到磁盘，命名规则如下节所述。
```

### 2.2 为什么用 σ 而不是固定 step 作为 checkpoint 触发条件

Wan2.2 使用带 shift 的噪声调度。不同的 `sample_shift` 值和 `sampling_steps` 数会改变 step 与 σ 的对应关系：

| Step (40步, shift=3) | 大致 σ |
|----------------------|--------|
| 10                   | 0.90   |
| 15                   | 0.83   |
| 20                   | 0.74   |
| 24-25                | 0.63-0.65 |
| 28-30                | 0.47-0.54 |
| 40                   | 0      |

用 σ 定义 checkpoint 的好处：
- 更换 `sampling_steps`（如从 40 步改为 25 步）时，实验定义仍然一致
- 更换 `sample_shift` 时自动适应
- σ 直接反映了当前噪声水平的物理含义

### 2.3 pred_x0 的正确提取

#### 关键技术细节

Wan2.2 使用 flow matching，模型预测的是**速度场** v_θ(x_t, t)。clean latent 预测公式为：

```
pred_x0 = x_t - σ_t × v_θ(x_t, t)
```

其中：
- `x_t`：当前带噪声的 latent（**必须是 scheduler.step() 调用前的值**）
- `σ_t`：当前步对应的 sigma（从 `scheduler.sigmas[step_index]` 获取，**必须是该步 forward pass 时的 index**）
- `v_θ(x_t, t)`：模型输出的速度场（CFG 合并后的 `noise_pred`）

#### 为什么不能直接用 scheduler.step() 的返回值

FlowUniPC scheduler 的 `step()` 方法返回的是 **x_{t-1}**（下一步的带噪 latent），不是 pred_x0。虽然内部 `convert_model_output()` 会计算 x0_pred，但它被用于多步外推后映射回了 x_t 空间。

#### 为什么需要保存 step 前的 latent

`scheduler.step()` 调用后：
1. `scheduler.step_index` 已递增（指向下一步）
2. `candidate['latent']` 已更新为 x_{t-1}

因此 `denoise_candidates()` 在每步的最后一次迭代中同时保存：
- `last_noise_preds[cand_idx]`：CFG 合并后的 v_pred
- `last_pre_step_latents[cand_idx]`：step 前的 x_t

`extract_pred_x0()` 使用 `sigmas[step_index - 1]`（因为 step_index 已递增）。

### 2.4 淘汰策略：软淘汰

#### 设计原则

早期 pred_x0 的 reward 有噪声，不能做精确排序。策略是**只淘汰"确定差"的候选**：

```python
threshold = mean(totals) - k × std(totals)
# k 默认 1.5

淘汰条件：total < threshold
保底规则：始终保留至少 2 个候选
```

#### 为什么不用固定百分比淘汰

- 如果所有候选分数接近（std 小），threshold 会很接近 mean，几乎不淘汰任何候选——这是正确的行为，因为此时排序可信度低
- 如果有明显的离群差值（std 大），只有那些真正差的会被淘汰
- 避免了"误杀最终最佳候选"的风险

#### 参数 `elimination_std` 的含义

| 值   | 行为 |
|------|------|
| 1.0  | 激进：淘汰所有低于 mean - 1σ 的候选（约 16% 尾部） |
| 1.5  | 适中（默认）：淘汰 mean - 1.5σ 以下的（约 7% 尾部） |
| 2.0  | 保守：只淘汰 mean - 2σ 以下的极端离群值（约 2.5% 尾部） |

### 2.5 Scheduler 状态管理

Progressive elimination 要求所有候选**按步同步推进**，而不是一个跑完再跑下一个。每个候选维护独立的 scheduler 实例，包含：

| 状态 | 说明 |
|------|------|
| `latent` | 当前 x_t，shape (16, 21, 60, 104) |
| `scheduler` | FlowUniPCMultistepScheduler 实例 |
| `scheduler.model_outputs` | FlowUniPC 二阶历史缓存（最多 2 个） |
| `scheduler.last_sample` | corrector 步需要的上一步样本 |
| `scheduler.step_index` | 当前步索引 |
| `scheduler.lower_order_nums` | warmup 计数器 |
| `scheduler.timesteps` | 时间步序列 |
| `generator` | CUDA RNG Generator（保证确定性） |
| `seed` | 该候选的种子 |

单个候选状态约 ~16MB (fp16)，N=8 总计 ~128MB，相对 14B 模型权重可忽略。

### 2.6 同步推进的额外好处

传统顺序 BoN 中，每个候选独立经历 high-noise → low-noise 模型切换（boundary 在 σ=0.9）：

```
候选1: load high → offload high, load low → generate → offload low
候选2: load high → offload high, load low → generate → offload low
...共 N 次模型搬运
```

Progressive 模式中所有候选同步推进：

```
所有候选的 high-noise 步: load high 一次
所有候选的 low-noise 步: offload high, load low 一次
...只有 1 次模型搬运
```

这在 `offload_model=True` 时额外节省了 (N-1) 次 ~14B 参数的 CPU↔GPU 搬运时间。

---

## 3. 计算节省分析

### 3.1 DiT 步数节省

以 N=8, 40 步, 2 个 checkpoint 为例：

**传统 BoN**：
- 去噪总步数 = 8 × 40 = 320 步
- 每步 2 次 DiT forward = 640 次 14B forward pass

**Progressive Elimination（典型淘汰场景）**：

假设 checkpoint1（σ=0.65, ~step 25）淘汰 2 个，checkpoint2（σ=0.45, ~step 30）再淘汰 2 个：

```
Step 0-25:  8 候选 × 25 步 = 200 步
Step 25-30: 6 候选 × 5 步  = 30 步
Step 30-40: 4 候选 × 10 步 = 40 步
总计 = 270 步（节省 15.6%）
```

更激进的淘汰场景（checkpoint1 淘汰 3 个，checkpoint2 淘汰 2 个）：

```
Step 0-25:  8 候选 × 25 步 = 200 步
Step 25-30: 5 候选 × 5 步  = 25 步
Step 30-40: 3 候选 × 10 步 = 30 步
总计 = 255 步（节省 20.3%）
```

### 3.2 额外开销

| 开销项 | 次数 | 单次耗时 | 总计 |
|--------|------|----------|------|
| VAE decode（checkpoint） | ~14 次 | 2-5s | 28-70s |
| DA3 推理 | ~14 次 | 3-8s | 42-112s |
| Reward 计算 | ~14 次 | <1s | <14s |
| **额外开销合计** | | | **~70-196s** |

**节省的去噪时间**（以 A100, ~7s/步 估算）：
- 保守场景（省 50 步）：50 × 7s = **350s**
- 典型场景（省 65 步）：65 × 7s = **455s**

**净节省 ≈ 150-360 秒**（单个推理案例），约占总时间的 **10-25%**。

### 3.3 节省的核心价值

除了纯时间节省，progressive elimination 的真正价值在于：
1. **降低显存峰值**：不需要同时在显存中保留 N 个完整视频张量
2. **提供中间诊断信息**：checkpoint 视频可直观观察去噪过程的演进
3. **为后续优化提供数据**：checkpoint reward 与 final reward 的相关性分析可以指导更激进的优化策略

---

## 4. 视频保存策略

### 4.1 输出目录结构

```
{output_dir}/
└── {image_stem}_{timestamp}/                            # ← 子目录由 CLI 脚本创建
    ├── seed_42_checkpoint1_sigma0.6425.mp4      # 候选0, checkpoint1 pred_x0 decode
    ├── seed_42_checkpoint2_sigma0.4283.mp4      # 候选0, checkpoint2 pred_x0 decode
    ├── seed_42_final_sigma0.00.mp4              # 候选0, 最终 clean decode
    ├── seed_43_checkpoint1_sigma0.6425.mp4      # 候选1, checkpoint1
    ├── seed_43_checkpoint2_sigma0.4283.mp4      # 候选1, checkpoint2
    ├── seed_43_final_sigma0.00.mp4              # 候选1, 最终
    ├── seed_44_checkpoint1_sigma0.6425.mp4      # 候选2, checkpoint1
    │                                            # ← 候选2 在 checkpoint1 后被淘汰，无后续视频
    ├── seed_45_checkpoint1_sigma0.6425.mp4      # 候选3, checkpoint1
    ├── seed_45_checkpoint2_sigma0.4283.mp4      # 候选3, checkpoint2
    │                                            # ← 候选3 在 checkpoint2 后被淘汰，无 final
    ├── ...
    ├── seed_42_BEST.mp4                         # ← CLI 脚本额外拷贝的最佳候选
    └── rewards.json                             # 完整的评分日志
```

> **职责划分**：checkpoint/final 视频由 `GeoRewardBoNProgressive._save_video()` 保存；`BEST.mp4` 拷贝和 `rewards.json` 中的 `prompt`/`image`/`config` 字段由 `run_bon.py` / `run_bon_batch.py` 在 pipeline 返回后附加。

### 4.2 文件命名规则

```
seed_{seed}_{phase_name}.mp4
```

三个维度：
- **seed**：唯一标识一个候选（由 seed_base + i 生成）
- **phase_name**：标识去噪阶段
  - `checkpoint{N}_sigma{σ:.4f}`：第 N 个 checkpoint，σ 值精确到**四位**小数（如 `sigma0.6425`）
  - `final_sigma0.00`：最终完全去噪后的 clean decode
- **BEST 标记**：`seed_{best_seed}_BEST.mp4` 是最终选出的最佳候选的额外拷贝（由 CLI 脚本创建，非 pipeline 类内部行为）

### 4.3 从文件夹结构判断淘汰情况

| 候选拥有的文件 | 含义 |
|---------------|------|
| 只有 checkpoint1 | 在 checkpoint1 后被淘汰 |
| checkpoint1 + checkpoint2 | 在 checkpoint2 后被淘汰 |
| checkpoint1 + checkpoint2 + final | 存活到最后 |

### 4.4 Rewards.json 结构

```json
{
  "mode": "progressive_elimination",
  "sigma_checkpoints": [0.65, 0.45],
  "checkpoint_plan": [
    {
      "completed_steps": 26,
      "scheduler_step_index": 25,
      "target_sigma": 0.65,
      "actual_sigma": 0.6424896717071533
    },
    {
      "completed_steps": 33,
      "scheduler_step_index": 32,
      "target_sigma": 0.45,
      "actual_sigma": 0.42826521396636963
    },
    {
      "completed_steps": 40,
      "scheduler_step_index": 40,
      "target_sigma": 0.0,
      "actual_sigma": 0.0
    }
  ],
  "elimination_std": 1.5,
  "seeds": [42, 43, 44, 45, 46, 47, 48, 49],
  "best_seed": 42,
  "total_time_sec": 1847.3,
  "rewards": {
    "seed_42": {
      "checkpoint1_sigma0.6425": {
        "total": 0.72, "scene": 0.81, "motion": 0.68,
        "motion_gate": 0.65, "shape": 0.71, "smoothness": 0.62, ...
      },
      "checkpoint2_sigma0.4283": {
        "total": 0.78, "scene": 0.85, "motion": 0.74, ...
      },
      "final_sigma0.00": {
        "total": 0.83, "scene": 0.88, "motion": 0.80, ...
      }
    },
    "seed_43": { ... },
    "seed_44": {
      "checkpoint1_sigma0.6425": { "total": 0.31, ... }
    }
  },
  "eliminated_at": {
    "seed_44": "checkpoint1_sigma0.6425",
    "seed_45": "checkpoint1_sigma0.6425",
    "seed_47": "checkpoint2_sigma0.4283",
    "seed_48": "checkpoint2_sigma0.4283"
  },
  // ---- 以下字段由 run_bon.py CLI 脚本在 pipeline 返回后附加 ----
  "prompt": "pick up the red cube and place it on the left",
  "image": "/path/to/first_frame.png",
  "config": {
    "reward_version": "v1",
    "progressive": true,
    "sigma_checkpoints": [0.65, 0.45],
    "elimination_std": 1.5,
    "da3_model": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "process_res": 504,
    "max_frames": 20,
    "size": "480*832",
    "frame_num": 81,
    "sampling_steps": 40,
    "guide_scale": 5.0,
    "seed_base": 42
  }
}
```

> **注意**：`checkpoint_plan` 记录了每个 checkpoint 的实际调度信息——`target_sigma` 是你配置的目标值，`actual_sigma` 是 scheduler 离散 sigma 序列中实际对应的值（文件名使用 actual_sigma 四位小数截断）。`prompt`/`image`/`config` 字段由 CLI 脚本附加，不属于 `GeoRewardBoNProgressive._build_result_log()` 的输出。

### 4.5 视频内容说明

| 阶段 | 视频来源 | 预期质量 |
|------|---------|----------|
| checkpoint1 (σ≈0.65) | pred_x0 经 VAE decode | 全局布局可辨认，但细节模糊、可能有伪影，颜色可能偏移 |
| checkpoint2 (σ≈0.45) | pred_x0 经 VAE decode | 结构清晰，运动方向明确，但纹理仍有噪声感 |
| final (σ=0.00) | clean latent 经 VAE decode | 完整质量，与传统 BoN 的输出完全一致 |

checkpoint 视频质量会随 σ 降低而提升。这些视频的主要价值是：
1. **人工验证**：直观确认淘汰决策是否合理
2. **相关性分析**：比较 checkpoint reward 排序 vs final reward 排序
3. **调参依据**：决定最优的 checkpoint σ 位置

---

## 5. 实现细节

### 5.1 代码架构

```
Wan2.2/wan/image2video.py    (WanI2V 类新增方法)
    ├── prepare_progressive()      # 初始化 N 个候选的状态
    ├── denoise_candidates()       # 分段去噪 + 返回最后步的 v_pred
    ├── extract_pred_x0()          # 计算 clean latent 预测
    ├── find_step_for_sigma()      # σ → step 映射
    ├── decode_latent()            # 单 latent VAE decode
    ├── get_current_sigma()        # 获取某候选当前 σ 值
    └── cleanup_progressive()      # 释放状态

geo_reward/bon_pipeline.py   (新增 GeoRewardBoNProgressive 类)
    ├── generate()                 # 主流程：循环 checkpoint → denoise → score → eliminate
    ├── _generate_prepared()       # 已初始化状态上的去噪主循环
    ├── _eliminate()               # 软淘汰逻辑
    ├── _save_video()              # 视频保存（checkpoint + final）
    └── _build_result_log()        # 构建 JSON 日志（不含 prompt/image/config）

run_bon.py                   (CLI, --no_progressive 开关)
    └── 负责：创建 case_dir, BEST.mp4 拷贝, 附加 prompt/image/config 到 result_log
run_bon_batch.py             (批量 CLI, 同上)
geo_reward/__init__.py       (导出 GeoRewardBoNProgressive)
```

### 5.2 关键 API

#### 初始化

```python
from geo_reward import GeoRewardBoNProgressive

bon = GeoRewardBoNProgressive(
    wan_i2v=wan_i2v,           # WanI2V 实例
    da3_reward=da3_reward,     # DA3GeoReward 实例
    max_frames=20,             # DA3 评分使用的关键帧数
    sigma_checkpoints=[0.65, 0.45],  # σ checkpoint 阈值
    elimination_std=1.5,       # 淘汰严格度
)
```

#### 调用

```python
best_video, result_log, best_seed = bon.generate(
    prompt="pick up the red cube",
    image=pil_image,
    N=8,
    frame_num=81,
    seed_base=42,
    output_dir="outputs/case_001",
    save_fn=my_save_function,    # callable(tensor, path)
    # 以下透传给 Wan2.2
    max_area=480*832,
    shift=3.0,
    sample_solver="unipc",
    sampling_steps=40,
    guide_scale=5.0,
    offload_model=True,
)
```

### 5.3 CLI 使用

```bash
# Progressive 模式（默认）
python run_bon.py \
    --ckpt_dir /path/to/Wan2.2-I2V-A14B \
    --image /path/to/first_frame.png \
    --prompt "pick up the red cube" \
    --N 8 \
    --size 480*832 \
    --sample_shift 3.0 \
    --t5_cpu

# 自定义 checkpoint
python run_bon.py \
    --ckpt_dir /path/to/Wan2.2-I2V-A14B \
    --image /path/to/first_frame.png \
    --prompt "pick up the red cube" \
    --N 8 \
    --sigma_checkpoints 0.70 0.50 0.30 \
    --elimination_std 2.0

# 禁用 progressive，使用原始顺序 BoN
python run_bon.py \
    --ckpt_dir /path/to/Wan2.2-I2V-A14B \
    --image /path/to/first_frame.png \
    --prompt "pick up the red cube" \
    --N 8 \
    --no_progressive

# 批量运行（Progressive 模式）
python run_bon_batch.py \
    --start 1 --end 10 \
    --ckpt_dir /path/to/Wan2.2-I2V-A14B \
    --da3_model /path/to/DA3 \
    --t5_cpu
```

---

## 6. 技术风险与注意事项

### 6.1 pred_x0 在高 σ 时的不稳定性

当 σ 较高时（如 0.65），pred_x0 = x_t - σ×v 中 σ 很大，v 的微小误差会被放大。这意味着：

- checkpoint1 的 reward **绝对值**可能与 final 差异很大
- 但**候选间的相对排序**仍然可能有效（这是核心假设）
- 软淘汰策略（只淘汰离群值）是对这种不确定性的自然对冲

### 6.2 Wan2.2 的 high/low noise 双模型边界

Wan2.2 在 σ=0.9（boundary=900）处切换模型。σ=0.65 的 checkpoint 意味着 low-noise 专家已经工作了约 10-15 步（从 step 10 到 step 25），几何结构此时**应该**已经初步稳定。

但这不是理论保证——对于特别复杂的场景（多物体交互、大幅度运动），结构可能需要更多步才能稳定。建议：
- 首轮实验使用较保守的 σ=0.55 作为第一个 checkpoint
- 通过相关性分析确定最优位置后再调整

### 6.3 FlowUniPC 多步求解器的状态连续性

FlowUniPC 是二阶求解器，内部缓存了前几步的 model output。progressive 模式中所有候选的 scheduler 是独立实例，天然保持状态连续，不需要额外处理。

唯一需要注意的是：不能在不同候选之间**共享** scheduler 实例。

### 6.4 显存估算

在 checkpoint 步需要临时多出的显存：

| 项目 | 大小 |
|------|------|
| pred_x0 latent | ~4MB (fp16) |
| VAE decoded 视频 (3, 81, 480, 832) | ~194MB (fp16) |
| DA3 推理中间状态 | ~500MB-1GB |

由于 checkpoint 评分是**逐候选串行**的（decode A → score → del → decode B），峰值额外显存仅为单个候选的解码视频 + DA3 状态，约 ~1.2GB。相对 14B DiT 模型占用的 ~28GB 显存可以接受。

---

## 7. 验证计划

### 7.1 核心假设验证

**假设**：在 σ ≈ 0.65 和 σ ≈ 0.45 时，pred_x0 的 GeoReward 排序与最终排序高度相关。

**验证方法**：
1. 对 M ≥ 10 个 prompt，每个生成 N=8 个候选
2. 在 σ ∈ {0.75, 0.65, 0.55, 0.45, 0.35} 各计算一次 reward
3. 计算每个 σ 下的 Spearman ρ 和 Kendall τ（与 final 对比）
4. 额外分析 motion_gate 的早期稳定性

**成功标准**：
- σ=0.65 时 Spearman ρ > 0.6（排序大致正确）
- σ=0.45 时 Spearman ρ > 0.8（排序基本可靠）
- 软淘汰不会误杀最终 top-2 候选的概率 > 95%

### 7.2 消融实验

| 变量 | 取值范围 | 目的 |
|------|---------|------|
| sigma_checkpoints | [0.65], [0.65, 0.45], [0.55] | 找最优 checkpoint 数量和位置 |
| elimination_std | 1.0, 1.5, 2.0, 3.0 | 找最优淘汰激进度 |
| N | 4, 8, 16 | 测试不同候选数下的效果 |

---

## 8. 与原始 BoN 的对比

| 维度 | 原始 BoN | Progressive Elimination |
|------|---------|------------------------|
| 去噪方式 | 候选逐个顺序完成 | 所有候选同步推进 |
| 评分时机 | 全部生成完后评分 | 多个 checkpoint + final |
| 计算量 | N × 40 步 (固定) | 动态减少（典型节省 15-25%） |
| 模型搬运次数 | N 次 high↔low 切换 | 1 次 |
| 输出内容 | 只有 final 视频 | checkpoint + final 全部保存 |
| 风险 | 无（完整评估） | 可能误杀（通过软淘汰缓解） |
| 可观测性 | 只有最终 reward | 全过程 reward 演进 |
| CLI 参数 | --N, --seed_base | + --sigma_checkpoints, --elimination_std |
| 开关 | 默认 | 默认开启，--no_progressive 回退 |

---

## 9. 后续优化方向

### 9.1 自适应 checkpoint（基于验证数据）

当积累了足够的相关性数据后，可以让 checkpoint 位置根据当前 prompt 的运动量自动调整：
- 大运动 prompt：更早 checkpoint（几何信号更早稳定）
- 精细操作 prompt：更晚 checkpoint（需要更多步确定细节）

### 9.2 Latent-space motion proxy（零成本预筛）

```python
# 在 latent shape (16, 21, 60, 104) 上直接计算
temporal_diff = (latent[:, 1:] - latent[:, :-1]).abs().mean(dim=(0, 2, 3))
motion_proxy = temporal_diff.median()
```

如果 motion_proxy ≈ 0，这个候选几乎没有运动，可以零成本淘汰（不需要 VAE decode 或 DA3）。

### 9.3 空间下采样 preview（降低 decode 成本）

将 pred_x0 空间 2x 下采样后 decode，产出 240×416 分辨率的 preview：
- Decode 计算量降为 ~25%
- 仍覆盖全部 81 帧的时间范围
- 适合作为 checkpoint2 之后的进一步优化

### 9.4 与 V2 reward 的结合

V2 计划增加 R_interaction（机器人-物体交互一致性）。交互发生的时间段通常在视频中后段，这意味着：
- checkpoint1 可能不适合评估 interaction
- 但 motion_gate 和 R_scene 在 checkpoint1 仍然有效
- 可以设计分层淘汰：checkpoint1 用 scene+gate，checkpoint2 用完整 V2 reward
