# Tree Branching BoN 加速方案 — 具体实现计划

## 1. 方案总览

### 核心思想

将渐进淘汰 BoN 的前半段从「8 条独立轨迹」改为「2 条共享主干 + 分叉扩展」，利用高噪声阶段的结构趋同性共享计算。

### 计算量对比

| 方案 | DiT 步数 | VAE 解码 | Reward 推理 | 模型搬运 |
|------|----------|----------|-------------|----------|
| 朴素 BoN (N=8) | 320 | 8 | 8 | 2 |
| 渐进淘汰 V2 | 190 | 14 | 14 | 6 |
| **Tree Branching** | **134** | **14** | **14** | **6** |

Tree Branching 相比渐进淘汰再省 29.5% DiT 步数（134 vs 190），相比朴素省 58.1%。
（以默认 shift=5.0, branch_sigma=0.90 计算。不同 shift 下略有差异，见下文详细计算。）

### 时间线

分叉点由 `branch_sigma` 参数动态确定（通过 `find_step_for_sigma()` 查询实际步号），不硬编码步数。以下以 40 步、shift=5.0、`branch_sigma=0.90` 为例：

```
Step 0 → 16 (σ=1.0→0.89):  K=2 条主干轨迹并行去噪     [2×16 = 32 步]
    │
Step 16 (σ≈0.89):          每条主干分叉为 4 条 → 共 8 条  [分叉点]
    │
Step 16 → 22 (σ→0.82):     8 条候选独立去噪             [8×6 = 48 步]
    │
Step 22 (σ≈0.82):          评分，淘汰 50% → 4 条        [Early Checkpoint]
    │
Step 22 → 31 (σ→0.63):     4 条候选去噪                 [4×9 = 36 步]
    │
Step 31 (σ≈0.63):          评分，淘汰 50% → 2 条        [Mid Checkpoint]
    │
Step 31 → 40 (σ→0):        2 条候选去噪                 [2×9 = 18 步]
    │
Step 40:                    最终评分，选出 best           [Final]
```

总计：32 + 48 + 36 + 18 = **134 步**

**`find_step_for_sigma()` 语义**：返回第一个 `σ[i] <= target` 的 `i+1`（exclusive end_step）。

以 **shift=5.0, 40 步** 为例：
- `branch_sigma=0.90` → σ[15]=0.893 ≤ 0.90 → branch_step = **16**
- `sigma_checkpoint=0.83` → σ[21]=0.819 ≤ 0.83 → early_cp = **22**
- `sigma_checkpoint=0.63` → σ[30]=0.625 ≤ 0.63 → mid_cp = **31**
- 总计：2×16 + 8×(22−16) + 4×(31−22) + 2×(40−31) = 32+48+36+18 = **134 步**

以 **shift=5.0, 40 步, `branch_sigma=0.93`** 为例：
- σ[11]=0.929 ≤ 0.93 → branch_step = **12**
- 总计：2×12 + 8×(22−12) + 4×(31−22) + 2×(40−31) = 24+80+36+18 = **158 步**

以 **shift=3.0, 40 步** 为例：
- `branch_sigma=0.90` → σ[10]=0.900 ≤ 0.90 → branch_step = **11**
- `sigma_checkpoint=0.83` → σ[16]=0.818 ≤ 0.83 → early_cp = **17**
- `sigma_checkpoint=0.63` → σ[26]=0.618 ≤ 0.63 → mid_cp = **27**
- 总计：2×11 + 8×(17−11) + 4×(27−17) + 2×(40−27) = 22+48+40+26 = **136 步**

---

## 2. 数学基础

### 2.1 Wan2.2 Flow Matching 回顾

Wan2.2 使用 rectified flow（线性插值）：

```
x_t = (1 - σ_t) * x_0 + σ_t * ε      (前向过程)
x_0_pred = x_t - σ_t * v_pred          (模型预测清晰帧)
```

sigma 调度使用 shifted linspace：

```python
σ_raw = linspace(1, 0, steps+1)[:steps]        # [1..near_0]
σ = shift * σ_raw / (1 + (shift - 1) * σ_raw)  # shift=5.0 (默认)
```

**注意**：项目默认 `sample_shift=5.0`（配置文件 `wan_i2v_A14B.py` 及 CLI 默认值），不是 3.0。

对应关系（40步, **shift=5.0**）：

```python
# σ_raw[i] = 1 - i/40
# σ[i] = 5 * σ_raw / (1 + 4 * σ_raw)
# t[i] = σ[i] * 1000
#
# i=0:  σ=1.000  t=1000  high_noise_model
# i=5:  σ=0.945  t=945   high_noise_model
# i=10: σ=0.938  t=938   high_noise_model  ← 仍在 high 域!
# i=14: σ=0.903  t=903   high_noise_model  ← 最后一步 high
# i=15: σ=0.893  t=893   low_noise_model   ← 首步 low
# i=20: σ=0.833  t=833   low_noise_model   ← σ=0.833 > 0.83!
# i=21: σ=0.819  t=819   low_noise_model   ← σ≤0.83 → Early CP end_step=22
# i=25: σ=0.750  t=750   low_noise_model
# i=30: σ=0.625  t=625   low_noise_model   ← σ≤0.63 → Mid CP end_step=31
# i=39: σ→0      t→0     low_noise_model
```

**高噪声/低噪声模型分界**：config `boundary=0.900`，实际比较时乘以 `num_train_timesteps=1000` → `boundary=900`。判定逻辑：`t.item() >= 900 → high_noise_model`（其中 `t = σ × 1000`）。

以 shift=5.0 计算：
- σ[14]=0.903 → t=903 ≥ 900 → high_noise_model（最后一步）
- σ[15]=0.893 → t=893 < 900 → low_noise_model（首步）

因此 Step 0-14 使用 high_noise_model，Step 15+ 使用 low_noise_model。

**`find_step_for_sigma` 返回值语义**：返回 `i+1`（exclusive end_step），即"运行完 Step i 后停下"。例如 target_sigma=0.90 → σ[15]=0.893≤0.90 → 返回 16，表示需要跑完 Step 0..15（共 16 步）。

### 2.2 分叉点选择策略（σ-based 而非 step-based）

**核心设计决策**：不硬编码分叉步数，改用 `branch_sigma` 参数 + `find_step_for_sigma()` 动态确定分叉步。

这确保在不同 shift 值和 sampling_steps 下都能正确工作：

```python
branch_step = self.wan.find_step_for_sigma(state, self.branch_sigma)
```

**推荐 `branch_sigma=0.90`**（恰好在模型切换点之后），理由：

1. **与模型切换点对齐**：`find_step_for_sigma(0.90)` 返回 16（shift=5.0 时），即运行完 Step 0-15 后分叉。此时 Step 0-14 全程 high_noise_model，Step 15 刚用 low_noise_model 走了一步。分叉后全程 low_noise_model。
2. **全局布局已定**：到达 σ≈0.89 时，全局空间布局（相机角度、物体大致位置）已确定，但细节远未收敛。分叉既保留全局一致性，又允许细节多样性。
3. **高 σ 仍可承受扰动**：σ≈0.89 意味着 latent 中约 89% 仍是噪声成分，注入 η×σ 级别的扰动不会破坏信号。

### 2.3 分叉噪声注入公式

基于 Kim et al. (2025) 的 ODE-to-SDE 转换理论，在分叉点注入扰动：

```python
z_branch = sqrt(1 - η²) * z_trunk + η * σ_t * ε,    ε ~ N(0, I)
```

其中：
- `z_trunk`: 主干轨迹在分叉点的 latent 状态
- `σ_t`: 分叉点的噪声水平（由 scheduler 查询得到）
- `η`: 多样性超参数（控制分叉后的发散程度）
- `ε`: 独立标准正态噪声

**为什么这个公式有效：**

1. **方差保持**：`Var(z_branch) = (1-η²) * Var(z_trunk) + η² * σ_t² * I`。当 η 较小时，扰动后的 latent 仍在正确的噪声水平 manifold 上。
2. **σ_t 缩放**：扰动大小与当前噪声水平成正比。高噪声阶段（σ 大）结构粗糙，可承受较大扰动。
3. **理论保证**：Kim et al. 证明该操作保持了 p_t(x) 的边缘分布一致性。

### 2.4 η 取值建议

| η 值 | 效果 | 适用场景 |
|-------|------|----------|
| 0.05 | 微小差异，候选高度相似 | 保守，当 N 很大时 |
| 0.10 | 适度多样性（推荐默认） | 一般场景 |
| 0.15 | 较大多样性 | 需要探索更多可能性 |
| 0.20 | 高多样性，可能偏离 manifold | 实验性，需验证 |

**推荐 η=0.10** 作为默认值。原因：
- 分叉点 σ≈0.90，实际扰动幅度 = η × σ = 0.09，相对于 latent 本身的噪声幅度 σ=0.90 约为 10%
- 足够创造视觉可区分的差异，同时不会严重偏离训练分布

---

## 3. 详细代码改动

### 3.1 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `Wan2.2/wan/image2video.py` | 新增方法 | `branch_candidates()` |
| `geo_reward/bon_pipeline.py` | 新增类 | `GeoRewardBoNTreeBranching` |
| `geo_reward/__init__.py` | 修改 | 导出 `GeoRewardBoNTreeBranching` |
| `run_bon_v2.py` | 修改 | 增加 `--tree_branching` 开关和相关参数 |
| `run_bon_batch_v2.py` | 修改 | 增加 `--tree_branching` 开关和相关参数 |

### 3.2 Wan2.2/wan/image2video.py — 新增 `branch_candidates()` 方法

```python
def branch_candidates(self, state, trunk_indices, branches_per_trunk, eta, branch_seeds):
    """
    从 K 条主干轨迹分叉出 K * branches_per_trunk 条子轨迹。
    
    Args:
        state: prepare_progressive() 返回的状态 dict
        trunk_indices: List[int]，主干候选的索引（通常 [0, 1]）
        branches_per_trunk: int，每条主干分叉数（通常 4）
        eta: float，多样性超参数
        branch_seeds: List[int]，每个分支的随机种子（长度 = K * branches_per_trunk）
    
    Returns:
        更新后的 state（candidates 列表扩展到 N 个）
    """
    import copy
    import math
    
    new_candidates = []
    branch_idx = 0
    
    for trunk_idx in trunk_indices:
        trunk_cand = state['candidates'][trunk_idx]
        z_trunk = trunk_cand['latent'].clone()  # 确保不修改原 trunk
        
        # 获取当前 sigma（scheduler.step_index 指向下一步要用的 sigma）
        step_idx = trunk_cand['scheduler'].step_index
        sigma_t = float(trunk_cand['scheduler'].sigmas[step_idx])
        
        for b in range(branches_per_trunk):
            seed = branch_seeds[branch_idx]
            generator = torch.Generator(device=z_trunk.device).manual_seed(seed)
            epsilon = torch.randn(z_trunk.shape, generator=generator,
                                  device=z_trunk.device, dtype=z_trunk.dtype)
            
            # 核心分叉公式
            z_branch = math.sqrt(1 - eta**2) * z_trunk + eta * sigma_t * epsilon
            
            # 深拷贝 scheduler（包含 model_outputs, lower_order_nums, last_sample 等）
            new_scheduler = copy.deepcopy(trunk_cand['scheduler'])
            
            new_candidates.append({
                'latent': z_branch,
                'scheduler': new_scheduler,
                'generator': generator,
                'seed': seed,
                'step_index': step_idx,
                'parent_trunk': trunk_idx,
            })
            branch_idx += 1
    
    state['candidates'] = new_candidates
    return state
```

**不需要** `prepare_tree_branching()` 包装方法 —— 直接复用 `prepare_progressive()` 传入 trunk_seeds 即可。

### 3.3 geo_reward/bon_pipeline.py — 新增 GeoRewardBoNTreeBranching 类

```python
import random
import time
import math

class GeoRewardBoNTreeBranching(GeoRewardBoNProgressiveV2):
    """
    Tree Branching + Progressive Elimination BoN.
    
    前半段（Step 0 → branch_sigma 对应步数）只运行 K 条主干轨迹，
    在分叉点分叉为 N 条候选，后半段复用渐进淘汰逻辑。
    """
    
    DEFAULT_NUM_TRUNKS = 2
    DEFAULT_BRANCHES_PER_TRUNK = 4  # 2 * 4 = 8 总候选
    DEFAULT_BRANCH_SIGMA = 0.90     # σ-based 分叉点（动态映射到步数）
    DEFAULT_BRANCH_ETA = 0.10
    
    def __init__(self, wan_i2v, recon_reward, 
                 num_trunks=None, branches_per_trunk=None,
                 branch_sigma=None, branch_eta=None,
                 frame_indices=None, max_frames=20,
                 sigma_checkpoints=None, elimination_ratio=None,
                 min_survivors=None, score_epsilon=None,
                 early_max_frames=None, offload_models=True):
        """
        Args:
            num_trunks: 主干轨迹数量（默认 2）
            branches_per_trunk: 每条主干的分叉数（默认 4）
            branch_sigma: 分叉点的目标 σ 值（默认 0.90，动态求步数）
            branch_eta: 分叉多样性超参数（默认 0.10）
            其余参数同 GeoRewardBoNProgressiveV2
        """
        super().__init__(
            wan_i2v=wan_i2v, recon_reward=recon_reward,
            frame_indices=frame_indices, max_frames=max_frames,
            sigma_checkpoints=sigma_checkpoints,
            elimination_ratio=elimination_ratio,
            min_survivors=min_survivors,
            score_epsilon=score_epsilon,
            early_max_frames=early_max_frames,
            offload_models=offload_models,
        )
        self.num_trunks = num_trunks or self.DEFAULT_NUM_TRUNKS
        self.branches_per_trunk = branches_per_trunk or self.DEFAULT_BRANCHES_PER_TRUNK
        self.branch_sigma = branch_sigma if branch_sigma is not None else self.DEFAULT_BRANCH_SIGMA
        self.branch_eta = branch_eta if branch_eta is not None else self.DEFAULT_BRANCH_ETA
    
    def generate(self, prompt, image, N, frame_num, seed_base, 
                 output_dir, save_fn, **wan_kwargs):
        """
        Tree Branching 生成流程。
        
        返回值与父类一致：(best_video, result_log, best_seed)
        """
        # --- seed 处理（与父类相同逻辑）---
        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)
        
        expected_N = self.num_trunks * self.branches_per_trunk
        if N != expected_N:
            print(f"[TreeBranching] Warning: N={N} != num_trunks({self.num_trunks}) "
                  f"* branches_per_trunk({self.branches_per_trunk}) = {expected_N}. "
                  f"Using {expected_N}.")
            N = expected_N
        
        trunk_seeds = [seed_base + i for i in range(self.num_trunks)]
        branch_seeds = [seed_base + 100 + i for i in range(N)]
        
        # --- 只初始化 K 条主干（使用关键字参数调用）---
        state = self.wan.prepare_progressive(
            input_prompt=prompt,
            img=image,
            seeds=trunk_seeds,
            frame_num=frame_num,
            **wan_kwargs,
        )
        
        try:
            return self._generate_tree(
                state, N, branch_seeds, frame_num, output_dir, save_fn
            )
        finally:
            self.wan.cleanup_progressive(state)
    
    def _generate_tree(self, state, N, branch_seeds, frame_num,
                       output_dir, save_fn):
        """Tree Branching 核心逻辑。"""
        t_start = time.time()
        
        # 动态求分叉步数
        branch_step = self.wan.find_step_for_sigma(state, self.branch_sigma)
        if branch_step is None:
            raise ValueError(
                f"branch_sigma={self.branch_sigma} 无法映射到有效步数，"
                f"请检查 sigma 调度是否包含该值")
        
        print(f"[TreeBranching] branch_sigma={self.branch_sigma} → "
              f"branch_step={branch_step}")
        
        # === Phase 1: 主干去噪 (Step 0 → branch_step) ===
        trunk_indices = list(range(self.num_trunks))
        print(f"[TreeBranching] Phase 1: Denoising {self.num_trunks} trunks "
              f"for {branch_step} steps...")
        
        self.wan.denoise_candidates(state, trunk_indices, 0, branch_step)
        
        # === Phase 2: 分叉 ===
        print(f"[TreeBranching] Phase 2: Branching {self.num_trunks} trunks "
              f"into {N} candidates (eta={self.branch_eta})...")
        
        state = self.wan.branch_candidates(
            state, trunk_indices, self.branches_per_trunk,
            self.branch_eta, branch_seeds
        )
        
        # === Phase 3: 渐进淘汰 ===
        print(f"[TreeBranching] Phase 3: Progressive elimination from "
              f"step {branch_step}...")
        
        return self._progressive_elimination(
            state, N, branch_seeds, frame_num, output_dir, save_fn,
            start_step=branch_step, t_start=t_start
        )
    
    def _progressive_elimination(self, state, N, seeds, frame_num,
                                  output_dir, save_fn, start_step, t_start):
        """
        从 start_step 开始执行渐进淘汰。
        
        返回值：(best_video, result_log, best_seed) —— 与父类 generate() 一致。
        """
        total_steps = len(state['timesteps'])
        
        # 构建 checkpoint 列表（只保留 start_step 之后的）
        checkpoint_steps = []
        sigmas = state['candidates'][0]['scheduler'].sigmas
        for sigma_target in sorted(self.sigma_checkpoints, reverse=True):
            end_step = self.wan.find_step_for_sigma(state, sigma_target)
            if end_step is not None and end_step > start_step:
                checkpoint_steps.append({
                    "end_step": end_step,
                    "target_sigma": sigma_target,
                    "scheduler_step_index": end_step - 1,
                    "actual_sigma": float(sigmas[end_step - 1]),
                })
        checkpoint_steps.append({
            "end_step": total_steps,
            "target_sigma": 0.0,
            "scheduler_step_index": total_steps - 1,
            "actual_sigma": float(sigmas[total_steps - 1]),
        })
        
        # frame_indices
        from .utils import sample_frames, wan_output_to_pil
        early_frame_indices = sample_frames(frame_num, self.early_max_frames)
        normal_frame_indices = sample_frames(frame_num, self.max_frames)
        
        alive = list(range(N))
        eliminated_at = {}
        rewards_log = {f"seed_{s}": {} for s in seeds}
        best_video = None
        best_cand_idx = None
        best_final_score = -float("inf")
        cur_step = start_step
        
        for cp_idx, cp in enumerate(checkpoint_steps):
            end_step = cp["end_step"]
            is_final = (cp_idx == len(checkpoint_steps) - 1)
            is_early = (cp_idx == 0 and not is_final)
            phase_name = f"{'early' if is_early else 'mid' if not is_final else 'final'}_sigma{cp['target_sigma']:.2f}"
            frame_indices = early_frame_indices if is_early else normal_frame_indices
            
            # 1. 去噪到 checkpoint
            print(f"[TreeBranching] Denoising {len(alive)} candidates: "
                  f"step {cur_step} -> {end_step}")
            last_preds, pre_step_latents = self.wan.denoise_candidates(
                state, alive, cur_step, end_step
            )
            
            # 2. VAE 解码
            decoded_videos = {}
            for cand_idx in alive:
                if is_final:
                    latent_to_decode = state['candidates'][cand_idx]['latent']
                else:
                    latent_to_decode = self.wan.extract_pred_x0(
                        state, cand_idx,
                        last_preds[cand_idx],
                        pre_step_latents[cand_idx]
                    )
                video_tensor = self.wan.decode_latent(latent_to_decode)
                decoded_videos[cand_idx] = video_tensor.cpu()
                del latent_to_decode
                torch.cuda.empty_cache()
            
            # 3. 模型交替：卸载 DiT，加载 4RC
            if self.offload_models:
                self._offload_dit()
                self._offload_vae()
                self._load_4rc()
            
            # 4. 评分
            scored = []
            for cand_idx in alive:
                seed = seeds[cand_idx]
                frames_pil = wan_output_to_pil(decoded_videos[cand_idx])
                sampled = [frames_pil[i] for i in frame_indices
                          if i < len(frames_pil)]
                r = self.recon_reward.compute_reward(sampled)
                rewards_log[f"seed_{seed}"][phase_name] = r
                scored.append((cand_idx, r))
                
                print(f"  seed_{seed}: total={r['total']:.4f}")
                
                # 保存中间/最终视频
                if output_dir is not None:
                    self._save_video(
                        decoded_videos[cand_idx], seed, phase_name,
                        output_dir, save_fn)
                
                if is_final:
                    total = float(r.get('total', float('nan')))
                    selection_score = total if np.isfinite(total) else -float("inf")
                    if selection_score > best_final_score:
                        if best_video is not None:
                            del best_video
                        best_video = decoded_videos[cand_idx]
                        best_cand_idx = cand_idx
                        best_final_score = selection_score
                    else:
                        del decoded_videos[cand_idx]
                else:
                    del decoded_videos[cand_idx]
            
            # 5. 模型交替：卸载 4RC，重载 DiT
            if self.offload_models:
                self._offload_4rc()
                if not is_final:
                    self._load_dit()
                    self._load_vae()
            
            # 6. 淘汰
            if not is_final:
                alive, _ = self._eliminate(
                    scored, seeds, eliminated_at, phase_name, is_early
                )
            
            cur_step = end_step
        
        # --- 构造返回值（与父类一致）---
        if best_cand_idx is None:
            raise RuntimeError("No final candidate was decoded and scored.")
        
        best_seed = seeds[best_cand_idx]
        elapsed = time.time() - t_start
        result_log = self._build_result_log(
            seeds, rewards_log, eliminated_at, best_seed,
            sigma_checkpoints=self.sigma_checkpoints,
            elapsed=elapsed,
            checkpoint_steps=checkpoint_steps,
        )
        
        print(f"[TreeBranching] Best: seed_{best_seed} "
              f"(total={best_final_score:.4f}) in {elapsed:.1f}s")
        
        return best_video, result_log, best_seed
```

### 3.4 geo_reward/__init__.py — 导出新类

```python
from .da3_reward import DA3GeoReward, GeometryRewardConfig
from .recon_reward import ReconstructionReward, ReconRewardConfig
from .bon_pipeline import (
    GeoRewardBoN,
    GeoRewardBoNOffline,
    GeoRewardBoNProgressive,
    GeoRewardBoNProgressiveV2,
    GeoRewardBoNTreeBranching,      # ← 新增
)
from .guidance import GeometricGuidance
from .utils import wan_output_to_pil, wan_output_to_da3_input, sample_frames
```

### 3.5 run_bon_v2.py — CLI 参数扩展

```python
# 新增参数组
parser.add_argument('--tree_branching', action='store_true',
                    help='启用 Tree Branching 加速（默认关闭）')
parser.add_argument('--num_trunks', type=int, default=2,
                    help='主干轨迹数量（默认 2）')
parser.add_argument('--branches_per_trunk', type=int, default=4,
                    help='每条主干分叉数（默认 4，总候选 = num_trunks * branches_per_trunk）')
parser.add_argument('--branch_sigma', type=float, default=0.90,
                    help='分叉点的目标 σ 值（默认 0.90，动态映射到步数）')
parser.add_argument('--branch_eta', type=float, default=0.10,
                    help='分叉多样性超参数 η（默认 0.10）')
```

**分支逻辑**：

```python
if args.tree_branching:
    from geo_reward import GeoRewardBoNTreeBranching
    bon = GeoRewardBoNTreeBranching(
        wan_i2v=wan_i2v,
        recon_reward=recon_reward,
        num_trunks=args.num_trunks,
        branches_per_trunk=args.branches_per_trunk,
        branch_sigma=args.branch_sigma,
        branch_eta=args.branch_eta,
        sigma_checkpoints=args.sigma_checkpoints,
        elimination_ratio=args.elimination_ratio,
        min_survivors=args.min_survivors,
        score_epsilon=args.score_epsilon,
        early_max_frames=args.early_max_frames,
        offload_models=True,
    )
    N = args.num_trunks * args.branches_per_trunk
    # 返回值与其他路径一致：(best_video, result_log, best_seed)
    best_video, result_log, best_seed = bon.generate(
        prompt=args.prompt,
        image=img,
        N=N,
        frame_num=args.frame_num,
        seed_base=args.seed_base,
        output_dir=case_dir,
        save_fn=_save_fn,
        max_area=MAX_AREA_CONFIGS[args.size],
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
    )
elif args.no_progressive:
    # 原有顺序 BoN 路径 ...
else:
    # 原有渐进淘汰路径 ...
```

### 3.6 run_bon_batch_v2.py — 同样的 CLI 扩展

与 `run_bon_v2.py` 相同的参数组和分支逻辑。

---

## 4. 关键实现细节

### 4.1 Scheduler 深拷贝

分叉时 `copy.deepcopy(scheduler)` 需要正确复制 UniPC/DPM++ 的内部状态：

```python
# UniPC 内部状态：
scheduler.model_outputs     # List[Optional[Tensor]]，历史模型输出
scheduler.timestep_list     # List[int]，历史时间步
scheduler.lower_order_nums  # int，低阶步数计数
scheduler.last_sample       # Optional[Tensor]，上一步 latent
scheduler.sigmas            # Tensor，完整 sigma 序列（只读）
scheduler.step_index        # int，当前步位置

# DPM++ 内部状态：
scheduler.model_outputs     # List[Optional[Tensor]]
scheduler.sample            # Optional[Tensor]
scheduler.lower_order_nums  # int
```

`copy.deepcopy` 能正确处理这些张量和列表。

### 4.2 高噪声/低噪声模型切换

模型切换逻辑在 `_prepare_model_for_timestep()` 中：

```python
if t.item() >= boundary:    # boundary = 900 (= 0.900 * 1000)
    → high_noise_model
else:
    → low_noise_model
```

以 shift=5.0 为例：
- Step 0-14: t ≥ 900 → `high_noise_model`
- Step 15+: t < 900 → `low_noise_model`

当 `branch_sigma=0.90` 时，`find_step_for_sigma` 返回 16（即跑完 Step 0-15 后分叉）。此时：
- **主干阶段 (Step 0-15)**: Step 0-14 用 high_noise_model，Step 15 用 low_noise_model
- **分叉后 (Step 16+)**: 全程 `low_noise_model`

但注意：如果用户设置 `branch_sigma > 0.90`（如 0.93），`find_step_for_sigma` 返回 12，意味着只跑完 Step 0-11 后分叉。此时分叉后的 Step 12-14 仍然处于 high_noise_model 域（σ>0.90, t≥900）。这不是问题 —— `denoise_candidates()` 内部每步都调用 `_prepare_model_for_timestep()`，按 timestep 动态选择正确模型。但 `offload_model=True` 时会导致分叉后仍有模型切换开销。建议 `branch_sigma` 不要设太高（≤ 0.93）。

### 4.3 主干数量 K 的选择

**推荐 K=2**：
- K=1: 所有分支来自同一父本，多样性完全依赖 η，可能不够
- K=2: 两条独立噪声初始化的轨迹提供基础多样性，η 进一步扩展
- K=4: 共享节省的步数变少（高噪声阶段本来就相对快），收益递减

### 4.4 显存峰值分析

| 阶段 | GPU 内容 | 显存占用 |
|------|----------|----------|
| 主干去噪 (2 候选) | DiT (14B) + 2 latent | ~28GB + 微量 |
| 分叉后去噪 (8 候选) | DiT (14B) + 8 latent | ~28GB + 微量 |
| VAE 解码 | VAE + 1 latent→video | ~4GB |
| 4RC 评分 | 4RC model + frames | ~8GB |

Latent 大小：`16 × 21 × 60 × 104 × 2 bytes (bf16) ≈ 4MB`。8 候选 ≈ 32MB，可忽略。

### 4.5 与现有渐进淘汰的兼容性

Tree Branching 是渐进淘汰的**前置优化**，完全向后兼容：

```
[Tree Branching 独有]               [复用现有渐进淘汰]
Step 0 → 16: K=2 trunk       →     Step 16 → 22: 8 candidates
                                    Step 22: Eliminate → 4
                                    Step 22 → 31: 4 candidates
                                    Step 31: Eliminate → 2
                                    Step 31 → 40: 2 candidates
                                    Step 40: Select best
```

（步数以 shift=5.0, branch_sigma=0.90 为例）

### 4.6 `prepare_progressive()` 调用注意事项

真实签名为：
```python
def prepare_progressive(self, input_prompt, img, seeds, max_area=720*1280,
                         frame_num=81, shift=5.0, sample_solver='unipc',
                         sampling_steps=40, guide_scale=5.0, n_prompt="",
                         offload_model=True):
```

**必须使用关键字参数**调用，否则 `frame_num` 会被当成 `max_area`：

```python
# 正确 ✓
state = self.wan.prepare_progressive(
    input_prompt=prompt,
    img=image,
    seeds=trunk_seeds,
    frame_num=frame_num,
    **wan_kwargs,          # 包含 max_area, shift, sample_solver 等
)

# 错误 ✗ (frame_num 会被当成 max_area)
state = self.wan.prepare_progressive(prompt, image, trunk_seeds, frame_num, **wan_kwargs)
```

---

## 5. 实验验证计划

### 5.1 正确性验证（η=0 退化测试）

当 η=0 时，`z_branch = z_trunk`（恒等变换）。此时：
- 2 条主干各复制 4 份 → 4+4=8 个完全相同的候选对
- 同一主干的 4 个分支应产生完全相同的最终视频

```bash
python run_bon_v2.py --tree_branching --branch_eta 0.0 \
  --ckpt_dir ... --fourrc_model ... --image ... --prompt "..." --N 8
# 检查：来自同一 trunk 的 4 个视频应完全一致（像素级）
```

### 5.2 多样性验证（不同 η 值）

```bash
for eta in 0.05 0.10 0.15 0.20; do
    python run_bon_v2.py --tree_branching --branch_eta $eta \
      --ckpt_dir ... --fourrc_model ... --image ... --prompt "..." --N 8 \
      --output_dir results/eta_${eta}
done
# 比较：
# 1. 每个 η 下 8 个候选的分数方差（越大 → 多样性越好）
# 2. 最终 best 的分数 vs 标准渐进淘汰的 best 分数
# 3. 视觉检查：分叉后的视频是否保持合理的全局结构
```

### 5.3 加速比验证

```bash
# Tree Branching
time python run_bon_v2.py --tree_branching --N 8 ...

# 标准渐进淘汰（对照）
time python run_bon_v2.py --N 8 ...

# 期望：DiT 时间节省 ~31%，总 wall-clock 节省 ~20%（VAE/4RC 不变）
```

### 5.4 质量对比实验

```bash
# Tree Branching (推荐配置)
python run_bon_batch_v2.py --start 1 --end 24 --tree_branching \
  --branch_eta 0.10 --branch_sigma 0.90 --num_trunks 2 ...

# 标准渐进淘汰（对照）
python run_bon_batch_v2.py --start 1 --end 24 ...

# 对比指标：
# 1. 平均 best score
# 2. best score 的标准差
# 3. worst-case (最低 best score)
```

---

## 6. 默认参数总结

```python
# Tree Branching 新增参数
DEFAULT_NUM_TRUNKS = 2              # 主干数量
DEFAULT_BRANCHES_PER_TRUNK = 4      # 每主干分叉数（总候选 N=8）
DEFAULT_BRANCH_SIGMA = 0.90         # 分叉点 σ（动态映射步数）
DEFAULT_BRANCH_ETA = 0.10           # 多样性超参数

# 沿用的渐进淘汰参数
DEFAULT_SIGMA_CHECKPOINTS = [0.83, 0.63]
DEFAULT_ELIMINATION_RATIO = 0.5
DEFAULT_MIN_SURVIVORS = 2
DEFAULT_SCORE_EPSILON = 0.02
DEFAULT_EARLY_MAX_FRAMES = 12
```

---

## 7. 运行命令示例

```bash
# === Tree Branching + 渐进淘汰（推荐配置）===
python run_bon_v2.py \
  --tree_branching \
  --num_trunks 2 --branches_per_trunk 4 \
  --branch_sigma 0.90 --branch_eta 0.10 \
  --sigma_checkpoints 0.83 0.63 \
  --ckpt_dir /pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B \
  --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832 --sample_shift 5.0 --t5_cpu

# === 批量运行 ===
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

# === 多卡并行 ===
CUDA_VISIBLE_DEVICES=1 python run_bon_batch_v2.py --start 1 --end 24 \
  --tree_branching --branch_sigma 0.90 --branch_eta 0.10 ...

CUDA_VISIBLE_DEVICES=2 python run_bon_batch_v2.py --start 25 --end 48 \
  --tree_branching --branch_sigma 0.90 --branch_eta 0.10 ...
```

---

## 8. 风险和注意事项

### 8.1 η 过大导致质量退化

如果 η 设置过大（>0.20），分叉后的 latent 可能偏离训练分布，导致 artifacts。**缓解**：默认 η=0.10 保守起步。

### 8.2 多样性不足（所有分支收敛）

如果 η 过小或分叉点 σ 太低，分支可能收敛到几乎相同的视频。**缓解**：
- 默认 `branch_sigma=0.90`（σ 仍然很高）
- K=2 条独立主干提供基础多样性
- 监控候选间分数方差

### 8.3 Scheduler 内部状态不兼容

如果未来更新 solver，deepcopy 可能遗漏新增缓存。**缓解**：η=0 退化测试可捕获。

### 8.4 branch_sigma 与 sigma_checkpoints 冲突

必须确保 `branch_sigma > max(sigma_checkpoints)`，否则分叉步会落在第一个 checkpoint 之后，渐进淘汰无法正常工作。代码中应加校验：

```python
max_cp = max(self.sigma_checkpoints)
if self.branch_sigma <= max_cp:
    raise ValueError(
        f"branch_sigma({self.branch_sigma}) must be > "
        f"max(sigma_checkpoints)({max_cp})")
```

### 8.5 分叉后仍有 high_noise_model 步

如果 `branch_sigma > 0.90`，分叉后部分步仍需 high_noise_model。这不是 bug —— `_prepare_model_for_timestep()` 按 timestep 动态切换。但 `offload_model=True` 时会导致额外的模型搬运开销。建议 `branch_sigma` 不要设太高（≤ 0.93）。

---

## 9. 后续优化方向

### 9.1 自适应分叉数

根据主干在分叉点的 pred_x0 质量决定分叉策略：
- 两条主干质量相近 → 各分叉 4 条
- 一条明显优于另一条 → 优势主干分叉 6 条，劣势主干分叉 2 条

### 9.2 多级分叉

```
Step 0 → step_a:    1 条主干
step_a:             分叉为 2 条
step_a → step_b:   2 条独立去噪
step_b:             各分叉为 4 条 → 8 条
step_b →:           渐进淘汰
```

### 9.3 σ-adaptive η

```python
eta_adaptive = eta_base * sigma_t / 0.90  # 归一化到 σ=0.90 时的标准值
```

当 branch_sigma 不同时自动适配扰动幅度。
