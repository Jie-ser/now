# GeoReward 执行方案：DA3几何Reward + BoN + 梯度引导

## Part 1：DA3几何Reward设计与Best-of-N

---

### 1.1 目标

实现一个基于Depth Anything 3的几何一致性评分函数，对Wan2.2 I2V生成的多个候选视频进行评分排序，选出几何上最合理的视频。

### 1.2 DA3调用方式

DA3的高层API `inference()` 内部有 `@torch.inference_mode()` 装饰器，不支持梯度。Part 1不需要梯度，直接用高层API即可。

```python
from depth_anything_3 import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-LARGE").to("cuda")

# 输入：PIL Image列表 或 numpy ndarray列表 (H, W, 3) uint8
# 输出：Prediction dataclass
#   .depth: np.ndarray (N, H, W)
#   .conf: np.ndarray (N, H, W)
#   .extrinsics: np.ndarray (N, 4, 4)  world-to-camera
#   .intrinsics: np.ndarray (N, 3, 3)
prediction = model.inference(image_list, process_res=504)
```

### 1.3 Wan2.2输出到DA3输入的格式转换

```python
def wan_output_to_da3_input(video_tensor):
    """
    video_tensor: Wan2.2 输出, shape (3, T, H, W), range [-1, 1]
    返回: PIL Image列表, 供DA3 inference使用
    """
    from PIL import Image
    import numpy as np

    video = (video_tensor + 1) / 2  # [-1,1] -> [0,1]
    video = video.clamp(0, 1)
    video = video.permute(1, 0, 2, 3)  # (T, 3, H, W)
    video = (video * 255).byte().cpu().numpy()  # (T, 3, H, W) uint8
    video = video.transpose(0, 2, 3, 1)  # (T, H, W, 3)

    frames = [Image.fromarray(video[t]) for t in range(video.shape[0])]
    return frames
```

### 1.4 帧抽取策略

81帧全部送DA3开销大，抽取关键帧：

```python
def sample_frames(total_frames=81, max_frames=20):
    """均匀抽帧，首帧必选"""
    indices = [0]
    step = (total_frames - 1) / (max_frames - 1)
    for i in range(1, max_frames):
        indices.append(int(round(i * step)))
    return sorted(set(indices))
```

### 1.5 GeoReward函数：双向深度投射一致性

核心逻辑：帧t的像素通过depth反投影到3D，再通过帧s的相机投影回2D，投影深度应与帧s的实际深度一致。

```python
import numpy as np

class DA3GeoReward:
    def __init__(self, device="cuda"):
        self.model = DepthAnything3.from_pretrained(
            "depth-anything/DA3NESTED-LARGE"
        ).to(device)
        self.device = device

    def compute_reward(self, frames_pil, stride=4):
        """
        frames_pil: 抽取后的PIL Image列表（约20帧）
        stride: 投射检验的帧间隔
        返回: reward字典 {"total": float, "proj": float, "anchor": float, "conf": float}
        """
        pred = self.model.inference(frames_pil, process_res=504)
        depths = pred.depth          # (N, H, W) numpy
        extrinsics = pred.extrinsics # (N, 4, 4) numpy
        intrinsics = pred.intrinsics # (N, 3, 3) numpy
        conf = pred.conf             # (N, H, W) numpy

        r_proj = self._projection_consistency(depths, extrinsics, intrinsics, conf, stride)
        r_anchor = self._anchor_consistency(depths, extrinsics, intrinsics, conf)
        r_conf = self._confidence_score(conf)

        total = 0.50 * r_proj + 0.35 * r_anchor + 0.15 * r_conf
        return {"total": total, "proj": r_proj, "anchor": r_anchor, "conf": r_conf}

    def _projection_consistency(self, depths, extrinsics, intrinsics, conf, stride):
        """双向深度投射一致性"""
        N, H, W = depths.shape
        total_error = 0.0
        count = 0

        # 像素网格（只需构建一次）
        u, v = np.meshgrid(np.arange(W), np.arange(H))  # u:(H,W), v:(H,W)
        ones = np.ones((H, W))
        pixels = np.stack([u, v, ones], axis=-1)  # (H, W, 3)
        pixels_flat = pixels.reshape(-1, 3).T  # (3, H*W)

        for t in range(0, N - stride, stride):
            s = t + stride

            # 帧t: 像素->相机坐标3D点
            K_t_inv = np.linalg.inv(intrinsics[t])  # (3, 3)
            rays_t = K_t_inv @ pixels_flat  # (3, H*W)
            depth_t_flat = depths[t].reshape(-1)  # (H*W,)
            pts_cam_t = rays_t * depth_t_flat[None, :]  # (3, H*W)

            # 相机坐标->世界坐标
            # extrinsics是world-to-camera: P_cam = R @ P_world + t
            # 所以 P_world = R^T @ (P_cam - t)
            R_t = extrinsics[t, :3, :3]
            t_t = extrinsics[t, :3, 3]
            pts_world = R_t.T @ (pts_cam_t - t_t[:, None])  # (3, H*W)

            # 世界坐标->帧s相机坐标
            R_s = extrinsics[s, :3, :3]
            t_s = extrinsics[s, :3, 3]
            pts_cam_s = R_s @ pts_world + t_s[:, None]  # (3, H*W)

            # 投影到帧s像素坐标
            proj_s = intrinsics[s] @ pts_cam_s  # (3, H*W)
            px = proj_s[0] / (proj_s[2] + 1e-8)
            py = proj_s[1] / (proj_s[2] + 1e-8)
            depth_projected = pts_cam_s[2]  # (H*W,) 投影深度

            # 在帧s深度图上双线性采样
            depth_s_sampled = self._bilinear_sample(depths[s], px, py)
            conf_s_sampled = self._bilinear_sample(conf[s], px, py)

            # 有效像素
            valid = (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1) \
                    & (depth_projected > 1e-3) & (depth_s_sampled > 1e-3) \
                    & (conf_s_sampled > 0.3)

            if valid.sum() < 100:
                continue

            # 尺度对齐（中位数法）
            ratio = depth_projected[valid] / depth_s_sampled[valid]
            scale = np.median(ratio)
            aligned = depth_projected[valid] / scale

            # log-ratio误差
            log_err = np.abs(np.log(aligned / depth_s_sampled[valid] + 1e-8))
            weighted_err = (log_err * conf_s_sampled[valid]).sum() / (conf_s_sampled[valid].sum() + 1e-8)

            total_error += weighted_err
            count += 1

        if count == 0:
            return 0.0
        return -total_error / count  # 越大越好（误差越小）

    def _anchor_consistency(self, depths, extrinsics, intrinsics, conf):
        """首帧锚定一致性：静态区域的3D结构应与首帧一致"""
        N, H, W = depths.shape

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pixels_flat = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], axis=0)  # (3, H*W)

        # 首帧3D点云（世界坐标）
        K0_inv = np.linalg.inv(intrinsics[0])
        rays_0 = K0_inv @ pixels_flat
        pts_cam_0 = rays_0 * depths[0].reshape(-1)[None, :]
        R_0 = extrinsics[0, :3, :3]
        t_0 = extrinsics[0, :3, 3]
        pts_world_0 = R_0.T @ (pts_cam_0 - t_0[:, None])

        errors = []
        for t in range(1, N, max(1, N // 10)):  # 均匀采样约10帧
            R_t = extrinsics[t, :3, :3]
            t_t = extrinsics[t, :3, 3]
            pts_cam_t = R_t @ pts_world_0 + t_t[:, None]

            proj_t = intrinsics[t] @ pts_cam_t
            px = proj_t[0] / (proj_t[2] + 1e-8)
            py = proj_t[1] / (proj_t[2] + 1e-8)
            depth_proj = pts_cam_t[2]

            depth_actual = self._bilinear_sample(depths[t], px, py)

            valid = (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1) \
                    & (depth_proj > 1e-3) & (depth_actual > 1e-3)

            if valid.sum() < 100:
                continue

            ratio = depth_proj[valid] / depth_actual[valid]
            scale = np.median(ratio)
            deviation = np.abs(np.log(ratio / scale + 1e-8))

            # 取偏差最小的70%像素（认为是静态区域）
            sorted_dev = np.sort(deviation)
            n_static = int(len(sorted_dev) * 0.7)
            errors.append(sorted_dev[:n_static].mean())

        if len(errors) == 0:
            return 0.0
        return -np.mean(errors)

    def _confidence_score(self, conf):
        """DA3置信度均值"""
        return float(conf.mean())

    @staticmethod
    def _bilinear_sample(image, x, y):
        """numpy双线性插值采样"""
        H, W = image.shape
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        x0 = np.clip(x0, 0, W - 1)
        x1 = np.clip(x1, 0, W - 1)
        y0 = np.clip(y0, 0, H - 1)
        y1 = np.clip(y1, 0, H - 1)

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        return wa * image[y0, x0] + wb * image[y0, x1] + wc * image[y1, x0] + wd * image[y1, x1]
```

### 1.6 Best-of-N Pipeline

```python
import random
from wan import WanI2V

class GeoRewardBoN:
    def __init__(self, wan_model, da3_reward, frame_indices=None):
        self.wan = wan_model
        self.reward = da3_reward
        self.frame_indices = frame_indices or sample_frames(81, 20)

    def generate(self, prompt, image, N=8):
        """
        生成N个候选，用GeoReward选最优
        
        prompt: str, 动作指令
        image: PIL Image, 首帧
        N: 候选数量
        返回: (best_video, all_rewards)
        """
        candidates = []
        rewards = []

        for i in range(N):
            seed = random.randint(0, 2**32 - 1)
            video = self.wan.generate(
                input_prompt=prompt,
                img=image,
                seed=seed
            )
            # video: (3, 81, H, W) in [-1, 1]
            candidates.append(video)

            # 转换格式并抽帧
            frames_pil = wan_output_to_da3_input(video)
            sampled_frames = [frames_pil[i] for i in self.frame_indices]

            # 计算reward
            r = self.reward.compute_reward(sampled_frames, stride=2)
            rewards.append(r)
            print(f"  Candidate {i+1}/{N}: reward={r['total']:.4f} "
                  f"(proj={r['proj']:.4f}, anchor={r['anchor']:.4f}, conf={r['conf']:.4f})")

        # 选最优
        best_idx = max(range(N), key=lambda i: rewards[i]["total"])
        print(f"  Selected candidate {best_idx+1} with reward {rewards[best_idx]['total']:.4f}")

        return candidates[best_idx], rewards
```

### 1.7 Part 1 实验计划

**实验1：Reward有效性验证**

- 用Wan2.2对同一prompt生成32个视频
- 人工标注物理合理性评分（1-5分）
- 计算GeoReward与人工评分的Spearman相关系数
- 目标：相关系数 > 0.5

**实验2：BoN效果**

- N=1(baseline), 4, 8, 16
- 30个机械臂操作prompt
- 指标：人工物理合理性评分、FVD、CLIPScore

**实验3：消融实验**

- 仅R_proj / 仅R_anchor / 仅R_conf / 全部
- 不同stride: 2, 4, 8
- 不同抽帧数: 10, 20, 40

**实验4：DA3模型规模**

- DA3-Large (0.36B, ~78FPS)
- DA3-Giant (1.1B, ~38FPS)
- 精度vs速度权衡

### 1.8 Part 1 预计时间与产出

| 工作项 | 时间 | 产出 |
|--------|------|------|
| DA3环境搭建+推理验证 | 1天 | 确认DA3可正常运行 |
| GeoReward函数实现 | 2-3天 | da3_reward.py |
| Wan2.2集成+BoN pipeline | 1-2天 | bon_pipeline.py |
| Reward有效性实验 | 3-5天 | 相关性分析报告 |
| BoN实验+消融 | 5-7天 | 完整实验结果 |
| **Part 1 总计** | **约2-3周** | **验证GeoReward有效+BoN基线** |

---

## Part 2：梯度引导

### 2.1 前置条件

Part 2在Part 1验证GeoReward有效的基础上进行。如果Part 1的Reward与人工评分相关性太低（< 0.3），需要先改进Reward设计再进入Part 2。

### 2.2 数学基础

目标：从倾斜分布 p*(x) ∝ p(x) · exp(λ · R(x)) 中采样。

在去噪步t，修正后的score为：

```
∇_{x_t} log p_t*(x_t) = DiT output (已有) + λ · ∇_{x_t} R(x0_hat)
```

其中 x0_hat 是Tweedie估计的干净latent。关键点：**梯度不穿过DiT**，DiT输出视为常数。

梯度链：
```
latent_t  --(Tweedie)-->  x0_hat  --(VAE decode)-->  pixels  --(DA3)-->  reward
    ^                                                                       |
    |________ grad = d(reward)/d(x0_hat) * (1/alpha_t) _____________________|
```

### 2.3 DA3可微调用

DA3的高层API有`@torch.inference_mode()`，Part 2需要绕过它直接调用底层模型：

```python
class DA3Differentiable:
    """可微分的DA3调用，用于梯度引导"""

    def __init__(self, model_name="depth-anything/DA3NESTED-LARGE", device="cuda"):
        da3 = DepthAnything3.from_pretrained(model_name).to(device)
        # 取出底层网络（没有inference_mode装饰）
        self.net = da3.model  # DepthAnything3Net
        self.net.eval()
        # 冻结权重但允许梯度穿过
        for p in self.net.parameters():
            p.requires_grad = False
        self.device = device
        # ImageNet归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    def forward(self, frames_01):
        """
        可微的前向传播
        
        frames_01: (T, 3, H, W) tensor in [0, 1], requires_grad可为True
        返回: depth (T, H, W), extrinsics (T, 4, 4), intrinsics (T, 3, 3), conf (T, H, W)
              所有输出都在计算图中
        """
        # ImageNet归一化（可微）
        T, C, H, W = frames_01.shape
        frames_norm = (frames_01 - self.mean.squeeze(0)) / self.std.squeeze(0)
        # DA3期望输入: (B, N, 3, H, W)
        x = frames_norm.unsqueeze(0)  # (1, T, 3, H, W)

        # 直接调用底层网络（无no_grad）
        output = self.net(x)

        # 提取输出
        depth = output.depth.squeeze(0)       # (T, H, W)
        conf = output.conf.squeeze(0) if hasattr(output, 'conf') and output.conf is not None else None
        extrinsics = output.extrinsics.squeeze(0) if hasattr(output, 'extrinsics') else None
        intrinsics = output.intrinsics.squeeze(0) if hasattr(output, 'intrinsics') else None

        return depth, extrinsics, intrinsics, conf
```

### 2.4 可微的投射一致性Reward

Part 1的Reward用numpy不可微。Part 2需要PyTorch全程可微版本：

```python
class DA3GeoRewardDifferentiable:
    """全程可微的几何Reward，用于梯度引导"""

    def __init__(self, device="cuda"):
        self.da3 = DA3Differentiable(device=device)
        self.device = device

    def compute_reward(self, frames_01, stride=4):
        """
        可微的Reward计算
        
        frames_01: (T, 3, H, W) tensor in [0, 1], 来自VAE decode
                   需要requires_grad=True（通过计算图连接到latent）
        stride: 帧间隔
        返回: 标量reward tensor（有grad_fn）
        """
        depth, extrinsics, intrinsics, conf = self.da3.forward(frames_01)

        T, H, W = depth.shape
        total_error = torch.tensor(0.0, device=self.device)
        count = 0

        # 像素网格
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        pixels_flat = torch.stack([
            u_coords.reshape(-1),
            v_coords.reshape(-1),
            torch.ones(H * W, device=self.device)
        ], dim=0)  # (3, H*W)

        for t_idx in range(0, T - stride, stride):
            s_idx = t_idx + stride

            # 帧t: 像素->3D
            K_t_inv = torch.inverse(intrinsics[t_idx, :3, :3])
            rays = K_t_inv @ pixels_flat  # (3, H*W)
            pts_cam_t = rays * depth[t_idx].reshape(1, -1)  # (3, H*W)

            # 相机坐标->世界->帧s相机坐标
            R_t = extrinsics[t_idx, :3, :3]
            t_t = extrinsics[t_idx, :3, 3]
            R_s = extrinsics[s_idx, :3, :3]
            t_s = extrinsics[s_idx, :3, 3]

            pts_world = R_t.T @ (pts_cam_t - t_t.unsqueeze(1))
            pts_cam_s = R_s @ pts_world + t_s.unsqueeze(1)

            # 投影到帧s
            proj_s = intrinsics[s_idx, :3, :3] @ pts_cam_s
            px = proj_s[0] / (proj_s[2] + 1e-6)
            py = proj_s[1] / (proj_s[2] + 1e-6)
            depth_projected = pts_cam_s[2]

            # 可微的grid_sample
            grid_x = (px / W) * 2 - 1
            grid_y = (py / H) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, -1, 2)

            depth_sampled = torch.nn.functional.grid_sample(
                depth[s_idx].unsqueeze(0).unsqueeze(0),
                grid, mode='bilinear', align_corners=True, padding_mode='zeros'
            ).squeeze()  # (H*W,)

            # 有效区域mask（不可微的选择，但不影响梯度流）
            with torch.no_grad():
                valid = (px >= 0) & (px < W) & (py >= 0) & (py < H) \
                        & (depth_projected > 1e-3) & (depth_sampled > 1e-3)

            if valid.sum() < 100:
                continue

            # 尺度对齐（detach中位数，避免梯度穿过排序）
            with torch.no_grad():
                ratio_detached = depth_projected[valid] / depth_sampled[valid]
                scale = ratio_detached.median()

            aligned = depth_projected[valid] / scale
            # L1 误差（可微）
            error = (aligned - depth_sampled[valid]).abs().mean()

            total_error = total_error + error
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 返回负误差作为reward（越大越好）
        return -(total_error / count)
```

### 2.5 Wan2.2 Tweedie估计

Wan2.2使用Flow Matching，Tweedie估计的形式与DDPM不同。Flow Matching的ODE是：

```
dx_t = v_theta(x_t, t) dt
```

其中 x_t = (1-t) * x_0 + t * epsilon（线性插值，t从0到1，t=0是clean，t=1是纯噪声）。

给定当前 x_t 和模型预测的velocity v_theta，干净样本估计为：

```
x0_hat = x_t - t * v_theta(x_t, t)
```

注意Wan2.2的scheduler可能使用不同的参数化（shift等）。需要查看scheduler的具体实现来确定正确的Tweedie公式。以下给出通用版本：

```python
def tweedie_estimate(latent_t, velocity_pred, t, scheduler):
    """
    从当前noisy latent和velocity预测估计干净latent
    
    latent_t: 当前含噪latent
    velocity_pred: DiT预测的velocity
    t: 当前时间步（Wan2.2的时间步可能是0-1000的整数）
    scheduler: Wan2.2的scheduler，用于获取信噪比参数
    
    返回: x0_hat，估计的干净latent
    """
    # 对Flow Matching: x_t = (1-sigma_t) * x_0 + sigma_t * noise
    # velocity = x_0 - noise = (x_t - sigma_t * noise) / (1 - sigma_t) - noise
    # 简化: x_0 = x_t + (1 - sigma_t / (1-sigma_t)) * velocity
    
    # 实际实现需要参考Wan2.2 scheduler的具体参数化
    # FlowUniPC的step实现中可以找到x0_pred的计算方式
    
    # 通用Flow Matching:
    # x_0 = x_t - t * velocity （当使用标准线性插值参数化时）
    sigma_t = t / scheduler.num_train_timesteps  # 归一化到[0,1]
    x0_hat = (latent_t - sigma_t * velocity_pred) / (1 - sigma_t + 1e-8)
    
    return x0_hat
```

**重要**：上面的Tweedie公式需要根据Wan2.2 scheduler的具体实现来校准。建议在实现时打印几个中间步的x0_hat，VAE decode后检查是否是合理的图像。如果不合理，说明Tweedie公式的参数化有误，需要调整。

### 2.6 梯度引导的去噪循环

```python
class GeoGuidedGeneration:
    """带DA3梯度引导的Wan2.2 I2V生成"""

    def __init__(self, wan_model, geo_reward_diff, vae, 
                 guidance_scale=0.005,
                 guidance_start=0.5,
                 guidance_interval=5,
                 num_guidance_frames=8):
        """
        wan_model: Wan2.2 DiT模型
        geo_reward_diff: DA3GeoRewardDifferentiable实例
        vae: Wan2.2 VAE
        guidance_scale: 梯度引导强度（关键超参，需要调）
        guidance_start: 从去噪进度多少开始引导（0.5=后半段）
        guidance_interval: 每隔多少步做一次引导
        num_guidance_frames: 引导时抽取多少帧计算reward
        """
        self.wan_model = wan_model
        self.geo_reward = geo_reward_diff
        self.vae = vae
        self.w_s = guidance_scale
        self.guidance_start = guidance_start
        self.guidance_interval = guidance_interval
        self.num_frames = num_guidance_frames

    def generate(self, prompt, image, sampling_steps=40, **kwargs):
        """
        带梯度引导的生成
        """
        # === 标准Wan2.2初始化（同image2video.py）===
        noise, y, context, context_null, scheduler, timesteps, cfg_scale = \
            self._init_generation(prompt, image, sampling_steps, **kwargs)
        latent = noise  # (16, T_lat, H', W')

        total_steps = len(timesteps)

        for step_idx, t in enumerate(timesteps):
            progress = step_idx / total_steps  # 0->1, 从高噪到低噪

            # === Step 1: 标准DiT forward（无梯度）===
            with torch.no_grad():
                velocity_cond = self.wan_model(
                    [latent], t=[t], context=[context], y=[y])[0]
                velocity_uncond = self.wan_model(
                    [latent], t=[t], context=[context_null], y=[y])[0]
                velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)

            # === Step 2: 决定是否执行梯度引导 ===
            do_guidance = (
                progress >= self.guidance_start
                and step_idx % self.guidance_interval == 0
            )

            if do_guidance:
                # === Step 3: Tweedie估计干净latent ===
                x0_hat = tweedie_estimate(latent, velocity, t, scheduler)
                x0_hat = x0_hat.detach().requires_grad_(True)

                # === Step 4: VAE decode -> pixel frames ===
                pixel_video = self.vae.decode([x0_hat])  # (3, T_pixel, H, W) in [-1, 1]

                # === Step 5: 转换格式 + 抽帧 ===
                frames_01 = (pixel_video + 1) / 2  # -> [0, 1]
                frames_01 = frames_01.permute(1, 0, 2, 3)  # (T, 3, H, W)
                T_total = frames_01.shape[0]
                indices = torch.linspace(0, T_total - 1, self.num_frames).long()
                frames_sampled = frames_01[indices]  # (num_frames, 3, H, W)
                # Resize到DA3处理分辨率
                frames_resized = torch.nn.functional.interpolate(
                    frames_sampled, size=(504, 504), mode='bilinear', align_corners=False
                )

                # === Step 6: DA3可微reward ===
                reward = self.geo_reward.compute_reward(frames_resized, stride=2)

                # === Step 7: 计算梯度 ===
                grad_x0 = torch.autograd.grad(reward, x0_hat)[0]

                # === Step 8: 修正velocity ===
                velocity = velocity - self.w_s * grad_x0

                # 清理
                del x0_hat, pixel_video, frames_01, frames_sampled, frames_resized
                del reward, grad_x0
                torch.cuda.empty_cache()

            # === Step 9: Scheduler step ===
            latent = scheduler.step(
                velocity.unsqueeze(0), t, latent.unsqueeze(0)
            )[0].squeeze(0)

        # === 最终decode ===
        with torch.no_grad():
            video = self.vae.decode([latent])

        return video
```

### 2.7 梯度引导 + BoN 组合

WMReward论文的核心结论：梯度引导+BoN组合 >> 任何一个单独使用。

```python
class GeoGuidedBoN:
    """梯度引导 + Best-of-N 组合"""

    def __init__(self, guided_generator, da3_reward_numpy):
        self.guided_gen = guided_generator    # GeoGuidedGeneration (梯度引导)
        self.reward_eval = da3_reward_numpy   # DA3GeoReward (numpy版，最终评分)

    def generate(self, prompt, image, N=4):
        """
        先用梯度引导生成N个候选，再用完整GeoReward选最优
        
        N不需要很大（4-8即可），因为每个候选本身已经被引导过
        """
        candidates = []
        rewards = []

        for i in range(N):
            # 梯度引导生成（每次不同seed）
            video = self.guided_gen.generate(prompt, image, seed=random.randint(0, 2**32))
            candidates.append(video)

            # 用Part 1的numpy版reward做最终评分（更快、更稳定）
            frames_pil = wan_output_to_da3_input(video)
            sampled = [frames_pil[j] for j in sample_frames(81, 20)]
            r = self.reward_eval.compute_reward(sampled, stride=2)
            rewards.append(r)

        best_idx = max(range(N), key=lambda i: rewards[i]["total"])
        return candidates[best_idx], rewards
```

### 2.8 关键超参与调优策略

| 超参 | 建议范围 | 调优方法 |
|------|---------|---------|
| guidance_scale (w_s) | 0.001 - 0.01 | 从0.005开始；太大导致伪影，太小无效果 |
| guidance_start | 0.4 - 0.7 | 0.5是安全起点；越早reward越不可靠 |
| guidance_interval | 3 - 10 | 5是平衡点；太频繁则慢，太稀疏则弱 |
| num_guidance_frames | 4 - 12 | 8帧；越多越准但显存越大 |
| DA3 process_res | 336 - 504 | 引导时可用336（快），BoN评分用504（准） |
| BoN N（组合时） | 4 - 8 | 有引导时不需要大N |

**调优流程**：
1. 先固定w_s=0.005，跑几个样本看有无明显效果
2. 如果无效果，逐步加大到0.01、0.02
3. 如果出现伪影/色彩异常，减小w_s或增大guidance_start
4. 验证Tweedie估计：VAE decode后的帧是否是合理图像（否则公式有误）

### 2.9 显存管理

**估算（A100 80GB）**：

| 组件 | 显存 |
|------|------|
| Wan2.2 DiT 14B (fp16 inference) | ~28GB |
| Wan2.2 VAE (decode, 有梯度) | ~8GB |
| DA3-Large (0.36B, 有梯度穿过) | ~5GB |
| 中间激活(VAE decode + DA3) | ~10GB |
| latent + 其他 | ~5GB |
| **总计** | **~56GB** |

如果超过80GB：
- 减少num_guidance_frames（8->4）
- 降低DA3 process_res（504->336）
- 使用gradient checkpointing（对VAE decode）
- 用DA3-Base (0.11B) 替代 DA3-Large

### 2.10 Part 2 实验计划

**实验5：梯度引导有效性**

- Guided (w_s=0.005) vs Unguided baseline
- 30个prompt，评估GeoReward改善幅度
- 同时检查CLIPScore是否下降（语义忠实度）

**实验6：超参搜索**

- w_s: [0.001, 0.003, 0.005, 0.01, 0.02]
- guidance_start: [0.3, 0.5, 0.7]
- 3x5=15组，每组10个样本

**实验7：组合效果**

- BoN-8 (Part 1)
- Guided-only (Part 2, N=1)
- Guided + BoN-4 (组合)
- Guided + BoN-8 (组合)
- 验证组合 > 任何单独使用

**实验8：与WMReward对比**

- GeoReward Guided vs WMReward Guided
- GeoReward + WMReward 双Reward引导
- velocity_guided = velocity - w_geo * grad_geo - w_wm * grad_wm

### 2.11 Part 2 预计时间与产出

| 工作项 | 时间 | 产出 |
|--------|------|------|
| DA3可微调用实现+验证 | 2-3天 | da3_differentiable.py |
| 可微Reward函数实现 | 2-3天 | geo_reward_differentiable.py |
| Tweedie估计实现+验证 | 2-3天 | 确认x0_hat质量合理 |
| 梯度引导去噪循环实现 | 3-5天 | guided_generation.py |
| 显存优化+调试 | 3-5天 | 能在A100上跑通 |
| 超参调优实验 | 5-7天 | 最佳超参配置 |
| 组合实验+对比 | 5-7天 | 完整实验结果 |
| **Part 2 总计** | **约4-5周** | **梯度引导+组合方案** |

---

## 实施总时间线

```
Week 1-2:   Part 1 代码实现 (DA3 Reward + BoN pipeline)
Week 2-3:   Part 1 实验 (Reward验证 + BoN效果)
            -> Go/No-Go 判断：Reward与人工评分相关性 > 0.3?
Week 4-5:   Part 2 代码实现 (可微DA3 + 梯度引导循环)
Week 5-6:   Part 2 调试 (Tweedie验证 + 显存优化)
Week 6-8:   Part 2 实验 (超参搜索 + 组合实验 + 对比)
```

---

## 关键风险与检查点

| 检查点 | 时间 | 判据 | 失败应对 |
|--------|------|------|---------|
| DA3能否对生成视频产出合理深度 | Week 1 | 深度图视觉上合理 | 换VGGTomega |
| GeoReward与人工评分相关 | Week 2-3 | Spearman > 0.3 | 改Reward设计 |
| Tweedie估计是否合理 | Week 4 | decode后是合理图像 | 调公式参数化 |
| 梯度引导是否有效果 | Week 5 | Reward可见提升 | 增大w_s或换引导时机 |
| 组合是否优于单独 | Week 7 | Guided+BoN > 两者单独 | 检查是否over-optimize |
