GeoReward V2（修订版）：基于 4RC 显式几何一致性的 Reward 设计方案

1. 核心动机与问题定义
1.1 背景
GEM-4D 论文证明：含有精准几何信息的操作类视频能够显著提升下游机械臂任务成功率（+20%）。其关键洞见是——Geometry Foundation Model 提取的几何特征能够编码完整的帧间对应关系（inter-frame correspondences），而帧间对应关系正是物理一致性的核心。
核心推论：如果一段生成视频能够被 4D 模型高质量重建——即其 3D 结构跨帧一致、运动轨迹连贯可追踪——说明其几何信息准确、物理合理，可以直接服务于机械臂操作。
1.2 问题形式化
给定：

首帧图像 I0I_0
I0​（机械臂场景）
语义动作指令 pp
p（如"抓取红色方块"）
I2V 模型 GG
G（Wan2.2-I2V-A14B）

目标：从 NN
N 个候选视频 {v1,...,vN}=G(I0,p,seeds)\{v_1, ..., v_N\} = G(I_0, p, \text{seeds})
{v1​,...,vN​}=G(I0​,p,seeds) 中，选出几何一致性最高的视频 v∗v^*
v∗。
Reward 定义：
R(v)=GeometricConsistency4D(v)R(v) = \text{GeometricConsistency}_{4D}(v)R(v)=GeometricConsistency4D​(v)
高分意味着：该视频的静态背景跨帧深度一致、动态物体 3D 轨迹连贯平滑、相机运动合理。
1.3 与 V1 的区别
方面V1（当前）V2 修订版（本方案）评价目标手工投影一致性 + 运动平滑度4RC 显式 3D 输出的几何验证几何来源DA3（3D，静态假设）4RC（4D，原生动态支持）Reward 性质手工公式 exp(-E/τ)基于 4RC 输出的显式几何一致性指标动态分割帧差 + 深度变化率（手工）4RC track 位移后验推断conf 的角色无仅作为 valid mask（不作为分数）梯度引导未实现对显式几何 loss 求梯度

2. 模型选择：4RC 作为唯一几何模型
2.1 选择 4RC 的核心理由

一次推理，完整输出：4RC 同时输出 pts（3D 点）、track（密集 3D 位移）、extrinsics（相机位姿）、conf/conf_track。从 track 中可以后验推断 static/dynamic 区域，无需外部预分割。
架构设计天然分离相机运动与物体运动：4RC 的分解表示 P(i→τ) = P(i) + ΔP(i→τ) 中，相机运动已被 extrinsics 吸收。世界坐标系中 ||ΔP|| ≈ 0 的像素是静态的，||ΔP|| > 0 的是动态的。
原生动态场景支持：在 PointOdyssey、Kubric、Waymo 等动态数据集上联合训练，理解物体运动。DA3 假设静态场景，对机械臂操作视频根本不适用。
可微性：Arc.forward() 无 @torch.no_grad 装饰器（经代码验证）。梯度引导可直接调用 model(views) 传递梯度。
DA3 的超集：基于 DA3 ViT-Giant 权重初始化，继承深度估计能力，额外增加运动解码。

2.2 DA3 的角色
DA3 不参与主流程，保留为：

Phase 0 中的 baseline 对比（V1 reward vs V2 新 reward）
消融实验中的 independent evaluator
交叉验证信号源

2.3 为什么不用 DA3 + 4RC 双模型
4RC 的输入是完整视频帧序列，它无法"只看静态部分"。同样 DA3 也接收同样的帧。如果要"DA3 负责静态，4RC 负责动态"，就需要先知道 static/dynamic mask——这本身就依赖某种分析。而 4RC 一次推理就直接给出了这个区分（从 track 位移推断），无需额外步骤。双模型方案增加了复杂度但没有增加信息。

3. Reward 设计
3.1 总体架构
输入：视频帧 [F₀, F₁, ..., F_T]
        ↓
    均匀抽帧 → 12~20 帧
        ↓
    4RC forward（一次推理）
        ↓
    输出: pts(N,H,W,3), track(N,H,W,3), conf(N,H,W),
          conf_track(N,H,W), extrinsics(N,4,4), intrinsics(N,3,3)
        ↓
    从 track 推断 dynamic_mask（逐帧最大位移法）
        ↓
    conf/conf_track → valid_mask（只决定哪些像素参与计算）
        ↓
    ┌─────────── R_static ───────────┐
    │ 静态区域：跨帧深度重投影误差     │
    │ + forward-backward cycle       │
    │ + 首帧深度锚定                  │
    └────────────────────────────────┘
    ┌─────────── R_dynamic ──────────┐
    │ 动态区域：轨迹加速度 penalty     │
    │ + coverage（有效追踪比例）       │
    │ + 轨迹平滑度                    │
    └────────────────────────────────┘
    ┌─────────── R_motion ───────────┐
    │ 相机加速度 penalty              │
    │ + 动态区域极端速度 penalty       │
    │ + motion gate（确保有运动）      │
    └────────────────────────────────┘
        ↓
    R_total = G_anchor × (0.40×R_static + 0.40×R_dynamic + 0.20×R_motion)
3.2 conf/conf_track 的角色：仅作为 Valid Mask
conf 和 conf_track 不参与分数计算。 它们的唯一作用是标记"哪些像素的 4RC 输出可信"：
pythondef compute_valid_mask(conf, conf_track, quantile=0.20):
    """
    conf/conf_track 仅用于筛选有效像素，不作为 reward 分数。
    理由：conf 是 aleatoric uncertainty head 的输出，是模型对自身预测的
    不确定性估计，不等于"几何真的准"。最大化 conf 可能被梯度攻击。
    """
    # 取视频内低分位数作为阈值
    conf_threshold = torch.quantile(conf.flatten(), quantile)
    track_threshold = torch.quantile(conf_track.flatten(), quantile)
    
    valid_geo = (conf > conf_threshold)
    valid_track = (conf_track > track_threshold)
    
    return valid_geo, valid_track
3.3 Dynamic Mask：逐帧最大位移法
不能只用首末帧位移（norm(track[-1] - track[0])），因为会遗漏"中途运动后回到原位"的情况，也可能把轨迹预测误差当成真实运动。
pythondef compute_dynamic_mask(track, threshold=0.01):
    """
    track: (N, H, W, 3) — 每帧每像素相对 query frame 的世界坐标位移
    
    使用逐帧最大位移，而非首末帧位移：
    - 捕捉"中途运动后回到原位"的情况
    - 对所有帧的运动信息做完整利用
    """
    # 计算每像素在所有帧中的最大位移（相对 query frame）
    displacement_per_frame = torch.norm(track, dim=-1)  # (N, H, W)
    max_displacement = displacement_per_frame.max(dim=0).values  # (H, W)
    
    # 阈值判定
    dynamic_mask = (max_displacement > threshold)  # (H, W)
    static_mask = ~dynamic_mask
    
    return static_mask, dynamic_mask
3.4 R_static — 静态区域几何一致性
在静态区域上，同一个世界点在不同帧的投影深度应该一致。
pythondef compute_R_static(pts, extrinsics, intrinsics, static_mask, valid_geo, conf):
    """
    pts: (N, H, W, 3) — 世界坐标 3D 点
    extrinsics: (N, 4, 4) — camera-to-world
    intrinsics: (N, 3, 3)
    static_mask: (H, W) — 静态区域
    valid_geo: (N, H, W) — conf 派生的有效 mask
    """
    N, H, W, _ = pts.shape
    errors_reproj = []
    errors_cycle = []
    valid_counts = []
    
    # 选取帧对（不只相邻，也包含间隔帧以检测长程一致性）
    frame_pairs = get_frame_pairs(N, stride=[1, 3, 5])
    
    for i, j in frame_pairs:
        # 只取静态 + 有效的像素
        mask = static_mask & valid_geo[i] & valid_geo[j]
        if mask.sum() < 100:
            continue
        
        # frame_i 的世界点
        pts_i = pts[i][mask]  # (K, 3)
        
        # 投影到 frame_j 的相机坐标系
        w2c_j = torch.inverse(extrinsics[j])  # (4, 4) world-to-cam
        pts_i_homo = F.pad(pts_i, (0, 1), value=1.0)  # (K, 4)
        pts_i_in_cam_j = (w2c_j @ pts_i_homo.T).T[:, :3]  # (K, 3)
        
        # 投影到 frame_j 的像素坐标
        proj_uv = (intrinsics[j] @ pts_i_in_cam_j.T).T  # (K, 3)
        proj_u = proj_uv[:, 0] / (proj_uv[:, 2] + 1e-8)
        proj_v = proj_uv[:, 1] / (proj_uv[:, 2] + 1e-8)
        proj_depth = pts_i_in_cam_j[:, 2]
        
        # 边界检查
        in_bounds = (proj_u >= 0) & (proj_u < W-1) & (proj_v >= 0) & (proj_v < H-1)
        # 正深度检查
        positive_depth = proj_depth > 0.01
        valid = in_bounds & positive_depth
        
        if valid.sum() < 50:
            continue
        
        # Bilinear sampling 获取 frame_j 在投影位置的实际深度
        # 将 pts[j] 的 z 通道作为深度图
        depth_map_j = pts[j][..., 2]  # (H, W) — 在相机坐标系中的深度
        # 注意：pts 是世界坐标，需要转换到 cam_j 坐标系获取深度
        pts_j_cam = transform_to_camera(pts[j], w2c_j)  # (H, W, 3)
        depth_map_j_cam = pts_j_cam[..., 2]  # (H, W)
        
        # Grid sample（bilinear interpolation）
        grid = torch.stack([
            2.0 * proj_u[valid] / (W - 1) - 1.0,
            2.0 * proj_v[valid] / (H - 1) - 1.0
        ], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, K_valid, 2)
        
        sampled_depth = F.grid_sample(
            depth_map_j_cam.unsqueeze(0).unsqueeze(0),
            grid, mode='bilinear', align_corners=True
        ).squeeze()  # (K_valid,)
        
        # 遮挡过滤：投影深度不应远大于实际深度（说明被遮挡）
        occlusion_margin = 1.05
        not_occluded = proj_depth[valid] < sampled_depth * occlusion_margin
        
        # Scale-aligned log-depth error
        if not_occluded.sum() > 20:
            log_error = torch.abs(
                torch.log(proj_depth[valid][not_occluded] / 
                         (sampled_depth[not_occluded] + 1e-8))
            )
            errors_reproj.append(log_error.median())
            valid_counts.append(not_occluded.sum().float() / mask.sum().float())
    
    if not errors_reproj:
        return torch.tensor(0.5)  # 无法计算时返回中性分数
    
    # 聚合
    E_reproj = torch.stack(errors_reproj).mean()
    V_ratio = torch.stack(valid_counts).mean()  # 有效重投影比例
    
    R_static = (torch.exp(-E_reproj / tau_reproj) * 
                torch.sigmoid((V_ratio - 0.3) / 0.1))
    
    return R_static
3.5 R_dynamic — 动态区域追踪质量
在动态区域上，评价 3D 轨迹的连贯性和物理合理性。
pythondef compute_R_dynamic(track, dynamic_mask, valid_track, scene_scale):
    """
    track: (N, H, W, 3) — 每帧每像素相对 query frame 的 3D 位移
    dynamic_mask: (H, W) — 动态区域
    valid_track: (N, H, W) — conf_track 派生的有效 mask
    scene_scale: float — 场景尺度归一化因子
    """
    N, H, W, _ = track.shape
    
    # 只取动态 + 有效的像素
    # 对每帧取交集
    combined_mask = dynamic_mask.unsqueeze(0) & valid_track  # (N, H, W)
    
    # ========== 1. Coverage ==========
    # 有效追踪像素占动态区域的比例
    dynamic_total = dynamic_mask.sum().float()
    if dynamic_total < 50:
        return torch.tensor(0.5)  # 动态区域太小，返回中性分数
    
    per_frame_coverage = combined_mask.float().sum(dim=(-1,-2)) / dynamic_total  # (N,)
    coverage = per_frame_coverage.mean()
    
    # ========== 2. 加速度 Penalty ==========
    # 3D 轨迹的二阶差分：||x_{t+1} - 2x_t + x_{t-1}||
    # track 是相对 query frame 的绝对位移，所以 track[t] 就是 t 时刻的位置
    if N >= 3:
        # 取动态区域像素的轨迹
        # 为稳定性，取动态区域中置信度最高的 top-K 像素
        K = min(1000, combined_mask[0].sum().item())
        if K < 10:
            return coverage  # 太少像素，只返回 coverage
        
        # 选取在所有帧都有效的像素
        all_valid = combined_mask.all(dim=0) & dynamic_mask  # (H, W)
        valid_indices = all_valid.nonzero()  # (M, 2)
        
        if valid_indices.shape[0] < 10:
            return coverage
        
        # 随机采样 K 个像素
        K = min(K, valid_indices.shape[0])
        selected = valid_indices[torch.randperm(valid_indices.shape[0])[:K]]
        
        # 提取这些像素的 3D 轨迹
        trajectories = track[:, selected[:, 0], selected[:, 1], :]  # (N, K, 3)
        
        # 二阶差分（加速度）
        accel = trajectories[2:] - 2 * trajectories[1:-1] + trajectories[:-1]  # (N-2, K, 3)
        accel_magnitude = torch.norm(accel, dim=-1) / scene_scale  # (N-2, K)
        
        # 取中位数作为鲁棒指标
        E_accel = accel_magnitude.median()
        
        # ========== 3. 速度平滑度 ==========
        # 帧间速度变化的一致性
        velocity = trajectories[1:] - trajectories[:-1]  # (N-1, K, 3)
        speed = torch.norm(velocity, dim=-1)  # (N-1, K)
        
        # p95/median ratio：惩罚极端速度（瞬移）
        speed_per_pixel = speed.mean(dim=0)  # (K,) 每个像素的平均速度
        if speed_per_pixel.numel() > 0:
            p95 = torch.quantile(speed.flatten(), 0.95)
            median_speed = speed.flatten().median()
            speed_excess = (p95 / (median_speed + 1e-8)) - 1.0
            E_speed = torch.clamp(speed_excess, min=0.0)
        else:
            E_speed = torch.tensor(0.0)
    else:
        E_accel = torch.tensor(0.0)
        E_speed = torch.tensor(0.0)
    
    # ========== 组合 ==========
    R_dynamic = (coverage ** 0.5 * 
                 torch.exp(-E_accel / tau_accel) * 
                 torch.exp(-E_speed / tau_speed))
    
    return R_dynamic
3.6 R_motion — 相机与整体运动评价
pythondef compute_R_motion(extrinsics, track, dynamic_mask, scene_scale):
    """
    extrinsics: (N, 4, 4) — camera-to-world
    track: (N, H, W, 3)
    dynamic_mask: (H, W)
    """
    N = extrinsics.shape[0]
    
    # ========== 1. 相机加速度 penalty ==========
    # 提取相机位置（c2w 的平移部分）
    cam_positions = extrinsics[:, :3, 3]  # (N, 3)
    if N >= 3:
        cam_accel = cam_positions[2:] - 2*cam_positions[1:-1] + cam_positions[:-1]
        E_cam_accel = torch.norm(cam_accel, dim=-1).median() / scene_scale
    else:
        E_cam_accel = torch.tensor(0.0)
    
    # 提取相机旋转变化
    cam_rotations = extrinsics[:, :3, :3]  # (N, 3, 3)
    if N >= 3:
        # 相邻帧的相对旋转角度
        rot_diffs = []
        for t in range(N-1):
            R_rel = cam_rotations[t+1] @ cam_rotations[t].T
            # 旋转角度 = arccos((trace(R)-1)/2)
            angle = torch.acos(
                torch.clamp((R_rel.trace() - 1) / 2, -1.0, 1.0)
            )
            rot_diffs.append(angle)
        rot_diffs = torch.stack(rot_diffs)
        # 旋转加速度
        if len(rot_diffs) >= 2:
            rot_accel = torch.abs(rot_diffs[1:] - rot_diffs[:-1])
            E_rot_accel = rot_accel.median()
        else:
            E_rot_accel = torch.tensor(0.0)
    else:
        E_rot_accel = torch.tensor(0.0)
    
    # ========== 2. Motion Gate ==========
    # 确保动态区域有足够运动量（避免"完全静止"的视频拿高分）
    displacement_per_frame = torch.norm(track, dim=-1)  # (N, H, W)
    max_displacement = displacement_per_frame.max(dim=0).values  # (H, W)
    # 动态区域的运动量
    if dynamic_mask.sum() > 0:
        dynamic_motion = max_displacement[dynamic_mask].median()
    else:
        dynamic_motion = max_displacement.quantile(0.90)  # fallback
    
    gate = torch.sigmoid((dynamic_motion - min_motion) / tau_motion)
    
    # ========== 3. 动态区域极端速度 penalty ==========
    # 如果有像素瞬间移动了场景尺度的 50% 以上，惩罚
    if dynamic_mask.sum() > 0:
        frame_displacements = torch.norm(
            track[1:] - track[:-1], dim=-1
        )  # (N-1, H, W) 帧间增量
        dynamic_frame_disp = frame_displacements[:, dynamic_mask]
        if dynamic_frame_disp.numel() > 0:
            teleport_ratio = (dynamic_frame_disp > 0.5 * scene_scale).float().mean()
            E_teleport = teleport_ratio
        else:
            E_teleport = torch.tensor(0.0)
    else:
        E_teleport = torch.tensor(0.0)
    
    # ========== 组合 ==========
    R_motion = (gate * 
                torch.exp(-E_cam_accel / tau_cam) *
                torch.exp(-E_rot_accel / tau_rot) *
                (1.0 - E_teleport))
    
    return R_motion
3.7 G_anchor — 首帧锚定门控
pythondef compute_anchor_gate(pts_frame0, pts_input_frame, static_mask, extrinsics):
    """
    确保生成视频的首帧与输入条件帧在几何上一致。
    pts_frame0: 生成视频第一帧的 3D 点
    pts_input_frame: 输入条件帧的 3D 点（如果有）
    
    如果没有输入帧的 GT 几何，退化为检查首帧静态区域深度的合理性。
    """
    # 首帧静态区域的深度分布是否合理（非零、非 inf、方差合理）
    static_depth = pts_frame0[static_mask][..., 2]
    
    # 基本合理性检查
    valid_depth = (static_depth > 0.01) & (static_depth < 100.0)
    anchor_validity = valid_depth.float().mean()
    
    G_anchor = torch.sigmoid((anchor_validity - 0.8) / 0.05)
    return G_anchor
3.8 最终 Reward 公式
pythondef compute_reward(video_frames, model_4rc, config):
    """
    统一的 reward 计算（所有 checkpoint 使用相同逻辑）。
    """
    # 1. 4RC 推理
    predictions = model_4rc.inference(video_frames)
    pts = predictions["pts"]           # (N, H, W, 3)
    track = predictions["track"]       # (N, H, W, 3)
    conf = predictions["conf"]         # (N, H, W)
    conf_track = predictions["conf_track"]  # (N, H, W)
    extrinsics = predictions["extrinsic"]   # (N, 4, 4)
    intrinsics = predictions["intrinsic"]   # (N, 3, 3)
    
    # 2. 场景尺度归一化（首帧深度中位数）
    scene_scale = pts[0, ..., 2].median()
    
    # 3. Valid mask（conf 的唯一用途）
    valid_geo, valid_track = compute_valid_mask(conf, conf_track)
    
    # 4. Dynamic mask（逐帧最大位移法）
    static_mask, dynamic_mask = compute_dynamic_mask(track, threshold=0.01*scene_scale)
    
    # 5. 各项 reward
    R_static = compute_R_static(pts, extrinsics, intrinsics, static_mask, valid_geo, conf)
    R_dynamic = compute_R_dynamic(track, dynamic_mask, valid_track, scene_scale)
    R_motion = compute_R_motion(extrinsics, track, dynamic_mask, scene_scale)
    G_anchor = compute_anchor_gate(pts[0], None, static_mask, extrinsics)
    
    # 6. 总分
    R_total = G_anchor * (
        config.static_weight * R_static +
        config.dynamic_weight * R_dynamic +
        config.motion_weight * R_motion
    )
    
    details = {
        "R_static": R_static.item(),
        "R_dynamic": R_dynamic.item(),
        "R_motion": R_motion.item(),
        "G_anchor": G_anchor.item(),
        "R_total": R_total.item(),
        "scene_scale": scene_scale.item(),
        "dynamic_ratio": dynamic_mask.float().mean().item(),
    }
    
    return R_total.item(), details
所有 checkpoint（early/mid/final）使用完全相同的 compute_reward 函数，区别只是输入帧数不同（early 用 12 帧，mid/final 用 20 帧）。不再有单独的"简化 reward"。

4. Best-of-N 策略
4.1 保留渐进淘汰框架
N=8 候选同步去噪
    │
    ├── σ ≈ 0.83 (Step ~15/40) — Early Checkpoint
    │   ├── extract pred_x0 → VAE decode → 抽 12 帧
    │   ├── 4RC forward → compute_reward（完整公式，同 final）
    │   ├── 淘汰 bottom 50% → 4 个存活
    │   └── 保底规则不变
    │
    ├── σ ≈ 0.63 (Step ~25/40) — Mid Checkpoint  
    │   ├── extract pred_x0 → VAE decode → 抽 20 帧
    │   ├── 4RC forward → compute_reward（完整公式，同 final）
    │   ├── 淘汰 bottom 50% → 2 个存活
    │   └── 保底规则不变
    │
    └── Step 40/40 — Final
        ├── 完整 decode → 抽 20 帧 → 4RC → compute_reward
        └── 选出 best
4.2 计算量分析
DiT 步数: 8×15 + 4×10 + 2×15 = 190 步（同 V1）
4RC 推理次数: 8 + 4 + 2 = 14 次
帧数: early 12 帧, mid/final 20 帧
4.3 显存管理：模型交替加载
python# 去噪阶段
dit_model.to("cuda")
model_4rc.to("cpu")

# Checkpoint 评分阶段
dit_model.to("cpu")
model_4rc.to("cuda")
scores = [compute_reward(frames_i, model_4rc, config) for frames_i in candidates]
model_4rc.to("cpu")
dit_model.to("cuda")

5. 梯度引导设计
5.1 核心原则：对显式几何 loss 求梯度
不对 conf/conf_track 求梯度（避免 reward hacking）。梯度目标是让 latent 朝着"几何更一致"的方向移动：
pythonL_guidance = L_reproj + λ₁ * L_track_smoothness + λ₂ * L_anchor
每一项都有明确的物理含义：

L_reproj：静态区域跨帧深度重投影误差（越小越好）
L_track_smoothness：动态区域 3D 轨迹的加速度（越小越好）
L_anchor：首帧深度保持（越小越好）

5.2 实现
pythondef guided_denoise_step(latent, v_pred, sigma_t, vae, model_4rc, config):
    if not (config.sigma_min < sigma_t < config.sigma_max):
        return v_pred
    if step_idx % config.guidance_frequency != 0:
        return v_pred
    
    # 1. 计算 pred_x0
    x0_hat = (latent - sigma_t * v_pred).detach().requires_grad_(True)
    
    # 2. VAE differentiable decode
    video = vae.decode_differentiable(x0_hat)
    
    # 3. 抽帧 → 4RC forward（绕过 inference wrapper）
    frames = sample_fixed_frames(video, n=8)
    raw_output = model_4rc.forward(prepare_views(frames))
    
    # 4. 计算显式几何 loss（conf 作为 detached valid mask）
    with torch.no_grad():
        valid_mask = (raw_output["conf"] > threshold).detach()
        dynamic_mask = compute_dynamic_mask(raw_output["track"]).detach()
    
    loss = geometry_loss(raw_output, valid_mask, dynamic_mask)
    
    # 5. 反向传播
    grad = torch.autograd.grad(loss, x0_hat)[0]
    
    # 6. WMReward 风格修正
    scaling_t = 1.0 - sigma_t ** 2
    norm_ratio = v_pred.norm(2) / (grad.norm(2) + 1e-8)
    v_guided = v_pred + config.guidance_scale * norm_ratio * scaling_t * grad
    # 注意符号：loss 越小越好，所以梯度方向是 +grad（减小 loss）
    
    return v_guided
5.3 关键工程边界
需要新增的组件说明vae.decode_differentiable()新路径，不包在 no_grad 中model_4rc.forward() 直接调用绕过 inference() 的 @torch.no_gradfourrc_adapter.pyPIL → view dict 转换 + 全 Torch 输出（不经过 NumPy postprocess）
5.4 分阶段实施
Phase 1：先只做 BoN（无梯度，验证 reward 信号有效性）
Phase 2：Progressive BoN（验证 early score 与 final score 相关性）
Phase 3：梯度引导 + BoN（先确保 Phase 1/2 成功）

6. 配置参数
python@dataclass
class ReconRewardConfig:
    # 权重
    static_weight: float = 0.40
    dynamic_weight: float = 0.40
    motion_weight: float = 0.20
    
    # Dynamic mask
    dynamic_threshold_ratio: float = 0.01  # 相对 scene_scale 的阈值
    
    # R_static
    tau_reproj: float = 0.10         # 重投影误差温度
    occlusion_margin: float = 1.05   # 遮挡过滤余量
    
    # R_dynamic
    tau_accel: float = 0.05          # 加速度 penalty 温度
    tau_speed: float = 3.0           # 极端速度 penalty 温度
    max_sample_pixels: int = 1000    # 动态区域采样像素数
    
    # R_motion
    tau_cam: float = 0.02            # 相机加速度温度
    tau_rot: float = 0.05            # 旋转加速度温度
    min_motion: float = 0.005        # 最低运动量
    tau_motion: float = 0.005        # 运动门控温度
    
    # Valid mask
    conf_valid_quantile: float = 0.20  # conf 有效阈值分位数
    
    # 帧采样
    max_frames: int = 20
    image_size: int = 512
    
    # 梯度引导（Phase 3）
    guidance_scale: float = 0.001
    guidance_frequency: int = 5
    sigma_min: float = 0.08
    sigma_max: float = 0.45

7. 实验计划
Phase 0：Reward 信号验证（不需要 Wan2.2 生成）
方法 1：候选间方差分析

用已有 batch 生成视频（每 prompt 8 个候选）
对每个视频跑 4RC → 计算 R_total
检查同 prompt 内的变异系数 cv > 0.05

方法 2：扰动退化实验

对正常视频施加可控破坏（帧抖动、时间乱序、局部形变、帧复制）
检查 R_total 是否单调下降（Spearman ρ < -0.7）

方法 3：与 DA3 V1 reward 交叉验证

两个独立信号对同一批候选排序
计算排序相关性

Phase 1：完整 BoN
每个 prompt 生成 N=8，比较：

Random baseline
DA3 V1 reward 选优
4RC 显式几何 reward 选优（本方案）

评估：独立 evaluator（DA3 V1 score、4RC confidence、reproj error）+ FVD + CLIP-sim
Phase 2：Progressive BoN
扫描 sigma checkpoint、帧数、淘汰比例。要求 early score 与 final score 的 Spearman ρ ≥ 0.7 才启用。
Phase 3：梯度引导 + BoN
四个条件：vanilla、guidance only、BoN only、guidance + BoN。固定种子，监控几何指标和视觉质量是否同时提升。
Phase 4：下游机械臂任务（如条件允许）
从生成视频提取 6D 轨迹 → 仿真执行 → 评估成功率。

8. 技术风险与应对
风险概率影响应对4RC 在模糊 pred_x0 上输出崩溃中高Phase 0 先验证；如果早期无区分度，推迟 progressive 到更晚的 sigma显存不足（DiT + VAE + 4RC）高中模型交替加载；减少帧数R_static 和 R_dynamic 权重需要标定中中Phase 0 中通过扰动实验的各项 sensitivity 决定正确实现 bilinear reproj 的工程量确定低标准 CV 操作，有成熟实现可参考conf 作为 valid mask 的阈值选择中低用 per-video 分位数（Q20），避免跨视频绝对值漂移梯度引导破坏视觉质量中中小 ρ + 低频 + 监控 LPIPS/CLIP

9. 代码文件结构
geo_reward/
├── __init__.py              # 导出
├── da3_reward.py            # V1 保留（backward compatible，消融用）
├── recon_reward.py          # 【新增】4RC 显式几何一致性 Reward
│   ├── ReconRewardConfig
│   └── ReconstructionReward
│       ├── compute_reward()                # 统一 reward（所有 checkpoint）
│       └── compute_differentiable_loss()   # 可微 loss（Phase 3 梯度引导）
├── fourrc_adapter.py        # 【新增】4RC 接口适配
│   ├── frames_to_views()           # PIL/tensor → 4RC view dict
│   ├── compute_valid_mask()        # conf → valid mask
│   └── compute_dynamic_mask()      # track → static/dynamic mask
├── region_masks.py          # 保留（V1 兼容）
├── motion_reward.py         # 保留（V1 兼容）
├── bon_pipeline.py          # 修改：reward_type 参数 + 模型交替加载
├── guidance.py              # 【新增】梯度引导模块（Phase 3）
└── utils.py                 # 扩展：帧采样、坐标变换工具函数

10. 总结
本方案的核心思想：

用 4RC 的显式 3D 输出（pts、track、extrinsics）计算几何一致性——而非最大化模型的自我置信度。conf/conf_track 仅作为 valid mask，不参与分数计算。
单模型、一次推理、后验区分 static/dynamic——从 track 位移（逐帧最大位移法）自然推断，无需外部预分割或额外模型。
所有 checkpoint 使用统一的 reward 公式——不区分 early/full，只是输入帧数不同。
梯度引导对显式几何 loss 求梯度——不对 conf 求梯度，避免 reward hacking。

一句话总结：让 4RC 重建视频的 3D 结构，然后用显式几何验证（重投影、轨迹平滑、覆盖率）来评判质量——"能被一致地重建"意味着"几何信息准确"。