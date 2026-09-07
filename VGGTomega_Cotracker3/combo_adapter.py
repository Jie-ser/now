"""
VGGT-Omega + CoTracker3 组合适配器。

组合两个模型的输出，生成与 4RC 兼容的格式：
- pts (N, H, W, 3): 世界坐标 3D 点
- track (N, H, W, 3): 逐像素 3D 位移（相对帧 0）
- extrinsic (N, 4, 4): camera-to-world
- intrinsic (N, 3, 3): 相机内参
- visibility (N, H, W): 逐帧逐像素可见性

核心流程：
1. VGGT-Omega → depth, extrinsic_c2w, intrinsic, pts
2. CoTracker3 → 2D tracks, visibility（在与 VGGT-Omega 相同分辨率下）
3. 2D tracks + depth + 相机参数 → 3D displacement (track)
"""

import torch
import torch.nn.functional as F

from .vggt_omega_adapter import preprocess_frames, run_vggt_omega_inference
from .cotracker3_adapter import frames_to_video_tensor, run_dense_tracking


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
        image_resolution: VGGT-Omega 输入分辨率（长边参考）
        chunk_size: CoTracker3 每批追踪的像素数
        device: 设备

    Returns:
        dict:
          - pts: (N, H, W, 3) 世界坐标 3D 点
          - track: (N, H, W, 3) 逐像素 3D 位移（相对帧 0），不可见像素为 0
          - extrinsic: (N, 4, 4) camera-to-world
          - intrinsic: (N, 3, 3) 内参
          - visibility: (N, H, W) 逐帧逐像素可见性（bool）
    """
    N = len(frames_pil)
    print(f"[ComboAdapter] 开始组合推理: {N} 帧, resolution={image_resolution}")

    # ===== 步骤 1: VGGT-Omega 推理 =====
    print(f"[ComboAdapter] 步骤 1/4: VGGT-Omega 推理...")
    images = preprocess_frames(frames_pil, image_resolution=image_resolution)
    vggt_result = run_vggt_omega_inference(vggt_model, images, device=device)

    depth = vggt_result["depth"]           # (N, H_v, W_v)
    extrinsic_c2w = vggt_result["extrinsic_c2w"]  # (N, 4, 4)
    intrinsic = vggt_result["intrinsic"]   # (N, 3, 3)
    pts = vggt_result["pts"]               # (N, H_v, W_v, 3)
    H_v, W_v = vggt_result["resolution"]

    print(f"[ComboAdapter] VGGT-Omega 输出分辨率: ({H_v}, {W_v})")

    # ===== 步骤 2: CoTracker3 dense tracking =====
    # 将原始帧 resize 到与 VGGT-Omega 相同分辨率，确保坐标对齐
    print(f"[ComboAdapter] 步骤 2/4: CoTracker3 逐像素追踪...")
    video = frames_to_video_tensor(frames_pil, target_size=(H_v, W_v))

    tracks_2d, visibility = run_dense_tracking(
        cotracker_model, video,
        query_frame=0, chunk_size=chunk_size, device=device
    )
    # tracks_2d: (T, H_v, W_v, 2) — (x, y) 像素坐标
    # visibility: (T, H_v, W_v) — bool

    # ===== 步骤 3: 2D tracks → 3D displacement =====
    print(f"[ComboAdapter] 步骤 3/4: 2D→3D 反投影构造 track...")
    track = _construct_3d_track(
        tracks_2d, depth, intrinsic, extrinsic_c2w, pts, visibility
    )
    # track: (N, H_v, W_v, 3)

    # ===== 步骤 4: 返回 =====
    print(f"[ComboAdapter] 步骤 4/4: 组装输出")

    return {
        "pts": pts,
        "track": track,
        "extrinsic": extrinsic_c2w,
        "intrinsic": intrinsic,
        "visibility": visibility,
    }


def _construct_3d_track(tracks_2d, depth, intrinsic, extrinsic_c2w, pts, visibility):
    """
    从 2D tracks + depth + 相机参数构造 3D displacement track。

    对于帧 t 中每个像素 (h, w):
      1. 从 tracks_2d 获取该像素在帧 t 的 2D 追踪位置 (x_t, y_t)
      2. 在帧 t 的 depth map 上双线性插值获取 depth_t
      3. 反投影 (x_t, y_t, depth_t) 到世界坐标 pts_3d_t
      4. track[t, h, w] = pts_3d_t - pts[0, h, w]
      5. 不可见像素: track 设为 0

    Args:
        tracks_2d: (T, H, W, 2) — (x, y) 像素坐标
        depth: (N, H, W) — 各帧 depth map
        intrinsic: (N, 3, 3) — 各帧内参
        extrinsic_c2w: (N, 4, 4) — 各帧 camera-to-world
        pts: (N, H, W, 3) — 各帧世界坐标 3D 点
        visibility: (T, H, W) — bool

    Returns:
        track: (N, H, W, 3) — 3D displacement 相对帧 0
    """
    N, H, W = depth.shape
    T = tracks_2d.shape[0]
    assert T == N, f"帧数不匹配: tracks_2d T={T}, depth N={N}"

    track = torch.zeros(N, H, W, 3, dtype=depth.dtype, device=depth.device)

    # 帧 0 的世界坐标（作为基准）
    pts_frame0 = pts[0]  # (H, W, 3)

    for t in range(N):
        if t == 0:
            # 帧 0 对自身的 displacement 为 0
            continue

        # 获取该帧所有像素的追踪位置
        xy_t = tracks_2d[t]  # (H, W, 2), (x, y) 坐标

        x_t = xy_t[..., 0]  # (H, W) 列坐标
        y_t = xy_t[..., 1]  # (H, W) 行坐标

        # Clamp 坐标到有效范围（追踪到帧外的像素）
        x_t_clamped = x_t.clamp(0, W - 1)
        y_t_clamped = y_t.clamp(0, H - 1)

        # 在帧 t 的 depth map 上双线性插值采样
        depth_t_sampled = _bilinear_sample_depth(
            depth[t], x_t_clamped, y_t_clamped
        )  # (H, W)

        # 数值稳定性: depth 必须 > 0.01
        depth_t_sampled = depth_t_sampled.clamp(min=0.01)

        # 反投影到世界坐标
        pts_3d_t = _unproject_2d_to_world(
            x_t_clamped, y_t_clamped, depth_t_sampled,
            intrinsic[t], extrinsic_c2w[t]
        )  # (H, W, 3)

        # 3D displacement
        displacement = pts_3d_t - pts_frame0  # (H, W, 3)

        # 数值检查: 替换 NaN/Inf
        valid_disp = displacement.isfinite().all(dim=-1)  # (H, W)
        displacement[~valid_disp] = 0.0

        # 不可见像素的 displacement 设为 0
        vis_t = visibility[t]  # (H, W) bool
        displacement[~vis_t] = 0.0

        track[t] = displacement

    return track


def _bilinear_sample_depth(depth, x, y):
    """
    在 depth map 上双线性插值采样。

    使用 F.grid_sample 实现，坐标归一化到 [-1, 1]。

    Args:
        depth: (H, W) 单帧 depth map
        x: (H, W) float — 列坐标（可能是亚像素）
        y: (H, W) float — 行坐标（可能是亚像素）

    Returns:
        sampled_depth: (H, W)
    """
    H, W = depth.shape

    # 归一化坐标到 [-1, 1]
    grid_x = 2.0 * x / (W - 1) - 1.0  # (H, W)
    grid_y = 2.0 * y / (H - 1) - 1.0  # (H, W)

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0)  # (1, H, W, 2)

    depth_input = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    sampled = F.grid_sample(
        depth_input, grid,
        mode='bilinear', align_corners=True, padding_mode='border'
    )  # (1, 1, H, W)

    return sampled.squeeze(0).squeeze(0)  # (H, W)


def _unproject_2d_to_world(x, y, depth, intrinsic, extrinsic_c2w):
    """
    将 2D 像素坐标 + depth 反投影到世界坐标。

    Args:
        x: (H, W) 列坐标（像素）
        y: (H, W) 行坐标（像素）
        depth: (H, W) 深度
        intrinsic: (3, 3) 内参
        extrinsic_c2w: (4, 4) camera-to-world

    Returns:
        world_pts: (H, W, 3)
    """
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # 反投影到相机坐标
    x_cam = (x - cx) / fx * depth
    y_cam = (y - cy) / fy * depth
    z_cam = depth
    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)

    # 变换到世界坐标: p_world = R @ p_cam + t
    R = extrinsic_c2w[:3, :3]  # (3, 3)
    t = extrinsic_c2w[:3, 3]   # (3,)
    pts_world = pts_cam @ R.T + t  # (H, W, 3)

    return pts_world
