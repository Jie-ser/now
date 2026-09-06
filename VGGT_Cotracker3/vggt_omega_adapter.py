"""
VGGT-Omega 模型适配器。

封装 VGGT-Omega 模型的加载、预处理和推理，
将输出转换为 GeoReward 需要的标准格式（depth, extrinsic_c2w, intrinsic, pts）。

坐标约定：
- VGGT-Omega 输出 extrinsic 是 camera-from-world (3, 4)
- 本模块转换为 camera-to-world (4, 4) 以兼容现有 reward 计算
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# 自动检测 VGGT-Omega 路径
def _ensure_vggt_importable():
    try:
        import vggt_omega  # noqa: F401
    except ImportError:
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'VGGTomega', 'vggt-omega-main'),
            os.path.join(os.path.dirname(__file__), '..', 'VGGTomega'),
        ]
        for path in candidates:
            path = os.path.abspath(path)
            if os.path.isdir(os.path.join(path, 'vggt_omega')):
                sys.path.insert(0, path)
                return
        raise ImportError(
            "Cannot import 'vggt_omega'. Please install or add VGGTomega to PYTHONPATH."
        )


_ensure_vggt_importable()


def load_vggt_omega(checkpoint_path, device="cuda"):
    """
    加载 VGGT-Omega 模型。

    Args:
        checkpoint_path: 权重路径（.pth 文件）
        device: 设备

    Returns:
        model: VGGTOmega 实例（eval 模式）
    """
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    return model


def preprocess_frames(frames_pil, image_resolution=512, patch_size=16):
    """
    将 PIL 帧列表转为 VGGT-Omega 输入格式。

    参考 vggt_omega/utils/load_fn.py 的 load_and_preprocess_images()。
    使用 "balanced" 模式：保持总 token 数大致恒定（目标 token 数 = (resolution/patch)^2）。

    Args:
        frames_pil: List[PIL.Image]
        image_resolution: 目标分辨率（长边参考值）
        patch_size: 对齐的 patch 大小

    Returns:
        images: (N, 3, H, W) tensor, 值域 [0, 1]
    """
    import torchvision.transforms.functional as tvf

    target_tokens = (image_resolution // patch_size) ** 2  # 例如 (512/16)^2 = 1024

    processed = []
    for pil_img in frames_pil:
        img = pil_img.convert("RGB")
        W, H = img.size

        # 限制宽高比到 [0.5, 2.0]（中心裁剪）
        aspect = W / H
        if aspect > 2.0:
            new_W = int(H * 2.0)
            left = (W - new_W) // 2
            img = img.crop((left, 0, left + new_W, H))
            W = new_W
        elif aspect < 0.5:
            new_H = int(W / 0.5)
            top = (H - new_H) // 2
            img = img.crop((0, top, W, top + new_H))
            H = new_H

        W, H = img.size
        aspect = W / H

        # balanced 模式：目标 token 数 = target_tokens
        # w_patches = sqrt(tokens / aspect_ratio), h_patches = w_patches * aspect_ratio
        # 但实际 aspect = W/H，token_aspect = w_patches / h_patches
        # target_tokens = w_patches * h_patches
        # w_patches = sqrt(target_tokens * aspect)
        w_patches = max(1, round((target_tokens * aspect) ** 0.5))
        h_patches = max(1, round(target_tokens / w_patches))

        new_W = w_patches * patch_size
        new_H = h_patches * patch_size

        img = img.resize((new_W, new_H), Image.Resampling.BICUBIC)
        img_tensor = tvf.to_tensor(img)  # (3, H, W), [0, 1]
        processed.append(img_tensor)

    # 如果不同帧尺寸不同，填充到最大尺寸（白色填充）
    max_H = max(t.shape[1] for t in processed)
    max_W = max(t.shape[2] for t in processed)

    padded = []
    for t in processed:
        h, w = t.shape[1], t.shape[2]
        if h < max_H or w < max_W:
            pad_top = (max_H - h) // 2
            pad_bottom = max_H - h - pad_top
            pad_left = (max_W - w) // 2
            pad_right = max_W - w - pad_left
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
        padded.append(t)

    images = torch.stack(padded)  # (N, 3, H, W)
    return images


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
          - resolution: (H, W) 实际输出分辨率
    """
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = images.to(device)
    N, _, H, W = images.shape

    # VGGT-Omega forward
    with torch.no_grad():
        predictions = model(images.unsqueeze(0))  # 加 batch 维 → (1, N, 3, H, W)

    # 解析 depth: (1, N, H, W, 1) → (N, H, W)
    depth = predictions["depth"].squeeze(0).squeeze(-1)  # (N, H, W)

    # 解析相机参数
    pose_enc = predictions["pose_enc"]  # (1, N, 9)
    extrinsic_cfw, intrinsic = encoding_to_camera(
        pose_enc, (H, W), build_intrinsics=True
    )
    # extrinsic_cfw: (1, N, 3, 4) camera-from-world [R|T]
    # intrinsic: (1, N, 3, 3)

    extrinsic_cfw = extrinsic_cfw.squeeze(0)  # (N, 3, 4)
    intrinsic = intrinsic.squeeze(0)          # (N, 3, 3)

    # 转换为 camera-to-world (N, 4, 4)
    extrinsic_c2w = _camera_from_world_to_camera_to_world(extrinsic_cfw)

    # 反投影 depth → 世界坐标 pts (N, H, W, 3)
    pts = _unproject_depth_to_world_pts(
        depth, extrinsic_cfw, intrinsic
    )

    return {
        "depth": depth,
        "extrinsic_c2w": extrinsic_c2w,
        "intrinsic": intrinsic,
        "pts": pts,
        "resolution": (H, W),
    }


def _camera_from_world_to_camera_to_world(extrinsic_cfw):
    """
    将 camera-from-world (N, 3, 4) 转为 camera-to-world (N, 4, 4)。

    camera-from-world: p_cam = R @ p_world + T
    camera-to-world:   p_world = R^T @ (p_cam - T) = R^T @ p_cam - R^T @ T

    Args:
        extrinsic_cfw: (N, 3, 4) [R | T]

    Returns:
        extrinsic_c2w: (N, 4, 4)
    """
    N = extrinsic_cfw.shape[0]
    device = extrinsic_cfw.device

    R = extrinsic_cfw[:, :3, :3]   # (N, 3, 3)
    T = extrinsic_cfw[:, :3, 3:]   # (N, 3, 1)

    R_inv = R.transpose(-1, -2)    # R^T, (N, 3, 3)
    T_inv = -R_inv @ T             # -R^T @ T, (N, 3, 1)

    c2w = torch.eye(4, device=device, dtype=extrinsic_cfw.dtype).unsqueeze(0).expand(N, -1, -1).clone()
    c2w[:, :3, :3] = R_inv
    c2w[:, :3, 3:] = T_inv

    return c2w


def _unproject_depth_to_world_pts(depth, extrinsic_cfw, intrinsic):
    """
    反投影 depth map 到世界坐标系 3D 点。

    参考 demo_gradio.py 的 unproject_depth_map_to_point_map()。

    Args:
        depth: (N, H, W) depth map
        extrinsic_cfw: (N, 3, 4) camera-from-world [R|T]
        intrinsic: (N, 3, 3) 内参

    Returns:
        pts: (N, H, W, 3) 世界坐标 3D 点
    """
    N, H, W = depth.shape
    device = depth.device

    # 生成像素网格
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth.dtype),
        torch.arange(W, device=device, dtype=depth.dtype),
        indexing='ij'
    )
    # x, y: (H, W)
    x = x.unsqueeze(0).expand(N, -1, -1)  # (N, H, W)
    y = y.unsqueeze(0).expand(N, -1, -1)  # (N, H, W)

    fx = intrinsic[:, 0, 0].reshape(N, 1, 1)  # (N, 1, 1)
    fy = intrinsic[:, 1, 1].reshape(N, 1, 1)
    cx = intrinsic[:, 0, 2].reshape(N, 1, 1)
    cy = intrinsic[:, 1, 2].reshape(N, 1, 1)

    # 反投影到相机坐标系
    x_cam = (x - cx) / fx * depth
    y_cam = (y - cy) / fy * depth
    z_cam = depth
    camera_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N, H, W, 3)

    # camera-to-world 变换: p_world = R^T @ (p_cam - t)
    R = extrinsic_cfw[:, :3, :3]    # (N, 3, 3)
    t = extrinsic_cfw[:, :3, 3]     # (N, 3)

    R_T = R.transpose(-1, -2)       # (N, 3, 3)

    # camera_points - t: (N, H, W, 3) - (N, 1, 1, 3)
    shifted = camera_points - t.reshape(N, 1, 1, 3)

    # einsum: R_T @ shifted
    pts = torch.einsum('nij,nhwj->nhwi', R_T, shifted)  # (N, H, W, 3)

    return pts
