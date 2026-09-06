"""
CoTracker3 逐像素 dense tracking 适配器。

封装 CoTracker3 offline 模型的加载和逐像素 dense tracking。
对指定帧的每个像素建立独立的 tracking query，分 chunk 处理避免显存溢出。

坐标约定：
- CoTracker3 queries 格式: (frame_idx, x, y)，x 是列坐标，y 是行坐标
- CoTracker3 tracks 输出: (x, y) 像素坐标
- CoTracker3 内部 resize 到 (384, 512) 然后 rescale 回原始分辨率
"""

import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image


# 自动检测 CoTracker3 路径
def _ensure_cotracker_importable():
    try:
        import cotracker  # noqa: F401
    except ImportError:
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'co-tracker-main', 'co-tracker-main'),
            os.path.join(os.path.dirname(__file__), '..', 'co-tracker-main'),
        ]
        for path in candidates:
            path = os.path.abspath(path)
            if os.path.isdir(os.path.join(path, 'cotracker')):
                sys.path.insert(0, path)
                return
        raise ImportError(
            "Cannot import 'cotracker'. Please install or add co-tracker to PYTHONPATH."
        )


_ensure_cotracker_importable()


def load_cotracker3(checkpoint_path=None, device="cuda"):
    """
    加载 CoTracker3 offline 模型。

    Args:
        checkpoint_path: 本地权重路径（.pth 文件）。None 则用 torch.hub 自动下载。
        device: 设备

    Returns:
        model: CoTrackerPredictor 实例
    """
    if checkpoint_path is not None:
        from cotracker.predictor import CoTrackerPredictor
        model = CoTrackerPredictor(checkpoint=checkpoint_path, offline=True)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to(device)
    return model


def frames_to_video_tensor(frames_pil, target_size=None):
    """
    将 PIL 帧列表转为 CoTracker3 输入格式。

    Args:
        frames_pil: List[PIL.Image]
        target_size: (H, W) 如果需要 resize（与 VGGT-Omega 输出分辨率对齐）

    Returns:
        video: (1, T, 3, H, W) float tensor, 值域 [0, 255]
    """
    import torchvision.transforms.functional as tvf

    tensors = []
    for pil_img in frames_pil:
        img = pil_img.convert("RGB")
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

        # tvf.to_tensor 输出 [0, 1]，CoTracker3 需要 [0, 255]
        t = tvf.to_tensor(img) * 255.0  # (3, H, W), [0, 255]
        tensors.append(t)

    video = torch.stack(tensors).unsqueeze(0)  # (1, T, 3, H, W)
    return video


def run_dense_tracking(model, video, query_frame=0, chunk_size=4096, device="cuda"):
    """
    逐像素 dense tracking：对 query_frame 的每个像素追踪到所有帧。

    对 H*W 个像素生成 queries，分 chunk 调用 CoTracker3，拼接结果。

    Args:
        model: CoTrackerPredictor
        video: (1, T, 3, H, W) tensor, 值域 [0, 255]
        query_frame: 起始帧索引（默认 0）
        chunk_size: 每批处理的查询点数（控制显存）
        device: 设备

    Returns:
        tracks_2d: (T, H, W, 2) float tensor — 每个像素在每帧的 (x, y) 坐标
        visibility: (T, H, W) bool tensor — 每个像素在每帧的可见性
    """
    video = video.to(device)
    B, T, C, H, W = video.shape
    assert B == 1, "batch size 必须为 1"

    total_pixels = H * W

    # 生成所有像素的 queries: (1, H*W, 3)
    # 格式: (frame_idx, x, y)，x 是列坐标，y 是行坐标
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    # xs: (H, W) 列坐标, ys: (H, W) 行坐标
    queries_all = torch.stack([
        torch.full((total_pixels,), query_frame, dtype=torch.float32),
        xs.reshape(-1),   # x（列坐标）
        ys.reshape(-1),   # y（行坐标）
    ], dim=-1).unsqueeze(0).to(device)  # (1, H*W, 3)

    # 分 chunk 处理
    tracks_chunks = []
    vis_chunks = []
    num_chunks = (total_pixels + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_pixels)
        queries_chunk = queries_all[:, start:end, :]  # (1, chunk, 3)

        if chunk_idx % 10 == 0:
            print(f"  [CoTracker3] 追踪 chunk {chunk_idx+1}/{num_chunks} "
                  f"(像素 {start}-{end}/{total_pixels})")

        # CoTracker3 predictor.forward() 有 @torch.no_grad()
        tracks_chunk, vis_chunk = model(video, queries=queries_chunk)
        # tracks_chunk: (1, T, chunk, 2)
        # vis_chunk: (1, T, chunk)

        tracks_chunks.append(tracks_chunk)
        vis_chunks.append(vis_chunk)

    # 拼接所有 chunk
    tracks_all = torch.cat(tracks_chunks, dim=2)  # (1, T, H*W, 2)
    vis_all = torch.cat(vis_chunks, dim=2)         # (1, T, H*W)

    # reshape 到空间维度
    tracks_2d = tracks_all.squeeze(0).reshape(T, H, W, 2)  # (T, H, W, 2)
    visibility = vis_all.squeeze(0).reshape(T, H, W).bool()  # (T, H, W)

    return tracks_2d, visibility
