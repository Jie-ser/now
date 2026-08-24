import numpy as np
from PIL import Image
import torch


def wan_output_to_pil(video_tensor):
    """
    Convert Wan2.2 output tensor to list of PIL Images.

    Args:
        video_tensor: Wan2.2 output, shape (3, T, H, W), range [-1, 1], torch.Tensor

    Returns:
        List of PIL Images, one per frame.
    """
    video = (video_tensor + 1) / 2
    video = video.clamp(0, 1)
    video = video.permute(1, 0, 2, 3)  # (T, 3, H, W)
    video = (video * 255).byte().cpu().numpy()
    video = video.transpose(0, 2, 3, 1)  # (T, H, W, 3)

    frames = [Image.fromarray(video[t]) for t in range(video.shape[0])]
    return frames


def sample_frames(total_frames=81, max_frames=20):
    """
    Uniformly sample frame indices, always including the first frame.

    Args:
        total_frames: Total number of frames in the video.
        max_frames: Maximum number of frames to sample.

    Returns:
        Sorted list of unique frame indices.
    """
    if max_frames >= total_frames:
        return list(range(total_frames))

    indices = [0]
    step = (total_frames - 1) / (max_frames - 1)
    for i in range(1, max_frames):
        indices.append(int(round(i * step)))
    return sorted(set(indices))


def transform_to_camera(pts_world, w2c):
    """
    Transform world-coordinate points to camera coordinates.

    Args:
        pts_world: (..., 3) world-coordinate points (torch tensor).
        w2c: (4, 4) world-to-camera transform.

    Returns:
        (..., 3) points in camera coordinates.
    """
    original_shape = pts_world.shape
    pts_flat = pts_world.reshape(-1, 3)
    pts_homo = torch.cat([pts_flat, torch.ones_like(pts_flat[:, :1])], dim=-1)
    pts_cam = (w2c @ pts_homo.T).T[:, :3]
    return pts_cam.reshape(original_shape)
