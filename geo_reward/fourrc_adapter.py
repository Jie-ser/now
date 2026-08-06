"""
4RC interface adapter for GeoReward V2.

Handles:
- PIL/tensor frames → 4RC view dict format
- 4RC model loading and inference
- conf/conf_track → valid mask computation
- track → static/dynamic mask computation (max-displacement method)

Requires:
- 4RC must be importable as `arc`. Either install with `pip install -e 4RC-main/4RC-main/`
  or add to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)/4RC-main/4RC-main`
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as tvf


# Auto-detect 4RC path if not already importable
def _ensure_4rc_importable():
    try:
        import arc  # noqa: F401
    except ImportError:
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', '4RC-main', '4RC-main'),
            os.path.join(os.path.dirname(__file__), '..', '4RC-main'),
        ]
        for path in candidates:
            path = os.path.abspath(path)
            if os.path.isdir(os.path.join(path, 'arc')):
                sys.path.insert(0, path)
                return
        raise ImportError(
            "Cannot import 'arc' (4RC). Install with `pip install -e 4RC-main/4RC-main/` "
            "or set PYTHONPATH to include the 4RC root directory."
        )


_ensure_4rc_importable()


PATCH_SIZE = 14


def pil_to_4rc_view(pil_img, target_size=518, patch_size=PATCH_SIZE):
    """
    Convert a single PIL Image to a 4RC view dict.

    4RC expects views with:
      - img: (1, 3, H, W) tensor, normalized to [-1, 1]
      - true_shape: np.int32 array of [H, W]
      - idx: int
      - instance: str

    Args:
        pil_img: PIL Image (RGB).
        target_size: Target long side resolution. Default 518 (4RC standard).
        patch_size: Patch size for alignment (must be divisible).

    Returns:
        dict with 'img', 'true_shape', 'idx', 'instance' keys.
    """
    img = pil_img.convert("RGB")
    W1, H1 = img.size

    img = _resize_pil_image(img, target_size)
    W, H = img.size

    cx, cy = W // 2, H // 2
    halfw = ((2 * cx) // patch_size) * patch_size // 2
    halfh = ((2 * cy) // patch_size) * patch_size // 2
    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    W2, H2 = img.size
    img_tensor = tvf.to_tensor(img)
    img_tensor = tvf.normalize(img_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    true_shape = np.int32([[H2, W2]])

    return {
        "img": img_tensor.unsqueeze(0),
        "true_shape": true_shape,
        "idx": 0,
        "instance": "0",
    }


def frames_to_views(frames, target_size=518):
    """
    Convert a list of PIL Images to a list of 4RC view dicts.

    Args:
        frames: List of PIL Images.
        target_size: Target resolution for 4RC input.

    Returns:
        List of view dicts ready for 4RC model inference.
    """
    views = []
    for i, frame in enumerate(frames):
        view = pil_to_4rc_view(frame, target_size=target_size)
        view["idx"] = i
        view["instance"] = str(i)
        views.append(view)
    return views


def run_4rc_inference(model, views, device="cuda", dtype="bf16-mixed"):
    """
    Run 4RC model inference on prepared views.

    Args:
        model: Loaded 4RC (Arc) model.
        views: List of view dicts from frames_to_views().
        device: Device string.
        dtype: Precision string for autocast.

    Returns:
        Dict with stacked outputs:
          - pts: (N, H, W, 3) world-coordinate 3D points
          - track: (N, H, W, 3) per-frame 3D displacement relative to query frame
          - conf: (N, H, W) depth confidence
          - conf_track: (N, H, W) track confidence
          - extrinsic: (N, 4, 4) camera-to-world transforms
          - intrinsic: (N, 3, 3) camera intrinsics
          - track_query_idx: int, which frame the track is relative to
    """
    from arc.dust3r.inference_multiview import inference

    result = inference(views, model, device=torch.device(device), dtype=dtype, verbose=False)
    preds = result["preds"]

    N = len(preds)
    pts_list = []
    track_list = []
    conf_list = []
    conf_track_list = []
    extrinsic_list = []
    intrinsic_list = []

    for i in range(N):
        pred = preds[i]
        pts_list.append(pred["pts"].squeeze(0))
        track_list.append(pred["track"].squeeze(0))
        conf_list.append(pred["conf"].squeeze(0))
        conf_track_list.append(pred["conf_track"].squeeze(0))
        extrinsic_list.append(pred["extrinsic"])
        intrinsic_list.append(pred["intrinsic"])

    pts = torch.stack(pts_list)
    track_abs = torch.stack(track_list)

    # 4RC track output is ABSOLUTE world coordinates (track_raw + pts[query_frame]).
    # Convert to relative displacement from the query frame's pts.
    track_query_idx = 0
    if N > 0 and "track_query_idx" in preds[0]:
        tqi = preds[0]["track_query_idx"]
        if isinstance(tqi, torch.Tensor):
            track_query_idx = int(tqi.flatten()[0].item())
        elif isinstance(tqi, (list, tuple)):
            track_query_idx = int(tqi[0])
        else:
            track_query_idx = int(tqi)

    # track_displacement[t, h, w] = track_abs[t, h, w] - pts[query_idx, h, w]
    track_displacement = track_abs - pts[track_query_idx].unsqueeze(0)

    return {
        "pts": pts,
        "track": track_displacement,
        "conf": torch.stack(conf_list),
        "conf_track": torch.stack(conf_track_list),
        "extrinsic": torch.stack(extrinsic_list),
        "intrinsic": torch.stack(intrinsic_list),
        "track_query_idx": track_query_idx,
    }


def compute_valid_mask(conf, conf_track, quantile=0.20):
    """
    Compute valid masks from conf/conf_track using per-video quantile thresholding.

    conf/conf_track only determine which pixels participate in scoring — they
    never contribute to the reward score itself (avoid reward hacking).

    Args:
        conf: (N, H, W) depth confidence.
        conf_track: (N, H, W) tracking confidence.
        quantile: Bottom fraction filtered out (default: Q20 → keep top 80%).

    Returns:
        valid_geo: (N, H, W) bool tensor.
        valid_track: (N, H, W) bool tensor.
    """
    conf_flat = conf.flatten()
    conf_finite = conf_flat[conf_flat.isfinite()]
    if conf_finite.numel() > 0:
        conf_threshold = torch.quantile(conf_finite, quantile)
        # Use >= to handle constant-value case (all equal → all valid)
        valid_geo = conf >= conf_threshold
    else:
        valid_geo = torch.ones_like(conf, dtype=torch.bool)

    track_flat = conf_track.flatten()
    track_finite = track_flat[track_flat.isfinite()]
    if track_finite.numel() > 0:
        track_threshold = torch.quantile(track_finite, quantile)
        valid_track = conf_track >= track_threshold
    else:
        valid_track = torch.ones_like(conf_track, dtype=torch.bool)

    return valid_geo, valid_track


def compute_dynamic_mask(track, threshold_ratio=0.01, scene_scale=None):
    """
    Compute static/dynamic masks from 4RC track output using max-displacement method.

    Uses per-frame maximum displacement rather than just first-last frame difference:
    captures "move then return" patterns and avoids false negatives.

    Args:
        track: (N, H, W, 3) per-frame 3D displacement relative to query frame.
        threshold_ratio: Displacement threshold as fraction of scene_scale.
        scene_scale: Scene depth scale. Required — caller must provide this
                     (e.g. from compute_scene_scale).

    Returns:
        static_mask: (H, W) bool tensor.
        dynamic_mask: (H, W) bool tensor.

    Raises:
        ValueError: If scene_scale is None.
    """
    if scene_scale is None:
        raise ValueError(
            "scene_scale is required for compute_dynamic_mask. "
            "Use compute_scene_scale(pts) to obtain it."
        )

    # track is relative to query frame (default: frame 0).
    # track[0] should be near-zero (self-to-self displacement).
    if track.shape[0] > 0:
        frame0_disp = torch.norm(track[0], dim=-1).mean()
        if frame0_disp > 0.1:
            import warnings
            warnings.warn(
                f"track[0] has mean displacement {frame0_disp:.4f}, expected ~0. "
                "This suggests query frame != frame 0. Dynamic mask may be inaccurate.",
                stacklevel=2,
            )

    displacement_per_frame = torch.norm(track, dim=-1)  # (N, H, W)
    max_displacement = displacement_per_frame.max(dim=0).values  # (H, W)

    threshold = threshold_ratio * scene_scale
    dynamic_mask = max_displacement > threshold
    static_mask = ~dynamic_mask

    return static_mask, dynamic_mask


def compute_scene_scale(pts, static_mask=None, extrinsic_frame0=None):
    """
    Compute scene scale from first frame depth (median of static region).

    If extrinsic_frame0 is provided, pts are transformed to the first frame's
    camera coordinate system before extracting depth (z-component). This handles
    the case where the world coordinate z-axis doesn't align with camera depth.

    If extrinsic_frame0 is None, assumes world coords are camera-canonical
    (4RC default: world frame = first camera frame).

    Args:
        pts: (N, H, W, 3) world-coordinate 3D points.
        static_mask: (H, W) bool. If None, uses entire first frame.
        extrinsic_frame0: (4, 4) camera-to-world for frame 0. If provided,
                          its inverse is used to transform pts to camera coords.

    Returns:
        Float scalar scene_scale (clamped to > 1e-6).
    """
    pts_frame0 = pts[0]  # (H, W, 3)

    if extrinsic_frame0 is not None:
        # Transform world points to camera-0 coordinate system
        w2c = torch.linalg.inv(extrinsic_frame0)  # (4, 4)
        H, W = pts_frame0.shape[:2]
        pts_flat = pts_frame0.reshape(-1, 3)
        pts_homo = torch.cat([pts_flat, torch.ones_like(pts_flat[:, :1])], dim=-1)
        pts_cam = (w2c @ pts_homo.T).T[:, :3].reshape(H, W, 3)
        depth_frame0 = pts_cam[..., 2]
    else:
        depth_frame0 = pts_frame0[..., 2]

    if static_mask is not None and static_mask.sum() > 50:
        depths = depth_frame0[static_mask]
    else:
        depths = depth_frame0.flatten()

    valid = depths[depths.isfinite() & (depths > 0.01)]
    if valid.numel() > 0:
        scale = valid.median().item()
    else:
        finite_depths = depths[depths.isfinite()]
        if finite_depths.numel() > 0:
            scale = finite_depths.abs().median().item()
        else:
            scale = 1.0

    return max(scale, 1e-6)


def _resize_pil_image(img, target_long_side):
    """Resize PIL image so that the longest side equals target_long_side."""
    W, H = img.size
    scale = target_long_side / max(W, H)
    new_W = int(round(W * scale))
    new_H = int(round(H * scale))
    return img.resize((new_W, new_H), Image.Resampling.LANCZOS)
