"""
Static/Dynamic region segmentation using RGB frame diff + DA3 depth change rate.

Provides global and per-frame masks for separating background (static) from
moving regions (dynamic), used by R_scene and R_motion respectively.
"""

import cv2
import numpy as np


def rgb_to_gray(images):
    """Convert (N, H, W, 3) uint8 images to (N, H, W) float32 grayscale."""
    return np.stack(
        [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images],
        axis=0,
    ).astype(np.float32)


def compute_pair_image_dynamic(gray, threshold=15.0):
    """
    Compute per-pair image dynamic masks via absolute frame difference.

    Args:
        gray: (N, H, W) float32 grayscale frames.
        threshold: Pixel difference threshold.

    Returns:
        pair_masks: (N-1, H, W) bool — pixels exceeding threshold.
        pair_diffs: (N-1, H, W) float32 — absolute differences.
    """
    pair_masks = []
    pair_diffs = []
    for t in range(len(gray) - 1):
        diff = np.abs(gray[t + 1] - gray[t])
        pair_diffs.append(diff)
        pair_masks.append(diff > threshold)
    return np.stack(pair_masks), np.stack(pair_diffs)


def compute_pair_depth_dynamic(depths, threshold=0.05, eps=1e-6):
    """
    Compute per-pair depth dynamic masks via log-ratio depth change.

    Args:
        depths: (N, H, W) float depth maps.
        threshold: Log-ratio change threshold.
        eps: Small constant to avoid log(0).

    Returns:
        pair_masks: (N-1, H, W) bool — pixels with significant depth change.
        pair_changes: (N-1, H, W) float32 — log-ratio changes (NaN where invalid).
    """
    pair_changes = []
    pair_masks = []

    for t in range(len(depths) - 1):
        d0 = np.maximum(depths[t].astype(np.float32), eps)
        d1 = np.maximum(depths[t + 1].astype(np.float32), eps)

        change = np.abs(np.log(d1 / d0))
        valid = np.isfinite(change) & (d0 > eps) & (d1 > eps)

        pair_changes.append(np.where(valid, change, np.nan))
        pair_masks.append(valid & (change > threshold))

    return np.stack(pair_masks), np.stack(pair_changes)


def build_confident_masks(conf, quantile=0.5):
    """
    Build per-frame confidence masks using per-frame quantile thresholding.

    Args:
        conf: (N, H, W) confidence maps, or None.
        quantile: Fraction of pixels considered reliable (top 50% by default).

    Returns:
        (N, H, W) bool masks, or None if conf is None.
    """
    if conf is None:
        return None

    masks = []
    for c in conf:
        finite = np.isfinite(c)
        if not finite.any():
            masks.append(np.zeros_like(c, dtype=bool))
            continue
        threshold = np.quantile(c[finite], quantile)
        masks.append(finite & (c >= threshold))
    return np.stack(masks)


def pair_masks_to_frame_masks(pair_masks):
    """
    Convert (N-1, H, W) pair masks to (N, H, W) per-frame masks.

    Frame t is dynamic if either the (t-1, t) or (t, t+1) pair is dynamic.
    """
    n_pairs, h, w = pair_masks.shape
    n_frames = n_pairs + 1
    frame_masks = np.zeros((n_frames, h, w), dtype=bool)

    frame_masks[0] = pair_masks[0]
    frame_masks[-1] = pair_masks[-1]

    for t in range(1, n_frames - 1):
        frame_masks[t] = pair_masks[t - 1] | pair_masks[t]

    return frame_masks


def clean_binary_mask(mask, kernel_size=3):
    """Morphological open + close to remove isolated noise."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    x = mask.astype(np.uint8)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)
    return x.astype(bool)


def remove_small_components(mask, min_area):
    """Remove connected components smaller than min_area pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    output = np.zeros_like(mask, dtype=bool)
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = True
    return output


def build_static_dynamic_masks(
    images,
    depths,
    conf=None,
    image_diff_threshold=15.0,
    image_vote_ratio=0.30,
    depth_change_threshold=0.05,
    min_component_area_ratio=0.002,
    morph_kernel_size=3,
):
    """
    Build global and per-frame static/dynamic/unknown masks.

    Uses OR combination: a pixel is dynamic if EITHER image OR depth detects change.

    Args:
        images: (N, H, W, 3) uint8 RGB frames (DA3-aligned resolution).
        depths: (N, H, W) depth maps.
        conf: (N, H, W) confidence maps or None.
        image_diff_threshold: Pixel diff threshold for frame difference.
        image_vote_ratio: Fraction of pairs a pixel must exceed to be globally dynamic.
        depth_change_threshold: Log-ratio threshold for depth change.
        min_component_area_ratio: Minimum connected component area as fraction of image.
        morph_kernel_size: Kernel size for morphological cleanup.

    Returns:
        Dict with keys:
            global_static, global_dynamic, global_unknown: (H, W) bool
            frame_static, frame_dynamic, frame_unknown: (N, H, W) bool
            pair_image_diff: (N-1, H, W) float32
            pair_depth_change: (N-1, H, W) float32
    """
    gray = rgb_to_gray(images)

    pair_image_dynamic, pair_image_diff = compute_pair_image_dynamic(
        gray,
        threshold=image_diff_threshold,
    )
    pair_depth_dynamic, pair_depth_change = compute_pair_depth_dynamic(
        depths,
        threshold=depth_change_threshold,
    )

    confident = build_confident_masks(conf, quantile=0.5)
    if confident is not None:
        pair_confident = confident[:-1] & confident[1:]
        pair_depth_dynamic = pair_depth_dynamic & pair_confident
        pair_unknown = ~pair_confident
    else:
        pair_unknown = np.zeros_like(pair_depth_dynamic)

    # OR combination: either signal detects motion → dynamic
    pair_dynamic = pair_image_dynamic | pair_depth_dynamic

    # Global image dynamic: pixel exceeds threshold in > vote_ratio of pairs
    image_motion_ratio = pair_image_dynamic.mean(axis=0)
    image_dynamic_global = image_motion_ratio > image_vote_ratio

    # Global depth dynamic: median log-ratio exceeds threshold
    depth_change_median = np.nanmedian(pair_depth_change, axis=0)
    depth_dynamic_global = depth_change_median > depth_change_threshold

    global_dynamic = image_dynamic_global | depth_dynamic_global
    global_unknown = pair_unknown.mean(axis=0) > 0.5
    global_static = ~global_dynamic & ~global_unknown

    frame_dynamic = pair_masks_to_frame_masks(pair_dynamic)
    frame_unknown = pair_masks_to_frame_masks(pair_unknown)
    frame_static = ~frame_dynamic & ~frame_unknown

    # Morphological cleanup + small component removal
    h, w = depths.shape[1], depths.shape[2]
    min_area = int(h * w * min_component_area_ratio)

    global_dynamic = clean_binary_mask(global_dynamic, morph_kernel_size)
    global_dynamic = remove_small_components(global_dynamic, min_area)

    n_frames = len(depths)
    for t in range(n_frames):
        frame_dynamic[t] = clean_binary_mask(frame_dynamic[t], morph_kernel_size)
        frame_dynamic[t] = remove_small_components(frame_dynamic[t], min_area)

    # Recompute static after cleanup
    global_static = ~global_dynamic & ~global_unknown
    frame_static = ~frame_dynamic & ~frame_unknown

    return {
        "global_static": global_static,
        "global_dynamic": global_dynamic,
        "global_unknown": global_unknown,
        "frame_static": frame_static,
        "frame_dynamic": frame_dynamic,
        "frame_unknown": frame_unknown,
        "pair_image_diff": pair_image_diff,
        "pair_depth_change": pair_depth_change,
    }
