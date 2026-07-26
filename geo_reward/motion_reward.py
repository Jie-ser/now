"""
Motion reward: motion_gate, R_shape, R_smoothness.

Evaluates 3D motion quality of the dynamic region using point cloud geometry:
- motion_gate: ensures video has sufficient motion (static → 0)
- R_shape: local distance distribution stability (structure preservation)
- R_smoothness: one-step velocity outlier detection (no teleportation)
"""

import numpy as np


def depth_to_world_points(depth, extrinsic, intrinsic):
    """
    Unproject depth map to world coordinates.

    Args:
        depth: (H, W) depth map.
        extrinsic: (4, 4) or (3, 4) world-to-camera transform [R|t].
        intrinsic: (3, 3) camera intrinsics.

    Returns:
        (H, W, 3) world coordinate map.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack(
        [u.reshape(-1), v.reshape(-1), np.ones(h * w)],
        axis=0,
    )

    rays = np.linalg.inv(intrinsic) @ pixels
    points_camera = rays * depth.reshape(-1)[None, :]

    r = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    points_world = r.T @ (points_camera - t[:, None])

    return points_world.T.reshape(h, w, 3)


def robust_centroid(points):
    """Per-coordinate median centroid, robust to outliers."""
    if points is None or len(points) < 10:
        return None
    return np.median(points, axis=0)


def uniform_subsample_points(points, max_points=256):
    """
    Deterministically subsample points by lexicographic 3D sort then uniform spacing.

    Sorting by 3D coordinates decouples sampling from mask scan order,
    ensuring stable coverage across frames.
    """
    if len(points) <= max_points:
        return points

    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = points[order]

    indices = np.linspace(
        0,
        len(points) - 1,
        max_points,
    ).round().astype(np.int64)
    return points[indices]


def local_shape_signature(points, scene_scale, k_neighbors=8):
    """
    Compute local distance distribution signature for a point cloud.

    Uses k-nearest-neighbor distances, summarized as quantiles.
    Invariant to rigid motion (translation/rotation), sensitive to deformation.

    Args:
        points: (M, 3) point cloud.
        scene_scale: Scene depth scale for normalization.
        k_neighbors: Number of nearest neighbors per point.

    Returns:
        Array of 5 quantiles [0.10, 0.25, 0.50, 0.75, 0.90], or None.
    """
    if points is None or len(points) < k_neighbors + 1:
        return None

    points = uniform_subsample_points(points, max_points=256)
    points = points / scene_scale

    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    # Exclude self-distance
    np.fill_diagonal(distances, np.inf)

    nearest = np.partition(
        distances,
        kth=k_neighbors - 1,
        axis=1,
    )[:, :k_neighbors]

    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) < 10:
        return None

    return np.quantile(
        nearest,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    )


def shape_change_error(signature_a, signature_b, eps=1e-8):
    """
    Compute shape change between two local distance signatures via log-ratio.

    Returns scalar error, or None if either signature is missing.
    """
    if signature_a is None or signature_b is None:
        return None

    ratio = (signature_b + eps) / (signature_a + eps)
    return float(np.mean(np.abs(np.log(ratio))))


def velocity_turn_angles(vectors, eps=1e-8):
    """
    Compute turn angles between consecutive velocity vectors (diagnostic only).

    Not used in R_smoothness for V1, recorded for ablation.
    """
    angles = []
    for a, b in zip(vectors[:-1], vectors[1:]):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < eps or nb < eps:
            continue
        cosine = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        angles.append(float(np.arccos(cosine)))
    return np.asarray(angles)


def compute_motion_reward(
    dynamic_point_clouds,
    scene_scale,
    min_motion_threshold=0.01,
    tau_smooth=2.0,
    tau_shape=0.10,
    shape_weight=0.7,
    smooth_weight=0.3,
):
    """
    Compute the full motion reward: gate * (shape_w * R_shape + smooth_w * R_smoothness).

    Args:
        dynamic_point_clouds: List of (M_t, 3) arrays, one per frame.
        scene_scale: Scene depth scale for normalization.
        min_motion_threshold: Minimum motion energy for gate=1.
        tau_smooth: Temperature for smoothness mapping.
        tau_shape: Temperature for shape mapping.
        shape_weight: Weight for R_shape in motion quality.
        smooth_weight: Weight for R_smoothness in motion quality.

    Returns:
        Dict with: reward, gate, shape, smoothness, motion_energy,
                   smoothness_error, shape_error, valid.
    """
    centroids = [robust_centroid(points) for points in dynamic_point_clouds]

    vectors = []
    speeds = []
    for t in range(len(centroids) - 1):
        if centroids[t] is None or centroids[t + 1] is None:
            continue
        velocity = (centroids[t + 1] - centroids[t]) / scene_scale
        vectors.append(velocity)
        speeds.append(float(np.linalg.norm(velocity)))

    if len(speeds) < 2:
        return {
            "reward": 0.0,
            "gate": 0.0,
            "shape": 0.0,
            "smoothness": 0.0,
            "motion_energy": 0.0,
            "smoothness_error": None,
            "shape_error": None,
            "valid": False,
        }

    speeds = np.asarray(speeds, dtype=np.float32)
    motion_energy = float(np.median(speeds))
    motion_gate = min(1.0, motion_energy / (min_motion_threshold + 1e-8))

    # R_smoothness: q95/median excess
    v_median = float(np.median(speeds))
    v_high = float(np.quantile(speeds, 0.95))
    smooth_error = v_high / (v_median + 1e-8)
    smooth_excess = max(smooth_error - 1.0, 0.0)
    r_smooth = float(np.exp(-smooth_excess / tau_smooth))

    # R_shape: local distance distribution stability
    signatures = [
        local_shape_signature(points, scene_scale)
        for points in dynamic_point_clouds
    ]
    shape_errors = []
    for a, b in zip(signatures[:-1], signatures[1:]):
        error = shape_change_error(a, b)
        if error is not None:
            shape_errors.append(error)

    if shape_errors:
        shape_error_val = float(np.median(shape_errors))
        r_shape = float(np.exp(-shape_error_val / tau_shape))
    else:
        shape_error_val = None
        r_shape = 0.0

    reward = motion_gate * (
        shape_weight * r_shape
        + smooth_weight * r_smooth
    )

    return {
        "reward": float(reward),
        "gate": float(motion_gate),
        "shape": float(r_shape),
        "smoothness": float(r_smooth),
        "motion_energy": motion_energy,
        "smoothness_error": smooth_error,
        "shape_error": shape_error_val,
        "valid": True,
    }
