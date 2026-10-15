"""Read MoSE3's dense per-pixel predictions at explicit, sub-pixel query points.

MoSE3 predicts a field indexed by query-frame pixels. These helpers map PDI source
pixels to MoSE3's resized grid (pixel-centre convention, matching ``cv2.resize``),
sample its fields bilinearly, and fit one SE(3) per query with the upstream
weighted-Horn solver. With integer query pixels the fit equals
``fit_per_pixel_se3`` at those pixels; ``tests/test_point_fields.py`` checks this.
"""
from __future__ import annotations

import numpy as np


def source_to_grid(xy, source_hw, grid_hw):
    """Source-image pixels [N, 2] (x, y) -> MoSE3 grid pixels."""
    sy, sx = grid_hw[0] / source_hw[0], grid_hw[1] / source_hw[1]
    xy = np.asarray(xy, np.float64)
    return np.stack([(xy[:, 0] + 0.5) * sx - 0.5, (xy[:, 1] + 0.5) * sy - 0.5], axis=1)


def grid_to_source(uv, source_hw, grid_hw):
    """MoSE3 grid pixels [..., 2] -> source-image pixels."""
    sy, sx = source_hw[0] / grid_hw[0], source_hw[1] / grid_hw[1]
    uv = np.asarray(uv, np.float64)
    return np.stack([(uv[..., 0] + 0.5) * sx - 0.5, (uv[..., 1] + 0.5) * sy - 0.5], axis=-1)


def bilinear_taps(xy, hw, nearest=None):
    """Flat indices [N, 4] and weights [N, 4] of the four pixels around each point.

    Points are clamped to the grid. ``nearest`` [N] selects a one-hot tap at the
    rounded pixel, used where bilinear blending would mix across a depth edge.
    """
    h, w = hw
    x = np.clip(np.asarray(xy, np.float64)[:, 0], 0, w - 1)
    y = np.clip(np.asarray(xy, np.float64)[:, 1], 0, h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    index = np.stack([y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1], axis=1)
    weight = np.stack([(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy], axis=1)
    if nearest is not None and np.any(nearest):
        rows = np.flatnonzero(nearest)
        rx, ry = np.rint(x[rows]).astype(np.int64), np.rint(y[rows]).astype(np.int64)
        index[rows] = (ry * w + rx)[:, None]
        weight[rows] = [1.0, 0.0, 0.0, 0.0]
    return index, weight


def sample(field, index, weight):
    """field [..., H, W, C] at the same taps in every leading slice -> [..., N, C].

    Works on NumPy arrays and Torch tensors; taps come from ``bilinear_taps``.
    """
    flat = field.reshape(*field.shape[:-3], -1, field.shape[-1])
    if hasattr(field, 'new_tensor'):
        import torch
        gathered = flat[..., torch.as_tensor(index, device=field.device), :]   # [..., N, 4, C]
        return (gathered * field.new_tensor(weight)[..., None]).sum(dim=-2)
    return (flat[..., index, :] * weight[..., None]).sum(axis=-2)


def sample_per_frame(field, index, weight):
    """field [T, H, W, C] with frame-specific taps [T, N, 4] -> [T, N, C] (Torch or NumPy)."""
    count, height, width, channels = field.shape
    offset = (np.arange(count, dtype=np.int64) * height * width)[:, None, None]
    flat = field.reshape(-1, channels)
    if hasattr(field, 'new_tensor'):
        import torch
        gathered = flat[torch.as_tensor(index + offset, device=field.device)]
        return (gathered * field.new_tensor(weight)[..., None]).sum(dim=-2)
    return (flat[index + offset] * weight[..., None]).sum(axis=-2)


def touches_mask(mask, index, weight):
    """True where any non-zero-weight tap lands on a masked pixel."""
    return (np.asarray(mask).reshape(-1)[index] & (weight > 0)).any(axis=1)


def mask_band(mask, grid_hw, band_px):
    """A frame-zero source mask on the MoSE3 grid, plus a band of ``band_px`` around it.

    A grid pixel is in the mask when the source mask covers at least half of it
    (``cv2.INTER_AREA`` fraction). Returns the flat grid indices [M] of the mask and
    band, each pixel's covered fraction, and its signed distance in grid pixels to the
    mask border: inside, the distance to the nearest outside pixel (1 on the border
    ring); outside, minus the distance to the nearest mask pixel.
    """
    import cv2
    height, width = grid_hw
    fraction = cv2.resize(np.asarray(mask, np.float32), (width, height), interpolation=cv2.INTER_AREA)
    inside = (fraction >= 0.5).astype(np.uint8)
    depth_in = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    depth_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    signed = np.where(inside > 0, depth_in, -depth_out).reshape(-1)
    index = np.flatnonzero(signed >= -band_px)
    return index, fraction.reshape(-1)[index], signed[index]


def project(points, camera_poses, k):
    """World points [T, N, 3] through camera-to-world poses [T, 4, 4] -> uv [T, N, 2], depth [T, N]."""
    rotation, translation = camera_poses[:, None, :3, :3], camera_poses[:, None, :3, 3]
    camera = np.einsum('tnji,tnj->tni', np.broadcast_to(rotation, (*points.shape[:2], 3, 3)),
                       points - translation)
    z = camera[..., 2]
    safe = np.where(z > 1e-6, z, np.nan)
    uv = np.stack([k[0, 0] * camera[..., 0] / safe + k[0, 2], k[1, 1] * camera[..., 1] / safe + k[1, 2]], axis=-1)
    return uv, z


def fit_query_se3(tracks, rigidity_emb, query_xy, query_emb, *, query_idx=0, num_ref_pts=12000,
                  temperature=0.01, sigma_px=0.0, seed=0):
    """Per-query SE(3) from the query frame to every frame (Torch tensors).

    Same pool, softmax weighting and Horn solve as upstream ``fit_per_pixel_se3``;
    only the query rows differ: their embedding and pixel position are sampled at
    explicit sub-pixel locations ``query_xy`` [Q, 2] (x, y on the MoSE3 grid).
    Returns ``quat`` [T, Q, 4] (wxyz), ``trans`` [T, Q, 3] and ``ok`` [T, Q].
    """
    import torch
    import torch.nn.functional as F
    from mose3.utils.per_point_se3 import _rotmat_to_quat_wxyz, _sample_ref_indices
    from mose3.utils.se3_fit import solve_rotation_horn, weighted_centroids

    device = tracks.device
    count, height, width, _ = tracks.shape
    pixels = height * width
    source = tracks[query_idx].reshape(pixels, 3).float()
    source_emb = F.normalize(rigidity_emb[query_idx].reshape(pixels, -1).float(), dim=-1, eps=1e-6)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    reference = _sample_ref_indices(pixels, num_ref_pts, generator, device)
    reference_source = source.index_select(0, reference)
    reference_emb = source_emb.index_select(0, reference)
    logits = (F.normalize(query_emb.float(), dim=-1, eps=1e-6) @ reference_emb.T) / max(float(temperature), 1e-6)
    if sigma_px > 0:
        query = torch.as_tensor(query_xy, dtype=torch.float32, device=device)
        dh = query[:, 1, None] - (reference // width).float()[None]
        dw = query[:, 0, None] - (reference % width).float()[None]
        logits = logits - (dh * dh + dw * dw) / (2.0 * sigma_px ** 2)
    weights = F.softmax(logits, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    queries = len(weights)
    quat = torch.zeros(count, queries, 4, device=device)
    trans = torch.zeros(count, queries, 3, device=device)
    ok = torch.zeros(count, queries, dtype=torch.bool, device=device)
    quat[query_idx, :, 0] = 1.0
    ok[query_idx] = True
    for t in range(count):
        if t == query_idx:
            continue
        destination = tracks[t].reshape(pixels, 3).float().index_select(0, reference)
        source_mean, destination_mean, _, _, covariance = weighted_centroids(reference_source, destination, weights)
        horn = solve_rotation_horn(covariance, eig_min_gap=1e-4, eig_jitter=1e-6)
        rotation = horn['R']
        translation = destination_mean - (source_mean.unsqueeze(1) @ rotation.transpose(-2, -1)).squeeze(1)
        quat[t] = _rotmat_to_quat_wxyz(rotation)
        trans[t] = translation
        ok[t] = horn['ok'] & torch.isfinite(translation).all(dim=-1)
    return {'quat': quat, 'trans': trans, 'ok': ok}


def quat_to_matrix(quat):
    """wxyz quaternions [..., 4] -> rotation matrices [..., 3, 3] (NumPy)."""
    q = np.asarray(quat, np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
