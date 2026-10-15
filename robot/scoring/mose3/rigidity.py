"""Per-frame rigidity diagnostics for one link from MoSE3 point outputs (NumPy only).

None of these is a validated deformation detector. Every series has frame zero
equal to zero (or NaN where undefined), and video summaries average frames 1..T-1.

- ``rigid_fit_residual``: RMS distance between each visible point and one
  least-squares rigid motion of the link (Kabsch), divided by the link's
  frame-zero RMS radius.
- ``pair_ratio_mad``: the PDI ratio formula, ``median |r - m| / (m + 1e-6)`` with
  ``r = d_ij(t) / d_ij(0)``, over **all** visible point pairs; no pair selection.
  With fewer than three available pairs the previous frame's value is carried.
- ``se3_rotation_spread_deg``: median angle between each point's MoSE3 rotation
  and the link's chordal-mean rotation; a rigid link shares one rotation.
- ``embedding_dispersion``: mean ``1 - cos`` between each visible point's tracked
  rigidity embedding and the link's mean direction.
"""
from __future__ import annotations

import numpy as np

MIN_POINTS = 5
MIN_PAIRS = 3


def kabsch(source, destination):
    """Least-squares rotation and translation mapping source [N, 3] onto destination."""
    source_mean, destination_mean = source.mean(0), destination.mean(0)
    covariance = (source - source_mean).T @ (destination - destination_mean)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, destination_mean - rotation @ source_mean


def rigid_fit_residual(trajectories, visible):
    """Normalized per-frame RMS residual [T] and per-point residual [T, N] (NaN when unavailable)."""
    count, points, _ = trajectories.shape
    reference = trajectories[0].astype(np.float64)
    radius = np.sqrt(np.mean(np.sum((reference - reference.mean(0)) ** 2, axis=1)))
    series = np.full(count, np.nan)
    per_point = np.full((count, points), np.nan)
    for t in range(count):
        keep = visible[t] & visible[0] & np.isfinite(trajectories[t]).all(1) & np.isfinite(reference).all(1)
        if keep.sum() < MIN_POINTS:
            continue
        rotation, translation = kabsch(reference[keep], trajectories[t, keep].astype(np.float64))
        error = np.linalg.norm(reference[keep] @ rotation.T + translation - trajectories[t, keep], axis=1)
        per_point[t, keep] = error / max(radius, 1e-9)
        series[t] = np.sqrt(np.mean(error ** 2)) / max(radius, 1e-9)
    return series, per_point


def pair_ratio_mad(trajectories, visible, min_baseline_fraction=1e-3):
    """PDI distance-ratio MAD/median over all visible pairs; returns (series [T], carried frames)."""
    count, points, _ = trajectories.shape
    i, j = np.triu_indices(points, 1)
    reference = trajectories[0].astype(np.float64)
    baseline = np.linalg.norm(reference[i] - reference[j], axis=1)
    extent = np.sqrt(np.mean(np.sum((reference - reference.mean(0)) ** 2, axis=1)))
    usable = np.isfinite(baseline) & (baseline > min_baseline_fraction * max(extent, 1e-9)) & visible[0, i] & visible[0, j]
    i, j, baseline = i[usable], j[usable], baseline[usable]
    if len(baseline) < MIN_PAIRS:  # nothing to carry: undefined rather than a fabricated zero
        series = np.full(count, np.nan)
        series[0] = 0.0
        return series, list(range(1, count))
    series, carried, previous = np.zeros(count), [], 0.0
    for t in range(1, count):
        available = visible[t, i] & visible[t, j]
        distance = np.linalg.norm(trajectories[t, i[available]].astype(np.float64)
                                  - trajectories[t, j[available]], axis=1)
        ratio = distance / baseline[available]
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio) < MIN_PAIRS:
            series[t] = previous
            carried.append(t)
            continue
        median = np.median(ratio)
        series[t] = previous = float(np.median(np.abs(ratio - median)) / (median + 1e-6))
    return series, carried


def rotation_spread_deg(rotations, valid):
    """Median geodesic angle to the chordal L2 mean rotation, per frame [T]."""
    count = len(rotations)
    series = np.full(count, np.nan)
    for t in range(count):
        keep = valid[t]
        if keep.sum() < MIN_POINTS:
            continue
        u, _, vt = np.linalg.svd(rotations[t, keep].mean(0))
        mean = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt)) or 1.0]) @ vt
        relative = np.einsum('ij,nik->njk', mean, rotations[t, keep])
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
        series[t] = float(np.degrees(np.median(np.arccos(cosine))))
    return series


def embedding_dispersion(embeddings, visible):
    """Per-frame mean (1 - cosine) to the link's mean embedding direction [T]."""
    unit = embeddings / np.clip(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-6, None)
    series = np.full(len(embeddings), np.nan)
    for t in range(len(embeddings)):
        keep = visible[t] & np.isfinite(unit[t]).all(1)
        if keep.sum() < MIN_POINTS:
            continue
        mean = unit[t, keep].mean(0)
        mean /= max(np.linalg.norm(mean), 1e-9)
        series[t] = float(np.mean(1.0 - unit[t, keep] @ mean))
    return series


def summarize(trajectories, visible, rotations, rotation_valid, embeddings):
    """All per-frame series and their frame 1..T-1 means for one link."""
    residual, _ = rigid_fit_residual(trajectories, visible)
    ratio, carried = pair_ratio_mad(trajectories, visible)
    spread = rotation_spread_deg(rotations, rotation_valid & visible)
    dispersion = embedding_dispersion(embeddings, visible)
    series = dict(rigid_fit_residual=residual, pair_ratio_mad=ratio,
                  se3_rotation_spread_deg=spread, embedding_dispersion=dispersion)

    def mean(values):
        later = values[1:]
        later = later[np.isfinite(later)]
        return float(later.mean()) if len(later) else None

    return dict(
        series={name: [None if not np.isfinite(v) else float(v) for v in values] for name, values in series.items()},
        scores={name: mean(values) for name, values in series.items()},
        carried_pair_frames=carried,
        visible_fraction=[float(v) for v in visible.mean(axis=1)],
        points=int(trajectories.shape[1]),
    )
