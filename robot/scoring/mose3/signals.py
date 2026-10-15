"""Three per-link deformation signals from MoSE3 outputs at the PDI points (CPU).

Points used: the link's PDI points that do not touch a MoSE3 depth edge and whose
query-frame depth is not an outlier (more than max(4 scaled MAD, 2% of the median)
from the link's median). The long axis ``a`` is the first principal direction of
their query-frame 3D positions ``x_i(q)``; ``s`` is the position along it and
``L = s_98 - s_2`` the link length (2nd-98th percentile, robust to stray points).

1. Rigidity discrepancy (the rigidity-vector signal at every frame):
       d_i(t) = 1 - cos(e_i(t), m(t)),  m(t) = normalized mean of the e_i(t)
       S_emb(t) = 90th percentile of d_i(t)
   over points visible and inside frame t; e_i(t) is the frame-t embedding at the
   point's tracked pixel.
2. Rigid residual (the track head against the rigidity head's SE(3)):
       r_i(t) = || R_i(t) x_i(q) + t_i(t) - x_i(t) || / L
       S_res(t) = 90th percentile of r_i(t) over visible points with se3_valid.
   ``S_refit`` replaces MoSE3's SE(3) with one Kabsch fit of the visible points.
   ``S_dense`` fits that one transform to the link's whole frame-zero mask instead
   (``dense_<link>.npz``): grid pixels at least 3 px inside the mask border, off depth
   edges, depth inliers (rule above) and visible at frame t, weighted by Cauchy IRLS
   (w = 1 / (1 + (r / 3 median r)^2), 10 iterations) so pixels where the mask leaks
   onto another surface lose their pull. It is scored at the same PDI points, with
   ``L_dense`` from the fitted pixels. A frame is undefined unless at least 200 pixels
   and 30% of the fitted pixels are visible and they span half of ``L_dense``.
   ``S_dense_all`` is the 90th percentile over the visible fitted pixels themselves.
3. Slice seams: K = 4 equal-length slices of [s_2, s_98] (outliers go to the end
   slices); one Kabsch fit per slice on its visible points, at least 8 of them. At
   seam k with point p_k = c0 + a e_{k+1}: gap = T_{k+1} p_k - T_k p_k;
   stretch = |gap . a(t)| / L, shear = |gap - (gap . a(t)) a(t)| / L, a(t) = R_k a;
   w = rotvec(R_k^T R_{k+1}); twist = |w . a|, bend = |w - (w . a) a| (degrees).

Video summaries use frames 1..T-1: the maximum (with its frame, and seam) and the
90th percentile over frames. No thresholds are applied; calibrating them needs real
robot clips.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot.workflows.mose3.inputs import LINKS
from robot.workflows.mose3.point_fields import quat_to_matrix
from robot.scoring.mose3.rigidity import kabsch

K = 4
MIN_SLICE = 8          # visible points per slice fit (the dense-pixel plan used 30)
MIN_POINTS = 5         # points per frame for the percentile signals
DENSE_ERODE_PX = 2     # mask border rings left out of the dense fit (signed distance > 2)
DENSE_MIN_PIXELS = 200
DENSE_MIN_VISIBLE = 0.3
DENSE_MIN_SPAN = 0.5
CAUCHY_C = 3.0
IRLS_ITERATIONS = 10
SEAM_PARTS = ('stretch', 'shear', 'bend_deg', 'twist_deg')


def depth_inliers(depth, keep):
    """``keep`` restricted to depths within max(4 scaled MAD, 2% of the median) of the median."""
    median = np.median(depth[keep])
    band = max(4 * 1.4826 * np.median(np.abs(depth[keep] - median)), 0.02 * median)
    return keep & (np.abs(depth - median) <= band)


def select_points(arrays):
    """Interior, depth-inlier PDI points; returns the boolean selection."""
    return depth_inliers(arrays['trajectories_world'][0, :, 2].astype(np.float64), ~arrays['on_depth_edge'])


def link_frame(xq):
    """Centre, unit long axis, positions along it, robust range and length."""
    c0 = xq.mean(0)
    a = np.linalg.svd(xq - c0, full_matrices=False)[2][0]
    s = (xq - c0) @ a
    low, high = np.percentile(s, [2, 98])
    return c0, a, s, low, high, high - low


def p90(values, ok):
    rows = np.full(len(values), np.nan)
    for t in range(len(values)):
        sel = ok[t] & np.isfinite(values[t])
        if sel.sum() >= MIN_POINTS:
            rows[t] = np.percentile(values[t][sel], 90)
    return rows


def rigidity_discrepancy(embeddings, ok):
    """Per-point 1 - cos to the link's mean embedding [T, N] (NaN where unavailable)."""
    unit = embeddings / np.clip(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-6, None)
    d = np.full(ok.shape, np.nan)
    for t in range(len(unit)):
        sel = ok[t] & np.isfinite(unit[t]).all(1)
        if sel.sum() >= MIN_POINTS:
            m = unit[t, sel].mean(0)
            d[t, sel] = 1.0 - unit[t, sel] @ (m / max(np.linalg.norm(m), 1e-9))
    return d


def rigid_residual(trajectories, quat, trans, ok, length):
    """Per-point residual against MoSE3's SE(3) [T, N] and the rigid predictions [T, N, 3]."""
    prediction = np.einsum('tnij,nj->tni', quat_to_matrix(quat), trajectories[0].astype(np.float64)) + trans
    r = np.linalg.norm(prediction - trajectories, axis=-1) / length
    return np.where(ok, r, np.nan), prediction


def refit_residual(trajectories, visible, length):
    """Per-point residual against one Kabsch fit of the visible points [T, N], and its predictions."""
    r = np.full(visible.shape, np.nan)
    prediction = np.full(trajectories.shape, np.nan)
    for t in range(len(trajectories)):
        sel = visible[t]
        if sel.sum() >= MIN_POINTS:
            rotation, translation = kabsch(trajectories[0, sel].astype(np.float64), trajectories[t, sel].astype(np.float64))
            prediction[t] = trajectories[0] @ rotation.T + translation
            r[t, sel] = np.linalg.norm(prediction[t, sel] - trajectories[t, sel], axis=1) / length
    return r, prediction


def weighted_kabsch(source, destination, weight):
    """Weighted least-squares rotation and translation mapping source [N, 3] onto destination."""
    w = weight / weight.sum()
    source_mean, destination_mean = w @ source, w @ destination
    covariance = (source - source_mean).T @ ((destination - destination_mean) * w[:, None])
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, destination_mean - rotation @ source_mean


def robust_rigid_fit(source, destination, floor):
    """Cauchy IRLS rigid fit; returns rotation, translation and each point's final weight.

    The Cauchy scale is ``CAUCHY_C`` times the median residual, never below ``floor``.
    """
    weight = np.ones(len(source))
    for _ in range(IRLS_ITERATIONS):
        rotation, translation = weighted_kabsch(source, destination, weight)
        residual = np.linalg.norm(source @ rotation.T + translation - destination, axis=1)
        scale = CAUCHY_C * max(float(np.median(residual)), floor)
        previous, weight = weight, 1.0 / (1.0 + (residual / scale) ** 2)
        if np.max(np.abs(weight - previous)) < 1e-4:
            break
    return rotation, translation, weight


def dense_fit_pixels(dense):
    """Mask pixels used by the dense fit: eroded, off depth edges, depth inliers."""
    keep = (dense['mask_signed_distance_px'].astype(np.float64) > DENSE_ERODE_PX) & ~dense['on_depth_edge']
    return depth_inliers(dense['tracks'][0, :, 2].astype(np.float64), keep)


def dense_residual(trajectories, visible, dense):
    """Residuals against one robust rigid fit of the link's mask pixels per frame.

    Returns the per-point residual [T, N] at the scored points (fractions of ``L_dense``),
    its rigid predictions [T, N, 3], the p90 over visible fitted pixels [T], the inlier
    fraction (weight >= 0.5) [T] and the fitted pixels' frame-zero summary.
    """
    fit = dense_fit_pixels(dense)
    tracks = dense['tracks'][:, fit].astype(np.float64)
    seen = dense['visibility_logit'][:, fit] > 0
    _, _, s, _, _, length = link_frame(tracks[0])
    count = len(trajectories)
    residual = np.full(visible.shape, np.nan)
    prediction = np.full(trajectories.shape, np.nan)
    pixel_p90, inliers = np.full(count, np.nan), np.full(count, np.nan)
    for t in range(count):
        sel = seen[t]
        if (sel.sum() < max(DENSE_MIN_PIXELS, DENSE_MIN_VISIBLE * len(sel))
                or np.ptp(np.percentile(s[sel], [2, 98])) < DENSE_MIN_SPAN * length):
            continue
        rotation, translation, weight = robust_rigid_fit(tracks[0, sel], tracks[t, sel], 1e-4 * length)
        prediction[t] = trajectories[0] @ rotation.T + translation
        scored = visible[t]
        residual[t, scored] = np.linalg.norm(prediction[t, scored] - trajectories[t, scored], axis=1) / length
        pixel = np.linalg.norm(tracks[0, sel] @ rotation.T + translation - tracks[t, sel], axis=1) / length
        pixel_p90[t], inliers[t] = np.percentile(pixel, 90), np.mean(weight >= 0.5)
    return residual, prediction, pixel_p90, inliers, dict(pixels=int(fit.sum()), length=float(length))


def seams(trajectories, visible, c0, a, s, low, high, length):
    """[T, K-1, 4]: stretch, shear (fractions of L), bend, twist (degrees); NaN where a slice is too small."""
    edges = np.linspace(low, high, K + 1)
    label = np.digitize(s, edges[1:-1])
    out = np.full((len(trajectories), K - 1, 4), np.nan)
    xq = trajectories[0].astype(np.float64)
    for t in range(len(trajectories)):
        fits = []
        for k in range(K):
            sel = (label == k) & visible[t]
            fits.append(kabsch(xq[sel], trajectories[t, sel].astype(np.float64)) if sel.sum() >= MIN_SLICE else None)
        for k in range(K - 1):
            if fits[k] is None or fits[k + 1] is None:
                continue
            (ra, ta), (rb, tb) = fits[k], fits[k + 1]
            p = c0 + a * edges[k + 1]
            gap = (rb @ p + tb) - (ra @ p + ta)
            axis = ra @ a
            along = gap @ axis
            w = Rotation.from_matrix(ra.T @ rb).as_rotvec()
            twist = abs(w @ a)
            out[t, k] = [abs(along) / length, np.linalg.norm(gap - along * axis) / length,
                         np.degrees(np.linalg.norm(w - (w @ a) * a)), np.degrees(twist)]
    return out


def summary(series):
    later = np.asarray(series[1:], np.float64)
    if not np.isfinite(later).any():
        return dict(max=None, max_frame=None, p90=None)
    frame = int(np.nanargmax(later)) + 1
    return dict(max=float(later[frame - 1]), max_frame=frame, p90=float(np.nanpercentile(later, 90)))


def link_signals(arrays, dense=None):
    """Signals for one link; the dense-fit signals need its cached mask fields ``dense``."""
    chosen = select_points(arrays)
    trajectories = arrays['trajectories_world'][:, chosen].astype(np.float64)
    visible = arrays['visible'][:, chosen]
    c0, a, s, low, high, length = link_frame(trajectories[0])
    embedding_ok = visible & arrays['in_frame'][:, chosen]
    se3_ok = visible & arrays['se3_valid'][:, chosen]
    residual, prediction = rigid_residual(trajectories, arrays['se3_quat_wxyz'][:, chosen],
                                          arrays['se3_trans'][:, chosen], se3_ok, length)
    refit, refit_prediction = refit_residual(trajectories, visible, length)
    per_point = dict(rigidity_discrepancy=rigidity_discrepancy(arrays['rigidity_embedding_tracked'][:, chosen], embedding_ok),
                     rigid_residual=residual, rigid_residual_refit=refit)
    predictions = dict(prediction=prediction, prediction_refit=refit_prediction)
    dense_result = {}
    if dense is not None:
        per_point['rigid_residual_dense'], predictions['prediction_dense'], pixel_p90, inliers, fitted = \
            dense_residual(trajectories, visible, dense)
        later = inliers[1:]
        dense_result = dict(dense_fit_pixels=fitted['pixels'], link_length_dense=fitted['length'],
                            dense_frames_defined=float(np.isfinite(later).mean()),
                            dense_inlier_fraction_median=float(np.nanmedian(later)) if np.isfinite(later).any() else None)
    series = {name: p90(values, np.isfinite(values)) for name, values in per_point.items()}
    if dense is not None:
        series['rigid_residual_dense_all'] = pixel_p90
    seam = seams(trajectories, visible, c0, a, s, low, high, length)
    result = dict(points=int(chosen.sum()), excluded_depth_edge=int(arrays['on_depth_edge'].sum()),
                  excluded_depth_outlier=int((~arrays['on_depth_edge'] & ~chosen).sum()), link_length=float(length),
                  **dense_result,
                  rigidity_discrepancy_frame0=None if not np.isfinite(series['rigidity_discrepancy'][0])
                  else float(series['rigidity_discrepancy'][0]),
                  **{name: summary(values) for name, values in series.items()})
    for index, part in enumerate(SEAM_PARTS):
        values = seam[:, :, index]                                     # [T, K-1]
        defined = np.isfinite(values).any(1)
        per_frame = np.full(len(values), np.nan)
        per_frame[defined] = np.nanmax(values[defined], axis=1)        # worst seam per frame
        result[f'seam_{part}'] = summary(per_frame)
        frame = result[f'seam_{part}']['max_frame']
        if frame is not None:
            result[f'seam_{part}']['seam'] = int(np.nanargmax(values[frame]))   # 0 = A|B, 1 = B|C, 2 = C|D
    result['seam_frames_defined'] = float(np.isfinite(seam).all(axis=(1, 2)).mean())
    frames = dict({name: [None if not np.isfinite(v) else float(v) for v in values] for name, values in series.items()},
                  seams=[[[None if not np.isfinite(v) else float(v) for v in parts] for parts in row] for row in seam])
    # Per-point values over every PDI point, NaN where a point is not scored (for the signal replay).
    count, total = len(trajectories), len(chosen)
    points = dict(selected=chosen)
    for name, values in dict(per_point, **predictions).items():
        full = np.full((count, total, *values.shape[2:]), np.nan, np.float32)
        full[:, chosen] = values
        points[name] = full
    return result, frames, points


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'inputs/manifest.json').read_text())
    output = args.run / 'signals'
    (output / 'frames').mkdir(parents=True, exist_ok=True)
    (output / 'points').mkdir(exist_ok=True)
    rows = []
    for entry in manifest['entries']:
        case = entry['video_id']
        per_case, arrays = {}, {}
        for link in LINKS:
            with np.load(args.run / 'cases' / case / f'{link}.npz', allow_pickle=False) as archive:
                link_arrays = {name: archive[name] for name in archive.files}
            dense_path = args.run / 'cases' / case / f'dense_{link}.npz'
            dense = None
            if dense_path.is_file():
                with np.load(dense_path, allow_pickle=False) as archive:
                    dense = {name: archive[name] for name in archive.files}
            result, frames, points = link_signals(link_arrays, dense)
            per_case[link] = frames
            arrays.update({f'{link}_{name}': values for name, values in points.items()})
            row = dict(case=case, link=link, mask_review=entry['masks'][link]['review'], frames=entry['frame_count'],
                       points=result['points'], excluded_depth_edge=result['excluded_depth_edge'],
                       excluded_depth_outlier=result['excluded_depth_outlier'], link_length=result['link_length'],
                       rigidity_discrepancy_frame0=result['rigidity_discrepancy_frame0'],
                       seam_frames_defined=result['seam_frames_defined'])
            row.update({key: result[key] for key in ('dense_fit_pixels', 'link_length_dense', 'dense_frames_defined',
                                                      'dense_inlier_fraction_median') if key in result})
            for name in ('rigidity_discrepancy', 'rigid_residual', 'rigid_residual_refit', 'rigid_residual_dense',
                         'rigid_residual_dense_all', *(f'seam_{p}' for p in SEAM_PARTS)):
                if name not in result:
                    continue
                for key in ('max', 'p90', 'max_frame', 'seam'):
                    if key in result[name]:
                        row[f'{name}_{key}'] = result[name][key]
            rows.append(row)
        (output / 'frames' / f'{case}.json').write_text(json.dumps(per_case) + '\n')
        np.savez_compressed(output / 'points' / f'{case}.npz', **arrays)
    names = list(dict.fromkeys(k for row in rows for k in row))
    with (output / 'signals.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(output=str(output), rows=len(rows))))


if __name__ == '__main__':
    main()
