"""Render MoSE3's 2D replay views for the PDI-initialized Link2/Link5 points (CPU).

Uses upstream ``visualize.render_tracks`` and ``render_se3_axes`` unchanged: our N
points are passed as a 1 x N "image" of tracks, SE(3) and visibility logits, so the
trails, hollow occluded heads, arrowed / dashed tripods and axis colours match the
project page. Frames are drawn at the MoSE3 grid scaled by 10/7 (740 px wide).

Views per video: ``tracks`` (projected through each frame's camera), ``tracks_world``
(re-rendered from the frame-zero camera over frame zero, so camera motion is
removed), ``se3`` (tripods on 24 spread-out points per link), ``combo`` (all trails
over those tripods) and, when the display-only render pass exists, ``pca`` (upstream
rigidity view) and ``se3_on_pca`` (tripods over it).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

from robot.replay.mose3.build_replay import TRIPODS_PER_LINK, spread_subset
from robot.workflows.mose3.inputs import LINKS, REPO, sha

MOSE3 = REPO / 'infrastructure/vendor/MoSE3'
SCALE = 10 / 7          # 518 x 294 grid -> 740 x 420 display, both even
TRAIL = 12


def upstream():
    if str(MOSE3) not in sys.path:
        sys.path.insert(0, str(MOSE3))
    import visualize
    return visualize


@contextmanager
def start_frames(visualize, frames, ok):
    """Give ``render_se3_axes`` the surface-normal start frames already saved per point.

    Upstream computes them from the dense query-frame point map, which a 1 x N point
    "image" cannot reproduce; the tracking run stored that exact upstream result.
    """
    original = visualize.surface_normal_frames
    visualize.surface_normal_frames = lambda *args, **kwargs: (frames, ok)
    try:
        yield
    finally:
        visualize.surface_normal_frames = original


def display_intrinsics(k):
    k = np.asarray(k, np.float64).copy()
    k[0, 0] *= SCALE
    k[1, 1] *= SCALE
    k[:2, 2] = (k[:2, 2] + 0.5) * SCALE - 0.5
    return k


def decode(video, count, size):
    capture = cv2.VideoCapture(str(video))
    frames = []
    for _ in range(count):
        ok, bgr = capture.read()
        if not ok:
            raise ValueError(f'Cannot decode {count} frames: {video}')
        frames.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA))
    capture.release()
    return np.stack(frames)


def write(path, frames, fps):
    imageio.mimwrite(path, list(frames), fps=fps, quality=7, macro_block_size=1)


def render_case(case, run, sources, output):
    visualize = upstream()
    folder = run / 'cases' / case
    receipt = json.loads((folder / 'receipt.json').read_text())
    entry = next(e for e in json.loads((run / 'inputs/manifest.json').read_text())['entries'] if e['video_id'] == case)
    links = []
    for link in LINKS:
        with np.load(folder / f'{link}.npz', allow_pickle=False) as archive:
            links.append({name: archive[name] for name in archive.files})
    # Per-frame arrays join along points (axis 1); per-point arrays along axis 0.
    joined = {name: np.concatenate([l[name] for l in links], axis=1)
              for name in ('trajectories_world', 'visibility_logit', 'se3_quat_wxyz', 'se3_trans', 'se3_valid')}
    joined.update({name: np.concatenate([l[name] for l in links])
                   for name in ('display_frame', 'display_frame_ok', 'trail_rgb')})
    tripod = np.concatenate([spread_subset(l['query_xy_source'].astype(np.float64), TRIPODS_PER_LINK) for l in links])
    count, n = joined['trajectories_world'].shape[:2]
    height, width = receipt['grid_hw']
    size = (round(width * SCALE), round(height * SCALE))
    k = display_intrinsics(receipt['intrinsics'])
    with np.load(folder / 'scene.npz', allow_pickle=False) as scene:
        poses = scene['camera_poses'].astype(np.float64)
    video = sources / entry['video']
    if sha(video) != entry['source_video_sha256']:
        raise ValueError(f'Local source copy differs from the tracked video: {case}')
    images = decode(video, count, size)
    fps = receipt['fps']

    tracks = joined['trajectories_world'][:, None]                         # [T, 1, N, 3] point "image"
    logits = joined['visibility_logit'][:, None]
    pts = np.stack([np.zeros(n, np.int64), np.arange(n)], axis=1)
    colors = joined['trail_rgb'].astype(np.uint8)
    destination = output / case
    destination.mkdir(parents=True, exist_ok=True)
    views = {}
    views['tracks'] = visualize.render_tracks(images, tracks, poses, k, logits, pts, colors, trail=TRAIL)
    fixed = np.repeat(poses[:1], count, axis=0)
    still = np.repeat(images[:1], count, axis=0)
    views['tracks_world'] = visualize.render_tracks(still, tracks, fixed, k, logits, pts, colors, trail=TRAIL)

    def tripods(background):
        sub = np.flatnonzero(tripod)
        with start_frames(visualize, joined['display_frame'][sub].astype(np.float64), joined['display_frame_ok'][sub]):
            return visualize.render_se3_axes(
                background, tracks[:, :, sub], joined['se3_quat_wxyz'][:, None, sub], joined['se3_trans'][:, None, sub],
                joined['se3_valid'][:, None, sub], logits[:, :, sub], poses, k, 0,
                np.stack([np.zeros(len(sub), np.int64), np.arange(len(sub))], axis=1), colors[sub],
                axis_len=20.0 * SCALE, normal_radius=12)

    views['se3'] = tripods(images)
    views['combo'] = visualize.render_tracks(views['se3'], tracks, poses, k, logits, pts, colors, trail=TRAIL)
    pca_video = run / 'renders' / case / 'rigidity_pca.mp4'
    if pca_video.is_file():
        render = json.loads((run / 'renders' / case / 'receipt.json').read_text())
        if render['files']['rigidity_pca.mp4'] != sha(pca_video):
            raise ValueError(f'Display render hash mismatch: {case}')
        pca = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_LINEAR)
                        for f in imageio.mimread(pca_video, memtest=False)])[:count]
        if len(pca) != count:
            raise ValueError(f'Rigidity render frame count differs: {case}')
        views['pca'] = pca
        views['se3_on_pca'] = tripods(pca)
    views['input'] = images
    for name, frames in views.items():
        write(destination / f'{name}.mp4', frames, fps)
    imageio.imwrite(destination / 'poster.jpg', views['combo'][0], quality=85)
    return case, sorted(views)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: <run>/replay/views')
    parser.add_argument('--processes', type=int, default=6)
    parser.add_argument('--cases', help='Comma-separated subset')
    args = parser.parse_args()
    output = args.output or args.run / 'replay/views'
    cases = args.cases.split(',') if args.cases else sorted(p.name for p in (args.run / 'cases').iterdir()
                                                           if (p / 'receipt.json').is_file())
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        for case, views in pool.map(partial(render_case, run=args.run, sources=args.sources, output=output), cases):
            print('VIEWS', case, ','.join(views), flush=True)


if __name__ == '__main__':
    main()
