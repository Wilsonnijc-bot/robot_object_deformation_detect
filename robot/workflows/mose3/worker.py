"""MoSE3 video worker: load the model once, then claim and process videos.

Runs in the isolated ``mose3`` environment with the pinned upstream source on
``--mose3-root``. For each video: one feed-forward pass over every original
frame (query frame 0), tracks re-expressed in the first camera's frame exactly as
upstream ``inference.py`` does, then the PDI-initialized Link2/Link5 queries are
read from the dense fields and given their own SE(3) fit. Writes per-link point
arrays, the dense fields over each link's frame-zero mask (for CPU-only rigid fits),
rigidity diagnostics, a replay scene and a receipt; each video is claimed with an
exclusive lock file so no two workers process it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from robot.scoring.mose3 import rigidity
from robot.workflows.mose3.inputs import LINKS, sha
from robot.workflows.mose3.point_fields import (
    bilinear_taps, fit_query_se3, grid_to_source, mask_band, project, quat_to_matrix, sample,
    sample_per_frame, source_to_grid, touches_mask,
)

SETTINGS = dict(max_side=518, query_idx=0, num_ref_pts=12000, se3_temp=0.01, se3_sigma_px=0.0,
                visibility_logit_threshold=0.0, edge_rtol=0.05, normal_radius=12, replay_stride=4, seed=0,
                dense_band_px=4, dense_mask_resize='cv2.INTER_AREA, covered fraction >= 0.5')


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def complete(output, case):
    receipt = output / case / 'receipt.json'
    return receipt.is_file() and json.loads(receipt.read_text()).get('status') == 'complete'


def claim(claims, case, worker):
    try:
        descriptor = os.open(claims / case, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(dict(worker=worker, pid=os.getpid(), unix_time=time.time()), stream)
    return True


def track_link(tracks, emb, visibility_logits, poses, k, edge, frames0, xy_source, point_ids,
               source_hw, settings, helpers):
    """Every MoSE3 output for one link's queries; NumPy results."""
    count, height, width, _ = tracks.shape
    xy = source_to_grid(xy_source, source_hw, (height, width))
    index, weight = bilinear_taps(xy, (height, width))
    on_edge = touches_mask(edge, index, weight)
    index, weight = bilinear_taps(xy, (height, width), nearest=on_edge)
    trajectories = sample(tracks, index, weight).cpu().numpy()                 # [T, N, 3]
    logits = sample(visibility_logits[..., None], index, weight)[..., 0].cpu().numpy()
    query_emb = sample(emb[:1], index, weight)[0]
    se3 = fit_query_se3(tracks, emb, xy, query_emb, query_idx=settings['query_idx'],
                        num_ref_pts=settings['num_ref_pts'], temperature=settings['se3_temp'],
                        sigma_px=settings['se3_sigma_px'], seed=settings['seed'])
    quat, trans, se3_ok = (se3[name].cpu().numpy() for name in ('quat', 'trans', 'ok'))

    uv, depth = project(trajectories.astype(np.float64), poses, k)
    inside = (np.isfinite(uv).all(-1) & (depth > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1)
              & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1))
    taps, taps_weight = bilinear_taps(np.where(inside[..., None], uv, 0).reshape(-1, 2), (height, width))
    tracked_emb = sample_per_frame(emb, taps.reshape(count, -1, 4), taps_weight.reshape(count, -1, 4)).cpu().numpy()
    tracked_emb[~inside] = np.nan

    pixels = np.rint(np.clip(xy[:, ::-1], 0, [height - 1, width - 1])).astype(np.int64)   # (h, w)
    start, start_ok = helpers['surface_normal_frames'](frames0['tracks'], poses[0][:3, 3], pixels,
                                                      settings['normal_radius'], edge)
    visible = logits > settings['visibility_logit_threshold']
    rotations = quat_to_matrix(quat)
    origin = np.einsum('tnij,nj->tni', rotations, trajectories[0].astype(np.float64)) + trans
    arrays = dict(
        point_ids=point_ids, query_xy_source=xy_source.astype(np.float32), query_xy_grid=xy.astype(np.float32),
        on_depth_edge=on_edge, trajectories_world=trajectories.astype(np.float32),
        trajectories_uv_source=grid_to_source(uv, source_hw, (height, width)).astype(np.float32),
        in_frame=inside, visibility_logit=logits.astype(np.float32), visible=visible,
        rigidity_embedding_query=query_emb.cpu().numpy().astype(np.float32),
        rigidity_embedding_tracked=tracked_emb.astype(np.float32),
        se3_quat_wxyz=quat.astype(np.float32), se3_trans=trans.astype(np.float32), se3_valid=se3_ok,
        se3_origin=origin.astype(np.float32), display_frame=start.astype(np.float32), display_frame_ok=start_ok,
        pca_rgb=frames0['pca'][pixels[:, 0], pixels[:, 1]], trail_rgb=helpers['point_colors'](pixels, height, width))
    summary = rigidity.summarize(trajectories, visible, rotations, se3_ok, tracked_emb)
    # Points touching a depth edge (often the PDI sampler's high-gradient mask-boundary
    # picks) mix surfaces; report the interior subset too, without dropping any point.
    interior = ~on_edge
    subset = rigidity.summarize(trajectories[:, interior], visible[:, interior], rotations[:, interior],
                                se3_ok[:, interior], tracked_emb[:, interior])
    summary['interior'] = {key: subset[key] for key in ('scores', 'series', 'points')}
    summary['on_depth_edge'] = int(on_edge.sum())
    error = np.linalg.norm(arrays['trajectories_uv_source'][0] - xy_source, axis=1)
    summary['frame0_reprojection_px'] = dict(median=float(np.median(error)), p90=float(np.percentile(error, 90)),
                                             max=float(error.max()))
    return arrays, summary


def dense_link(tracks, emb, visibility_logits, edge, mask, band_px):
    """MoSE3's fields at every grid pixel of one link's frame-zero mask and a band around it.

    Cached so that rigid fits over the whole link run later on CPU; the band lets the
    mask be eroded or dilated without another forward pass.
    """
    import torch
    count, height, width, _ = tracks.shape
    index, fraction, signed = mask_band(mask, (height, width), band_px)
    rows = torch.as_tensor(index, device=tracks.device)
    return dict(
        pixels_hw=np.stack(np.divmod(index, width), axis=1).astype(np.int16),
        mask_fraction=fraction.astype(np.float16), mask_signed_distance_px=signed.astype(np.float16),
        on_depth_edge=np.asarray(edge).reshape(-1)[index],
        tracks=tracks.reshape(count, -1, 3).index_select(1, rows).cpu().numpy().astype(np.float32),
        visibility_logit=visibility_logits.reshape(count, -1).index_select(1, rows).cpu().numpy().astype(np.float32),
        rigidity_embedding_query=emb[0].reshape(height * width, -1).index_select(0, rows).cpu().numpy().astype(np.float32))


def process(case, entry, args, model, torch, helpers, settings):
    timings = {}
    started = time.perf_counter()
    video = args.sources / entry['video']
    if sha(video) != entry['source_video_sha256']:
        raise ValueError('Source video SHA-256 mismatch')
    query_receipt = json.loads((args.queries / f'{case}.json').read_text())
    if query_receipt['queries_sha256'] != sha(args.queries / f'{case}.npz'):
        raise ValueError('Query archive hash mismatch')
    with np.load(args.queries / f'{case}.npz', allow_pickle=False) as archive:
        queries = {name: archive[name] for name in archive.files}
    bundle = args.inputs / 'frame0_masks.npz'
    if sha(bundle) != json.loads((args.inputs / 'manifest.json').read_text())['frame0_masks_sha256']:
        raise ValueError('Frame-zero mask bundle hash mismatch')
    with np.load(bundle, allow_pickle=False) as archive:
        masks = {link: archive[f'{case}/{link}'] for link in LINKS}
    if any(mask.shape != tuple(entry['source_hw']) for mask in masks.values()):
        raise ValueError('Frame-zero mask resolution differs from the frozen manifest')
    count, source_hw = entry['frame_count'], tuple(entry['source_hw'])
    frames, picked, (width0, height0) = helpers['load_video_clip'](
        video, num_frames=count, start_frame=0, frame_interval=1, max_w=settings['max_side'],
        max_h=settings['max_side'], frame_indices=list(range(count)))
    if list(picked) != list(range(count)) or (height0, width0) != source_hw:
        raise ValueError('Decoded frames differ from the frozen manifest')
    height, width = frames.shape[1:3]
    timings['decode_seconds'] = time.perf_counter() - started

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    forward_started = time.perf_counter()
    with torch.inference_mode():
        images = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float().div_(255.0)[None]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(images, query_frame_idx=settings['query_idx'])
        del images
        tracks_local = out['tracks_local'][0].float()
        emb = out['rigidity_emb'][0].float()
        visibility_logits = out['visibility_logits'][0].float()
        camera_poses = out['camera_poses'][0].float()
        fx, fy = out['track_focal'][0].float().tolist()
        del out
        tracks, camera_poses = helpers['to_first_camera_frame'](tracks_local, camera_poses)
        del tracks_local
        torch.cuda.synchronize()
        timings['forward_seconds'] = time.perf_counter() - forward_started

        points_started = time.perf_counter()
        poses = camera_poses.cpu().numpy().astype(np.float64)
        k = np.array([[fx, 0, 0.5 * width - 0.5], [0, fy, 0.5 * height - 0.5], [0, 0, 1]], np.float64)
        tracks0 = tracks[0].cpu().numpy()
        query_view = dict(tracks=tracks0[None], camera_poses=poses[:1], query_idx=0)
        edge = helpers['depth_edges'](query_view, settings['edge_rtol'])
        frames0 = dict(tracks=tracks0, pca=helpers['pca_rgb'](emb[:1].cpu().numpy())[0])
        links, summaries = {}, {}
        for link in LINKS:
            links[link], summaries[link] = track_link(
                tracks, emb, visibility_logits, poses, k, edge, frames0,
                queries[f'{link}_xy_source'], queries[f'{link}_point_ids'], source_hw, settings, helpers)
        timings['points_seconds'] = time.perf_counter() - points_started
        dense = {link: dense_link(tracks, emb, visibility_logits, edge, masks[link], settings['dense_band_px'])
                 for link in LINKS}

        stride = settings['replay_stride']
        keep = ~edge[::stride, ::stride].reshape(-1)
        cloud = tracks[:, ::stride, ::stride].reshape(count, -1, 3)[:, torch.as_tensor(keep, device=tracks.device)]
        scene = dict(points=cloud.cpu().numpy().astype(np.float16),
                     rgb=frames[0][::stride, ::stride].reshape(-1, 3)[keep],
                     pca=frames0['pca'][::stride, ::stride].reshape(-1, 3)[keep],
                     camera_poses=poses.astype(np.float32), intrinsics=k.astype(np.float32),
                     grid_hw=np.asarray([height, width]), source_hw=np.asarray(source_hw),
                     median_depth=np.asarray(float(np.median(tracks0[..., 2]))))
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    del tracks, emb, visibility_logits

    save_started = time.perf_counter()
    staging = args.output / f'.{case}.partial-{os.getpid()}'
    staging.mkdir(parents=True)
    for link, arrays in links.items():
        np.savez_compressed(staging / f'{link}.npz', **arrays)
        np.savez_compressed(staging / f'dense_{link}.npz', **dense[link])
    np.savez_compressed(staging / 'scene.npz', **scene)
    write_json(staging / 'rigidity.json', dict(video_id=case, links=summaries,
               masks={link: entry['masks'][link]['review'] for link in LINKS}))
    timings['save_seconds'] = time.perf_counter() - save_started
    timings['total_seconds'] = time.perf_counter() - started
    import cv2
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    receipt = dict(
        status='complete', video_id=case, source_video_sha256=entry['source_video_sha256'],
        queries_sha256=query_receipt['queries_sha256'], frame_count=count, frame_indices='0..T-1 (every frame)',
        fps=fps, source_hw=list(source_hw), grid_hw=[height, width], intrinsics=k.tolist(),
        coordinates='first camera: x right, y down, z forward; MoSE3 (pi3) scale',
        settings=settings, checkpoint=args.checkpoint_identity, worker=args.worker_id,
        files={p.name: sha(p) for p in sorted(staging.iterdir())}, timings=timings,
        process_peak_gpu_allocated_bytes=peak_allocated, process_peak_gpu_reserved_bytes=peak_reserved,
        points={link: summaries[link]['points'] for link in LINKS},
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         numpy=np.__version__, gpu=torch.cuda.get_device_name(0), hostname=platform.node(),
                         slurm_job_id=os.getenv('SLURM_JOB_ID')))
    write_json(staging / 'receipt.json', receipt)
    destination = args.output / case
    if destination.exists():
        raise FileExistsError(f'Completed output already exists: {destination}')
    staging.replace(destination)
    return receipt


def serve(process, description, extra=()):
    """Load MoSE3 once, then claim and ``process`` videos until none remain or a stop file appears."""
    parser = argparse.ArgumentParser(description=description)
    for name in ('inputs', 'queries', 'sources', 'output', 'checkpoint', 'mose3-root', 'claims', 'status', 'stop-file',
                 'checkpoint-identity', *extra):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker-id', required=True)
    parser.add_argument('--order', required=True, help='Comma-separated video IDs in processing order')
    args = parser.parse_args()
    args.checkpoint_identity = json.loads(args.checkpoint_identity.read_text())
    sys.path.insert(0, str(args.mose3_root))
    import torch
    from inference import _to_first_camera_frame
    from mose3.models.mose3 import MoSE3
    from mose3.utils.video_io import load_video_clip
    from visualize import depth_edges, pca_rgb, point_colors, surface_normal_frames

    helpers = dict(to_first_camera_frame=_to_first_camera_frame, load_video_clip=load_video_clip,
                   depth_edges=depth_edges, pca_rgb=pca_rgb, point_colors=point_colors,
                   surface_normal_frames=surface_normal_frames)
    entries = {e['video_id']: e for e in json.loads((args.inputs / 'manifest.json').read_text())['entries']}
    loaded = time.perf_counter()
    model = MoSE3.from_pretrained(str(args.checkpoint), strict=True).cuda().eval()
    load_seconds = time.perf_counter() - loaded
    stop_after = Path(str(args.stop_file) + f'.{args.worker_id}')

    def report(**row):
        with args.status.open('a') as stream:
            stream.write(json.dumps(dict(worker=args.worker_id, pid=os.getpid(), unix_time=time.time(), **row)) + '\n')

    report(event='ready', model_load_seconds=load_seconds)
    for case in args.order.split(','):
        if args.stop_file.exists() or stop_after.exists():
            report(event='stopped_before_claim')
            break
        if complete(args.output, case) or not claim(args.claims, case, args.worker_id):
            continue
        report(event='started', video_id=case, frames=entries[case]['frame_count'])
        try:
            receipt = process(case, entries[case], args, model, torch, helpers, dict(SETTINGS))
            report(event='complete', video_id=case, frames=receipt['frame_count'],
                   seconds=receipt['timings']['total_seconds'],
                   peak_reserved_bytes=receipt['process_peak_gpu_reserved_bytes'])
        except Exception as error:
            text = traceback.format_exc()
            out_of_memory = isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in text.lower()
            report(event='failed', video_id=case, out_of_memory=out_of_memory, error=text[-4000:])
            for partial in args.output.glob(f'.{case}.partial-{os.getpid()}'):
                partial.rename(partial.with_name(partial.name.replace('.partial-', '.failed-')))
        finally:
            torch.cuda.empty_cache()
    report(event='exit')


if __name__ == '__main__':
    serve(process, __doc__)
