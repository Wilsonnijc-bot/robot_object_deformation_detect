"""Display-only MoSE3 pass for the replay: upstream's per-frame rigidity-embedding video.

The tracking run kept the rigidity embedding only at its query points, not the dense
[T, H, W, 16] field. This pass repeats the tracking forward pass (same frames, 518 px
grid, query frame 0, bf16) and renders the field with upstream ``pca_rgb`` over the
whole clip, as upstream ``visualize.py`` writes ``rigidity.mp4``. It re-samples the
saved queries and records how closely this pass reproduces the saved trajectories.
Tracking outputs are read, never rewritten.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import imageio.v2 as imageio
import numpy as np

from robot.workflows.mose3.inputs import LINKS, sha
from robot.workflows.mose3.worker import serve, write_json
from robot.workflows.mose3.point_fields import bilinear_taps, sample


def process(case, entry, args, model, torch, helpers, settings):
    started = time.perf_counter()
    video = args.sources / entry['video']
    if sha(video) != entry['source_video_sha256']:
        raise ValueError('Source video SHA-256 mismatch')
    tracked = json.loads((args.tracking / case / 'receipt.json').read_text())
    if tracked['status'] != 'complete' or tracked['source_video_sha256'] != entry['source_video_sha256']:
        raise ValueError('No completed tracking output for this video')
    count, source_hw = entry['frame_count'], tuple(entry['source_hw'])
    frames, picked, (width0, height0) = helpers['load_video_clip'](
        video, num_frames=count, start_frame=0, frame_interval=1, max_w=settings['max_side'],
        max_h=settings['max_side'], frame_indices=list(range(count)))
    if list(picked) != list(range(count)) or (height0, width0) != source_hw:
        raise ValueError('Decoded frames differ from the frozen manifest')
    height, width = frames.shape[1:3]
    if [height, width] != tracked['grid_hw']:
        raise ValueError('MoSE3 grid differs from the tracking run')

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        images = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float().div_(255.0)[None]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(images, query_frame_idx=settings['query_idx'])
        del images
        tracks_local = out['tracks_local'][0].float()
        emb = out['rigidity_emb'][0].float()
        camera_poses = out['camera_poses'][0].float()
        del out
        tracks, _ = helpers['to_first_camera_frame'](tracks_local, camera_poses)
        del tracks_local
        depth = float(np.median(tracks[0][..., 2].cpu().numpy()))
        reproduction = {}
        for link in LINKS:
            with np.load(args.tracking / case / f'{link}.npz', allow_pickle=False) as saved:
                index, weight = bilinear_taps(saved['query_xy_grid'], (height, width), nearest=saved['on_depth_edge'])
                again = sample(tracks, index, weight).cpu().numpy()
                delta = np.linalg.norm(again - saved['trajectories_world'], axis=-1) / depth
            reproduction[link] = dict(median=float(np.median(delta)), p99=float(np.percentile(delta, 99)),
                                      max=float(delta.max()))
        pca = helpers['pca_rgb'](emb.cpu().numpy())
        del tracks, emb
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    staging = args.output / f'.{case}.partial-{os.getpid()}'
    staging.mkdir(parents=True)
    imageio.mimwrite(staging / 'rigidity_pca.mp4', list(pca), fps=fps, quality=9, macro_block_size=1)
    receipt = dict(
        status='complete', display_only=True, video_id=case, source_video_sha256=entry['source_video_sha256'],
        frame_count=count, fps=fps, grid_hw=[height, width], view='upstream pca_rgb over every frame of the clip',
        reproduction_relative_to_median_depth=reproduction, checkpoint=args.checkpoint_identity,
        files={p.name: sha(p) for p in sorted(staging.iterdir())},
        timings=dict(total_seconds=time.perf_counter() - started),
        process_peak_gpu_allocated_bytes=peak_allocated, process_peak_gpu_reserved_bytes=peak_reserved,
        environment=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                         slurm_job_id=os.getenv('SLURM_JOB_ID')))
    write_json(staging / 'receipt.json', receipt)
    destination = args.output / case
    if destination.exists():
        raise FileExistsError(f'Completed output already exists: {destination}')
    staging.replace(destination)
    return receipt


if __name__ == '__main__':
    serve(process, __doc__, extra=('tracking',))
