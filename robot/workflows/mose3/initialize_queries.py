"""Initialize Link2/Link5 frame-zero queries with PDI's unchanged region sampler.

Run in the geometry (PDI/CoTracker) environment; CPU only. This calls the
production ``TrackWrapper.prepare_multi`` on a model-free instance: decode at most
880 px, resize each frame-zero mask with nearest neighbor, gather SIFT, Shi-Tomasi
and grid candidates inside it, deduplicate, then farthest-point balance. Counts
come from ``robot/configs/default.yaml`` (Link2 256; Link5 grid_size**2 = 100).
Queries are mapped back to source pixels with the same ``scale_xy`` as CoTracker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import platform

import cv2
import numpy as np
import yaml

from robot.workflows.mose3.inputs import LINKS, REPO, sha

CONFIG = REPO / 'robot/configs/default.yaml'
SAMPLER_SOURCES = ('infrastructure/shared/inference/tracking.py',
                   'infrastructure/shared/inference/cotracker_core.py')
COLORS = {'link2': (255, 170, 0), 'link5': (0, 200, 255)}


def tracking_config():
    config = yaml.safe_load(CONFIG.read_text())['multi_object_tracking']
    counts = {name: int(count) for name, count in config.get('object_query_counts', {}).items() if name in LINKS}
    return dict(grid_size=int(config['grid_size']), bg_grid_size=int(config['background_grid_size']),
                background_dilation=int(config['background_dilation']), max_dim=int(config['max_dimension']),
                object_query_counts=counts)


def preview(path, video, masks, queries):
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f'Cannot decode preview frame: {video}')
    for link, mask in zip(LINKS, masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = COLORS[link][::-1]
        cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)
        for x, y in queries[link]:
            cv2.circle(frame, (int(round(x)), int(round(y))), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), frame)


def initialize(case, entry, masks, sources, output, config):
    import torch
    from infrastructure.shared.inference.tracking import TrackWrapper

    cv2.setNumThreads(1)
    video = sources / entry['video']
    if sha(video) != entry['source_video_sha256']:
        raise ValueError(f'Source video SHA-256 mismatch: {case}')
    # The production sampler only reads ``device``; no CoTracker weights are loaded.
    wrapper = TrackWrapper.__new__(TrackWrapper)
    wrapper.device = torch.device('cpu')
    stacked = np.stack([masks[f'{case}/{link}'] for link in LINKS])
    prepared = wrapper.prepare_multi(str(video), stacked, LINKS, **config)
    if prepared.frames_count != entry['frame_count'] or list(prepared.original_hw) != entry['source_hw']:
        raise ValueError(f'Decoded frames or resolution differ from the mask archives: {case}')
    arrays, counts, queries = {}, {}, {}
    for index, link in enumerate(LINKS):
        tracker = prepared.object_queries[index].astype(np.float32)
        if not np.all(tracker[:, 0] == 0):
            raise ValueError('PDI sampler returned a non-zero query frame')
        source = tracker[:, 1:] * np.asarray(prepared.scale_xy, np.float32)
        arrays[f'{link}_xy_tracker'] = tracker[:, 1:]
        arrays[f'{link}_xy_source'] = source
        arrays[f'{link}_point_ids'] = np.arange(len(source), dtype=np.int64)
        counts[link] = dict(requested=int(prepared.requested_object_query_counts[index]), returned=len(source))
        queries[link] = source
    destination = output / f'{case}.npz'
    np.savez_compressed(destination, tracker_hw=np.asarray(prepared.tracker_hw), scale_xy=np.asarray(prepared.scale_xy),
                        source_hw=np.asarray(prepared.original_hw), **arrays)
    preview(output / f'{case}.png', video, stacked, queries)
    receipt = dict(status='complete', video_id=case, source_video_sha256=entry['source_video_sha256'],
                   queries_sha256=sha(destination), tracker_hw=list(prepared.tracker_hw),
                   scale_xy=list(prepared.scale_xy), counts=counts, query_time=0)
    (output / f'{case}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='Folder with manifest.json and frame0_masks.npz')
    parser.add_argument('--sources', type=Path, required=True, help='ERIS inputs/videos root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    bundle = args.inputs / 'frame0_masks.npz'
    if sha(bundle) != manifest['frame0_masks_sha256']:
        raise ValueError('Frame-zero mask bundle does not match its manifest')
    with np.load(bundle, allow_pickle=False) as archive:
        masks = {name: archive[name] for name in archive.files}
    args.output.mkdir(parents=True, exist_ok=False)
    config = tracking_config()
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {e['video_id']: pool.submit(initialize, e['video_id'], e, masks, args.sources, args.output, config)
                   for e in manifest['entries']}
        receipts = {case: future.result() for case, future in futures.items()}
    summary = dict(status='complete', cases=len(receipts), config=config,
                   config_sha256=sha(CONFIG), inputs_manifest_sha256=sha(args.inputs / 'manifest.json'),
                   sampler='TrackWrapper.prepare_multi -> CoTrackerCore._sample_region_queries (unchanged)',
                   sampler_sources={name: sha(REPO / name) for name in SAMPLER_SOURCES},
                   environment=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                                    hostname=platform.node(), slurm_job_id=os.getenv('SLURM_JOB_ID')),
                   counts={case: r['counts'] for case, r in receipts.items()})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('QUERY_INITIALIZATION_COMPLETE', len(receipts), flush=True)


if __name__ == '__main__':
    main()
