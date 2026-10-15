"""Freeze all 45 videos and frame-0-only normal clouds from the observation store.

Link2 references pass through the same detached-island filter as every Link2 query.
"""
import json
import numpy as np
from common import RESULTS, SOURCE_ROOT, OBSERVATIONS, sha, write, CHECKPOINT
from link2_islands import filter_islands, settings

CORE_SOURCES = ['paired_scoring.py', 'reg2invbaselinecode/reg2inv.py', 'reg2invbaselinecode/geometry.py',
                'gpu_ransac.py', 'execution_cache.py']


def freeze():
    store_path = OBSERVATIONS / 'manifest.json'
    store = json.loads(store_path.read_text())
    destination = RESULTS / 'inputs/manifest.json'
    assert not destination.exists(), 'Frozen manifest already exists; never replace a live normal reference'
    entries = []
    for original in store['entries']:
        case = original['video_id']
        entry = dict(original, observations=dict(original['observations']), references={})
        for link in ['link2', 'link5']:
            name = f'00000_{link}.npz'
            path = OBSERVATIONS / case / name
            assert sha(path) == original['observations'][name]
            relative = f'frame0/{case}_{link}.npz'
            normal = RESULTS / 'inputs' / relative
            normal.parent.mkdir(parents=True, exist_ok=True)
            with np.load(path, allow_pickle=False) as a:
                assert int(a['frame_index']) == 0 and str(a['link']) == link
                points, pixels = a['points_camera_m'], a['pixels_yx']
                keep, island = filter_islands(points, pixels, link)
                assert points.shape[1:] == (3,) and np.isfinite(points).all() and keep.sum() >= 256
                np.savez_compressed(normal, points_m=points[keep], pixels_yx=pixels[keep],
                    intrinsics=a['intrinsics'], source_observation_sha256=sha(path), reference_frame_index=0,
                    island_filter_info_json=json.dumps(island), island_removed_pixels_yx=pixels[~keep])
            entry['references'][link] = dict(path=relative, sha256=sha(normal), source_observation_sha256=sha(path),
                points=int(keep.sum()), source_points=len(points), island_removed_points=int((~keep).sum()))
        entries.append(entry)
        print('FROZEN', case, entry['frame_count'], flush=True)
    write(destination, dict(schema=2, cases=len(entries), selection='all selected videos, observation-store order',
        observation_store_sha256=sha(store_path),
        checkpoint_sha256=sha(CHECKPOINT),
        core_sources={p: sha(SOURCE_ROOT / p) for p in CORE_SOURCES},
        methods=['reg2inv_frame0'], links=['link2', 'link5'], all_frames=True,
        reference='per video and per link, frozen refined frame-0 cloud only',
        mesh=False, cad_used=False, foundationpose=False, later_frames_in_normal_bank=False,
        normalization='official reference-derived radius and feature norms, fixed for all query frames',
        observation_filter=settings(), entries=entries))


if __name__ == '__main__':
    freeze()
