"""Freeze the 45-video Link2/Link5 frame-zero masks used for query initialization.

CPU only; reads retained mask archives and never calls SAM, a VLM, or a tracker.
Link2 comes from the four-way selected-45 six-link base masks. Link5 comes from the
latest guarded mask cache. Both are checked against the same source-video SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robot.experiments.link5_pair_selection_ablation.prepare_manifest import EXCLUDED as LINK5_MASK_EXCLUSIONS

REPO = Path(__file__).resolve().parents[3]
LINK2_RUN = REPO / 'robot/experiments/link7_tracker_filter_comparison/results/link5-link7-four-way-selected45-20260927'
LINK5_CACHE = REPO / 'robot/experiments/link5_shape_codebook/results/round_four_frame0'
LINKS = ('link2', 'link5')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_video(case):
    """Video path relative to the ERIS inputs/videos root."""
    generator, number = case.rsplit('_', 1)
    return f"{'LVP' if generator == 'LVP_ROBOWM' else generator}/{number}.mp4"


def frame0_channel(path, name):
    with np.load(path, allow_pickle=False) as archive:
        names = archive['object_names'].tolist()
        masks = archive['object_masks']
        return masks[0, names.index(name)].astype(bool), masks.shape[0]


def build(link2_run=LINK2_RUN, link5_cache=LINK5_CACHE):
    link5_manifest_path = link5_cache / 'mask_cache_manifest.json'
    link5_entries = {e['video_id']: e for e in json.loads(link5_manifest_path.read_text())['entries']}
    cases = sorted(p.name for p in (link2_run / 'cases').iterdir() if p.is_dir())
    if len(cases) != 45 or not set(cases) <= set(link5_entries):
        raise ValueError('Expected the selected 45 videos in both mask sources')
    entries, masks = [], {}
    for case in cases:
        link5 = link5_entries[case]
        link2_path = link2_run / 'cases' / case / 'base_segmentation.npz'
        link2_source = json.loads((link2_path.parent / 'base_segmentation_source.json').read_text())
        link5_path = link5_cache / link5['mask']
        if link2_source['source_video_sha256'] != link5['source_video_sha256']:
            raise ValueError(f'Link2 and Link5 masks were made from different videos: {case}')
        if link2_source['base_segmentation_sha256'] != sha(link2_path) or link5['mask_sha256'] != sha(link5_path):
            raise ValueError(f'Mask archive hash mismatch: {case}')
        link2_mask, link2_frames = frame0_channel(link2_path, 'link2')
        link5_mask, link5_frames = frame0_channel(link5_path, 'link5')
        if link2_frames != link5_frames or link2_frames != link5['frame_count'] or link2_mask.shape != link5_mask.shape:
            raise ValueError(f'Mask frame count or resolution mismatch: {case}')
        for link, mask in (('link2', link2_mask), ('link5', link5_mask)):
            if mask.sum() < 64:
                raise ValueError(f'{case} {link} frame-zero mask is nearly empty')
            masks[f'{case}/{link}'] = mask
        entries.append(dict(
            video_id=case, video=source_video(case), source_video_sha256=link5['source_video_sha256'],
            frame_count=int(link5['frame_count']), source_hw=list(link2_mask.shape),
            masks={
                'link2': dict(archive=str(link2_path.relative_to(REPO)), archive_sha256=sha(link2_path),
                              tracked_fraction=link2_source['tracked_fraction']['link2'],
                              frame0_pixels=int(link2_mask.sum()), review='four_way_base_mask'),
                'link5': dict(archive=str(link5_path.relative_to(REPO)), archive_sha256=link5['mask_sha256'],
                              frame0_pixels=int(link5_mask.sum()),
                              review=LINK5_MASK_EXCLUSIONS.get(case, 'latest_guarded_mask')),
            }))
    manifest = dict(
        schema_version=1, links=list(LINKS), cases=len(entries),
        link2_source=str(link2_run.relative_to(REPO)), link5_source=str(link5_manifest_path.relative_to(REPO)),
        link5_source_manifest_sha256=sha(link5_manifest_path),
        note='Frame-zero masks only. Link5 review labels are carried, not used to drop cases.',
        entries=entries)
    return manifest, masks


def write(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, masks = build()
    bundle = output / 'frame0_masks.npz'
    np.savez_compressed(bundle, **masks)
    manifest['frame0_masks_sha256'] = sha(bundle)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), cases=manifest['cases'], masks=len(masks))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    write(parser.parse_args().output)
