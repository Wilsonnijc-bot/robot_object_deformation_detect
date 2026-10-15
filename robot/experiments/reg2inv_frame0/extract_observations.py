"""Copy the 45-video depth-filtered observations into a frame0-owned store (CPU).

The observed clouds were saved inside the CAD-reference run's case archives next
to that run's scores. Only the method-independent observation arrays are copied:
camera XYZ, depth-grid pixels, intrinsics, frame/link identity and the depth-filter
receipt. Every copied array is re-read and compared exactly with its source before
the store manifest is written. Nothing from the CAD reference, FoundationPose or
the CAD scores is retained.
"""
import json
from multiprocessing import Pool
import numpy as np
from common import ROOT, SOURCE_ROOT, sha, write

STORE = ROOT / 'observations'
FIELDS = ('points_camera_m', 'pixels_yx', 'intrinsics', 'frame_index', 'link', 'depth_filter_info_json')
VIDEO_FIELDS = ('video_id', 'video', 'source_video_sha256', 'frame_count', 'source_hw', 'mask', 'mask_sha256',
                'link2_source', 'link2_source_sha256', 'link5_source', 'link5_source_sha256',
                'quality_warning', 'link2_tracked_fraction')


SOURCE = SOURCE_ROOT / 'results/20261012'


def extract(entry):
    case = entry['video_id']
    folder = STORE / case
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for index in range(entry['frame_count']):
        for link in ('link2', 'link5'):
            name = f'{index:05d}_{link}.npz'
            with np.load(SOURCE / 'cases' / case / name, allow_pickle=False) as old:
                arrays = {k: old[k] for k in FIELDS}
            assert int(arrays['frame_index']) == index and str(arrays['link']) == link
            target = folder / name
            np.savez_compressed(target.with_suffix('.partial.npz'), **arrays)
            target.with_suffix('.partial.npz').replace(target)
            with np.load(target, allow_pickle=False) as copied:
                assert sorted(copied.files) == sorted(FIELDS)
                for k in FIELDS:
                    np.testing.assert_array_equal(copied[k], arrays[k])
                    assert copied[k].dtype == arrays[k].dtype
            files[name] = sha(target)
    print('EXTRACTED', case, entry['frame_count'], flush=True)
    return dict({k: entry[k] for k in VIDEO_FIELDS}, observations=files)


def main():
    entries = json.loads((SOURCE / 'inputs/manifest.json').read_text())['entries']
    assert len(entries) == 45 and not (STORE / 'manifest.json').exists()
    with Pool(8) as pool:
        videos = pool.map(extract, entries, chunksize=1)
    assert [v['video_id'] for v in videos] == [e['video_id'] for e in entries]
    write(STORE / 'manifest.json', dict(schema=1, videos=len(videos), links=['link2', 'link5'], fields=list(FIELDS),
        depth_filter='shared Link5 ray-normalized 3D density filter, as recorded per observation',
        content='method-independent observed clouds only; no CAD, FoundationPose or score arrays',
        entries=videos))
    print('STORE_COMPLETE', len(videos), sum(len(v['observations']) for v in videos), flush=True)


if __name__ == '__main__':
    main()
