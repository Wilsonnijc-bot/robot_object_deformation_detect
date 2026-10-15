"""Fail closed on source/reference drift; validate the actual allocation before inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, RESULTS, SOURCE_ROOT, OBSERVATIONS, manifest, sha, write, code_sha, CHECKPOINT
from link2_islands import filter_islands, settings


def inputs():
    from freeze_inputs import CORE_SOURCES
    raw = json.loads((RESULTS / 'inputs/manifest.json').read_text())
    assert raw['cad_used'] is False and raw['foundationpose'] is False
    assert raw['later_frames_in_normal_bank'] is False and raw['observation_filter'] == settings()
    store_path = OBSERVATIONS / 'manifest.json'
    assert sha(store_path) == raw['observation_store_sha256']
    store = json.loads(store_path.read_text())['entries']
    assert [e['video_id'] for e in store] == [e['video_id'] for e in raw['entries']] and len(store) == raw['cases']
    assert sha(CHECKPOINT) == raw['checkpoint_sha256']
    assert sorted(raw['core_sources']) == sorted(CORE_SOURCES)
    for source, digest in raw['core_sources'].items():
        assert sha(SOURCE_ROOT / source) == digest
    observations = references = 0
    for entry, stored in zip(manifest(), store):
        case = entry['video_id']
        assert entry['observations'] == stored['observations'] and len(entry['observations']) == 2 * entry['frame_count']
        for name, digest in entry['observations'].items():
            assert sha(OBSERVATIONS / case / name) == digest
            observations += 1
        for link, ref in entry['references'].items():
            path = RESULTS / 'inputs' / ref['path']
            assert sha(path) == ref['sha256']
            with np.load(path, allow_pickle=False) as frozen, np.load(
                    OBSERVATIONS / case / f'00000_{link}.npz', allow_pickle=False) as old:
                assert int(frozen['reference_frame_index']) == 0
                keep, _ = filter_islands(old['points_camera_m'], old['pixels_yx'], link)
                np.testing.assert_array_equal(frozen['points_m'], old['points_camera_m'][keep])
                np.testing.assert_array_equal(frozen['pixels_yx'], old['pixels_yx'][keep])
                assert len(frozen['points_m']) == ref['points'] and int((~keep).sum()) == ref['island_removed_points']
            references += 1
        print('INPUT_VERIFIED', case, flush=True)
    receipt = dict(status='passed', cases=len(store), normal_references=references, observations=observations,
        observation_filter=settings(), manifest_sha256=sha(RESULTS / 'inputs/manifest.json'), code_sha256=code_sha(),
        later_frames_in_normal_bank=False, cad_used=False, foundationpose=False)
    write(RESULTS / 'metadata/input_validation.json', receipt)
    return receipt


def gpu():
    import os
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work must be scheduled'
    receipt = inputs()
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert 'H200' in props.name
    assert float((torch.ones(16, device='cuda') * 2).sum()) == 32.
    from worker import configure, run_case
    configure()
    entries = manifest()
    reports = [run_case(entries[i]['video_id'], tag='preflight', frames=2, verify=True) for i in [0, 15]]
    for report in reports:
        assert all(n == 2 for n in report['scored_frames'].values()), 'Preflight scores unavailable'
    receipt.update(status='passed', job_id=os.environ['SLURM_JOB_ID'], gpu_uuid=str(props.uuid),
        gpu_name=props.name, gpu_total_bytes=props.total_memory, node=os.uname().nodename,
        scorer_equivalence_checks=4, reports=reports)
    write(RESULTS / 'metadata/preflight.json', receipt)
    print('PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    inputs() if args.inputs_only else gpu()
