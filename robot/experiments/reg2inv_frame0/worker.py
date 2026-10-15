"""Official Reg2Inv with frame-0 clouds substituted for CAD normal prototypes."""
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from common import ROOT, RESULTS, SOURCE_ROOT, OBSERVATIONS, manifest, sha, write, code_sha, CHECKPOINT
from link2_islands import filter_islands

sys.path.append(str(SOURCE_ROOT))
import torch
torch.set_num_threads(2)
from paired_scoring import PairedBank, InvalidCloud
from reg2invbaselinecode.reg2inv import seed_all
import gpu_ransac
from execution_cache import install_geometry_cache, install_backbone_cache


def configure():
    import faiss
    import geotransformer.modules.geotransformer.local_global_registration as registration
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)
    def timed_registration(*args, **kwargs):
        started = time.monotonic()
        result = gpu_ransac.register(*args, **kwargs)
        torch.cuda.synchronize()
        if gpu_ransac.register.last is not None:
            gpu_ransac.register.last['seconds'] = time.monotonic() - started
        return result
    registration.registration_with_ransac_from_correspondences = timed_registration
    gpu_ransac.extension()


def predict(bank, points, verify=False):
    started = time.monotonic()
    shared = bank.extract(points, retain_output=verify)
    torch.cuda.synchronize()
    extracted = time.monotonic()
    result = bank.score(shared)
    torch.cuda.synchronize()
    timing = dict(feature_registration_seconds=extracted-started, score_seconds=time.monotonic()-extracted,
                  gpu_ransac_seconds=(gpu_ransac.register.last or {}).get('seconds', 0.))
    if verify and result['score_status'] == 'ok':
        original = bank.forward
        def frozen_forward(query):
            np.testing.assert_array_equal(query, shared['query'])
            return shared['model_output']
        bank.forward = frozen_forward
        try:
            native = bank.predict(points)
        finally:
            bank.forward = original
        np.testing.assert_allclose(native['link_score'], result['link_score'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(native['point_scores'], result['point_scores'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(native['transform_camera_to_cad_m'], result['transform_camera_to_cad_m'], rtol=1e-5, atol=1e-7)
    return result, timing


def run_case(case, tag=None, frames=None, verify=False):
    entry = next(e for e in manifest() if e['video_id'] == case)
    folder = RESULTS / ('pilot/' + tag if tag else 'cases') / case
    folder.mkdir(parents=True, exist_ok=True)
    manifest_sha, digest = sha(RESULTS / 'inputs/manifest.json'), code_sha()
    restore = install_geometry_cache()
    banks = {}
    started = time.monotonic()
    seed_all(7351)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    try:
        (RESULTS / 'banks' / case).mkdir(parents=True, exist_ok=True)
        for link in ['link2', 'link5']:
            normal_seed = int(hashlib.sha256(f'{case}:reference:0:{link}'.encode()).hexdigest()[:8], 16) % (2**31-1)
            seed_all(normal_seed)
            torch.set_num_threads(2)
            ref = entry['references'][link]
            path = RESULTS / 'inputs' / ref['path']
            assert sha(path) == ref['sha256']
            banks[link] = PairedBank(link, path, RESULTS / 'banks' / case,
                                    CHECKPOINT)
            assert banks[link].cfg.ransac.num_iterations == 50000 and banks[link].cfg.ransac.num_points == 3
            install_backbone_cache(banks[link])
        initialization_seconds = time.monotonic() - started
        print('REFERENCE_BANKS_READY', case, initialization_seconds, flush=True)
        count = frames if frames is not None else entry['frame_count']
        timings, available = [], {link: 0 for link in banks}
        scoring_started = time.monotonic()
        for index in range(count):
            for link, bank in banks.items():
                name = f'{index:05d}_{link}.npz'
                source = OBSERVATIONS / case / name
                source_sha = entry['observations'][name]
                assert sha(source) == source_sha
                target = folder / name
                if target.exists():
                    with np.load(target, allow_pickle=False) as saved:
                        assert str(saved['manifest_sha256']) == manifest_sha and str(saved['code_sha256']) == digest
                        assert str(saved['source_archive_sha256']) == source_sha
                        available[link] += int(str(saved['score_status']) == 'ok')
                    continue
                with np.load(source, allow_pickle=False) as old:
                    assert int(old['frame_index']) == index and str(old['link']) == link
                    points, pixels = old['points_camera_m'], old['pixels_yx']
                    intrinsics = old['intrinsics']
                    depth_info = str(old['depth_filter_info_json'])
                keep, island = filter_islands(points, pixels, link)
                removed_pixels = pixels[~keep]
                points, pixels = points[keep], pixels[keep]
                seed = int(hashlib.sha256(f'{case}:{index}:{link}'.encode()).hexdigest()[:8], 16) % (2**31 - 1)
                seed_all(seed)
                torch.set_num_threads(2)
                t = time.monotonic()
                gpu_ransac.register.last = None
                try:
                    result, timing = predict(bank, points, verify=verify and index == 0)
                except InvalidCloud as error:
                    result = dict(link_score=np.nan, point_scores=np.full(len(points), np.nan, np.float32),
                        score_status=str(error), registration_status='unavailable', registration_fitness=np.nan,
                        registration_median_error_m=np.nan, transform_camera_to_cad_m=np.full((4, 4), np.nan),
                        transform_normalized=np.full((4, 4), np.nan), voxel_points=0, selected_feature_points=0)
                    timing = {}
                torch.cuda.synchronize()
                elapsed = time.monotonic() - t
                timing['registration_and_score_seconds'] = elapsed
                payload = dict(points_camera_m=points, pixels_yx=pixels, point_scores=result['point_scores'],
                    intrinsics=intrinsics, frame_index=index, link=link, method='reg2inv_frame0', seed=seed,
                    link_score=result['link_score'], score_status=result['score_status'],
                    registration_status=result['registration_status'], registration_fitness=result['registration_fitness'],
                    registration_median_error_m=result['registration_median_error_m'],
                    transform_camera_to_reference_m=result['transform_camera_to_cad_m'],
                    transform_normalized=result['transform_normalized'], voxel_points=result['voxel_points'],
                    selected_feature_points=result['selected_feature_points'], reference_frame_index=0,
                    reference_sha256=entry['references'][link]['sha256'], reference_radius_m=bank.radius_m,
                    reference_bank_sha256=sha(RESULTS / 'banks' / case / (link + '.npz')),
                    source_archive_sha256=source_sha, manifest_sha256=manifest_sha, code_sha256=digest,
                    depth_filter_info_json=depth_info, island_filter_info_json=json.dumps(island),
                    island_removed_pixels_yx=removed_pixels, timings_json=json.dumps(timing),
                    execution_backend='gpu_ransac_float64_frozen_frame0_cache',
                    registration_execution_json=json.dumps(gpu_ransac.register.last))
                temp = target.with_suffix('.partial.npz')
                np.savez_compressed(temp, **payload)
                temp.replace(target)
                available[link] += int(result['score_status'] == 'ok')
                timings.append(elapsed)
                print('FRAME_DONE', case, index, link, result['score_status'], result['link_score'], flush=True)
        report = dict(status='complete', video_id=case, frame_count=count, method='reg2inv_frame0',
            scored_frames=available, seconds=time.monotonic()-started,
            initialization_seconds=initialization_seconds, scoring_seconds=time.monotonic()-scoring_started,
            new_frame_links=len(timings), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), manifest_sha256=manifest_sha, code_sha256=digest)
        write(folder / 'complete.json', report)
        print('VIDEO_COMPLETE', case, report['seconds'], flush=True)
        return report
    finally:
        restore()
        banks.clear()
        bank = None
        gpu_ransac._reference.clear()
        gc.collect()
        torch.cuda.empty_cache()
