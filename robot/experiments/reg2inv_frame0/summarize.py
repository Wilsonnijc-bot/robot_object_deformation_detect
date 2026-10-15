"""Require all requested outputs; preserve score availability and reference identity."""
import csv
import json
from collections import Counter
import numpy as np
from common import RESULTS, manifest, sha, code_sha
from link2_islands import settings


def summarize():
    rows, frames = [], []
    manifest_sha, digest = sha(RESULTS / 'inputs/manifest.json'), code_sha()
    for entry in manifest():
        folder = RESULTS / 'cases' / entry['video_id']
        complete = json.loads((folder / 'complete.json').read_text())
        assert complete['status'] == 'complete' and complete['frame_count'] == entry['frame_count']
        assert len(list(folder.glob('*.npz'))) == entry['frame_count'] * 2
        for link in ['link2', 'link5']:
            values, statuses, poor, frame0_score = [], Counter(), 0, None
            for index in range(entry['frame_count']):
                with np.load(folder / f'{index:05d}_{link}.npz', allow_pickle=False) as a:
                    assert str(a['manifest_sha256']) == manifest_sha and str(a['code_sha256']) == digest
                    assert str(a['reference_sha256']) == entry['references'][link]['sha256']
                    assert int(a['reference_frame_index']) == 0 and str(a['method']) == 'reg2inv_frame0'
                    assert a['point_scores'].shape == (len(a['points_camera_m']),)
                    assert np.isfinite(a['points_camera_m']).all()
                    score, status = float(a['link_score']), str(a['score_status'])
                    statuses[status] += 1
                    if status == 'ok':
                        assert np.isfinite(score) and np.isfinite(a['point_scores']).all()
                        values.append(score)
                        poor += int(str(a['registration_status']) == 'poor_alignment')
                        if index == 0:
                            frame0_score = score
                    else:
                        assert np.isnan(score)
                    frames.append(dict(video_id=entry['video_id'], link=link, method='reg2inv_frame0',
                        frame_index=index, score=score if status == 'ok' else '', status=status,
                        island_removed_points=json.loads(str(a['island_filter_info_json']))['removed_points'],
                        registration_status=str(a['registration_status']), fitness=float(a['registration_fitness']),
                        median_error_m=float(a['registration_median_error_m']), reference_radius_m=float(a['reference_radius_m']),
                        reference_sha256=str(a['reference_sha256']), timings_json=str(a['timings_json'])))
            rows.append(dict(video_id=entry['video_id'], link=link, method='reg2inv_frame0',
                score=float(np.mean(values)) if values else '', score_p90=float(np.quantile(values, .9)) if values else '',
                score_max=float(np.max(values)) if values else '', frame0_score=frame0_score,
                scored_frames=len(values), total_frames=entry['frame_count'], availability=len(values)/entry['frame_count'],
                poor_alignment_frames=poor, statuses=json.dumps(statuses), quality_warning=entry['quality_warning'],
                reference_sha256=entry['references'][link]['sha256']))
    assert len(rows) == 2 * len(manifest()) and len(frames) == sum(e['frame_count'] * 2 for e in manifest())
    for name, data in [('scores.csv', rows), ('frame_scores.csv', frames)]:
        temp = RESULTS / (name + '.tmp')
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        temp.replace(RESULTS / name)
    return dict(videos=len(manifest()), video_link_scores=len(rows), frame_link_rows=len(frames),
        available=sum(r['scored_frames'] for r in rows), reference='frame0', cad_used=False,
        observation_filter=settings(),
        methods=['reg2inv_frame0'], manifest_sha256=manifest_sha, code_sha256=digest,
        scores_sha256=sha(RESULTS / 'scores.csv'), frame_scores_sha256=sha(RESULTS / 'frame_scores.csv'))
