"""Verify collected MoSE3 case folders against their receipts and tabulate scores (CPU).

Transfer first, e.g. ``ssh ERIS "tar -C $RUN -cf - cases metadata" | tar -C <run> -xf -``.
Every file listed in a receipt must match its SHA-256; a mismatch stops collection.
Writes ``<run>/collection.json`` and ``<run>/rigidity_scores.csv`` (one row per video and
link; scores for all points and for points away from depth edges).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from robot.workflows.mose3.inputs import LINKS, sha

SCORES = ('rigid_fit_residual', 'pair_ratio_mad', 'se3_rotation_spread_deg', 'embedding_dispersion')


def collect(run):
    manifest = json.loads((run / 'inputs/manifest.json').read_text())
    rows, verified, missing = [], {}, []
    for entry in manifest['entries']:
        case = entry['video_id']
        folder = run / 'cases' / case
        if not (folder / 'receipt.json').is_file():
            missing.append(case)
            continue
        receipt = json.loads((folder / 'receipt.json').read_text())
        if receipt['status'] != 'complete' or receipt['source_video_sha256'] != entry['source_video_sha256']:
            raise ValueError(f'Receipt does not describe the frozen video: {case}')
        for name, expected in receipt['files'].items():
            if sha(folder / name) != expected:
                raise ValueError(f'Transferred file hash mismatch: {case}/{name}')
        verified[case] = len(receipt['files'])
        rigidity = json.loads((folder / 'rigidity.json').read_text())
        for link in LINKS:
            summary = rigidity['links'][link]
            row = dict(case=case, link=link, mask_review=entry['masks'][link]['review'],
                       frames=receipt['frame_count'], points=summary['points'],
                       interior_points=summary['interior']['points'],
                       visible_last_frame=summary['visible_fraction'][-1],
                       frame0_offset_px_median=summary['frame0_reprojection_px']['median'])
            row.update({name: summary['scores'][name] for name in SCORES})
            row.update({f'interior_{name}': summary['interior']['scores'][name] for name in SCORES})
            rows.append(row)
    with (run / 'rigidity_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(cases=len(manifest['entries']), verified=len(verified), missing=missing,
                  files_verified=sum(verified.values()), scores_csv_sha256=sha(run / 'rigidity_scores.csv'))
    (run / 'collection.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(collect(parser.parse_args().run)))
