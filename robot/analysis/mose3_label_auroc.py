"""AUROC of the MoSE3 link signals against the selected-45 deformation labels (CPU).

Reuses the repository's workbook reader and AUROC: label 1 (severe) against 0 (none),
moderate 0.5 excluded, ties count half, higher signal predicts deformation. Link2 uses
the original workbook column AD; Link5 the v2 workbook column AB, as in the published
link analyses. Labels are confounded with generator (Link2: COSMOS2.5 11 severe / 2 none, COSMOS3 0 / 13),
so ``within_generator`` compares severe and none videos of the same generator only.
``with_moderate`` keeps the moderate videos three ways: severe + moderate vs none,
severe vs moderate + none, and an ordinal C-index over all three levels (each pair of
videos with different labels counts; ties half), each pooled, with a stratified 95%
bootstrap interval, and within generator. Descriptive only: no threshold is fitted.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from robot.analysis.analyze_link5_updated_mask_auroc import labels, metrics
from robot.workflows.mose3.inputs import REPO, sha

WORKBOOKS = {'link2': ('selected_45_matched_videos_styled.xlsx', 'AD'),
             'link5': ('selected_45_matched_videos_styledv2.xlsx', 'AB')}
SELECTION = REPO / 'robot/experiments/link7_tracker_filter_comparison/results/link5-link7-four-way-selected45-20260927/selection.json'
BOOTSTRAP = 4000


def interval(points, seed=0):
    """95% stratified bootstrap interval of the severe-vs-none AUROC (ties half)."""
    positive = np.array([s for _, l, s in points if l == 1 and s is not None])
    negative = np.array([s for _, l, s in points if l == 0 and s is not None])
    rng, values = np.random.default_rng(seed), []
    for _ in range(BOOTSTRAP):
        p, n = rng.choice(positive, len(positive)), rng.choice(negative, len(negative))
        values.append(((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean()))
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def within_generator(points):
    """AUROC over severe-none pairs from the same generator only (labels are confounded with generator)."""
    wins = pairs = 0.0
    for generator in ('COSMOS2.5', 'COSMOS3', 'LVP_ROBOWM'):
        group = [(l, s) for c, l, s in points if c.startswith(generator + '_') and s is not None]
        for lp, sp in group:
            for ln, sn in group:
                if lp == 1 and ln == 0:
                    wins += 1.0 if sp > sn else 0.5 if sp == sn else 0.0
                    pairs += 1
    return dict(auroc=wins / pairs if pairs else None, pairs=int(pairs))


MODERATE_SCHEMES = {'severe_moderate_vs_none': lambda label: float(label > 0),
                    'severe_vs_moderate_none': lambda label: float(label == 1),
                    'ordinal_c_index': lambda label: label}


def c_index(points, same_generator=False):
    """P(higher label scores higher) over pairs with different labels; ties half."""
    wins = pairs = 0.0
    for case_a, label_a, score_a in points:
        for case_b, label_b, score_b in points:
            if label_a > label_b and (not same_generator or case_a.rsplit('_', 1)[0] == case_b.rsplit('_', 1)[0]):
                wins += 1.0 if score_a > score_b else 0.5 if score_a == score_b else 0.0
                pairs += 1
    return (wins / pairs if pairs else None), int(pairs)


def with_moderate(points, seed=0):
    """The three ways of keeping moderate videos (see the module docstring)."""
    usable = [(c, l, s) for c, l, s in points if l is not None and s is not None and np.isfinite(s)]
    out = {}
    for name, recode in MODERATE_SCHEMES.items():
        coded = [(c, recode(l), s) for c, l, s in usable]
        value, pairs = c_index(coded)
        strata = [[p for p in coded if p[1] == level] for level in sorted({p[1] for p in coded})]
        rng, draws = np.random.default_rng(seed), []
        for _ in range(BOOTSTRAP // 4):
            sample = [stratum[i] for stratum in strata for i in rng.integers(0, len(stratum), len(stratum))]
            draws.append(c_index(sample)[0])
        within, within_pairs = c_index(coded, same_generator=True)
        out[name] = dict(auroc=value, pairs=pairs, ci95=[float(v) for v in np.percentile(draws, [2.5, 97.5])],
                         within_generator=dict(auroc=within, pairs=within_pairs),
                         counts={str(level): sum(p[1] == level for p in coded) for level in sorted({p[1] for p in coded})})
    return out


SIGNALS = ('rigidity_discrepancy', 'rigid_residual', 'rigid_residual_refit', 'rigid_residual_dense',
           'rigid_residual_dense_all', 'seam_stretch', 'seam_shear', 'seam_bend_deg', 'seam_twist_deg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.run / 'signals/signals.csv').open()))
    selection = json.loads(SELECTION.read_text())
    out = dict(definition='label 1 vs 0; 0.5 excluded; ties half; higher signal predicts deformation',
               selection_sha256=sha(SELECTION), signals_sha256=sha(args.run / 'signals/signals.csv'), links={})
    for link, (name, column) in WORKBOOKS.items():
        workbook = REPO / 'documentation/data/labels' / name
        label = labels(workbook, selection, column)
        mine = {r['case']: r for r in rows if r['link'] == link}
        reviewed = {case for case, r in mine.items() if r['mask_review'] in ('latest_guarded_mask', 'four_way_base_mask')}
        results, accepted_masks = {}, {}
        for signal in SIGNALS:
            if not any(f'{signal}_max' in r for r in mine.values()):
                continue  # dense-fit signals exist only for runs that cached the mask fields
            for stat in ('max', 'p90'):
                key = f'{signal}_{stat}'
                points = [(case, label.get(case), float(r[key]) if r.get(key) not in (None, '', 'None') else None)
                          for case, r in mine.items()]
                results[key] = dict(metrics(points), ci95=interval(points), within_generator=within_generator(points),
                                    with_moderate=with_moderate(points))
                # Sensitivity: drop videos whose mask the user rejected or the handoff excluded.
                accepted_masks[key] = metrics([p for p in points if p[0] in reviewed])['auroc']
        frames = [(case, label.get(case), float(r['frames'])) for case, r in mine.items()]
        out['links'][link] = dict(workbook=name, column=column, workbook_sha256=sha(workbook), results=results,
                                  labels={case: label.get(case) for case in sorted(mine)},
                                  clip_length_alone_auroc=metrics(frames)['auroc'],
                                  accepted_masks_only=dict(videos=len(reviewed), auroc=accepted_masks))
    (args.run / 'signals/label_auroc.json').write_text(json.dumps(out, indent=2) + '\n')
    for link, block in out['links'].items():
        counts = next(iter(block['results'].values()))['counts']
        print(f"{link} ({block['workbook']} {block['column']}; severe {counts['1.0']}, none {counts['0.0']}, moderate {counts['0.5']})")
        for key, r in block['results'].items():
            print(f'   {key:28s} AUROC {r["auroc"]:.3f}  95% CI {r["ci95"][0]:.2f}-{r["ci95"][1]:.2f}'
                  f'  | within generator {r["within_generator"]["auroc"]:.3f} ({r["within_generator"]["pairs"]} pairs)'
                  f'  | accepted masks only {block["accepted_masks_only"]["auroc"][key]:.3f}')


if __name__ == '__main__':
    main()
