"""Reg2Inv frame-0 video scores against the selected-45 deformation labels (CPU).

Labels come from the v2 workbook: Link2 uses upper-arm column AD and Link5 uses
forearm column AB. Binary AUROC compares label 1 (severe) with 0 (none),
excludes 0.5 and counts ties as half; higher score predicts deformation.
Labels are confounded with generator, so within-generator AUROC compares
severe/none videos of the same generator only. Descriptive only: no threshold is
fitted.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from robot.analysis.analyze_link5_updated_mask_auroc import labels, metrics
from robot.analysis.mose3_label_auroc import SELECTION, interval, within_generator
from robot.workflows.mose3.inputs import REPO, sha

HERE = Path(__file__).resolve().parent
WORKBOOK = REPO / 'documentation/data/labels/selected_45_matched_videos_styledv2.xlsx'
COLUMNS = {'link2': 'AD', 'link5': 'AB'}
RESULTS = HERE / 'results/20261014-45'
RUNS = {'frame0': (RESULTS / 'scores.csv', 'reg2inv_frame0')}
AGGREGATIONS = ('score', 'score_p90', 'score_max')
GENERATORS = ('COSMOS2.5', 'COSMOS3', 'LVP_ROBOWM')


def within_interval(points, seed=0, rounds=4000):
    """95% bootstrap interval of within-generator AUROC, resampling severe/none per generator."""
    groups = {}
    for case, label, score in points:
        if label in (0., 1.) and score is not None:
            groups.setdefault(case.rsplit('_', 1)[0], {0.: [], 1.: []})[label].append(score)
    groups = {g: v for g, v in groups.items() if v[0.] and v[1.]}
    if not groups:
        return None
    rng, values = np.random.default_rng(seed), []
    for _ in range(rounds):
        wins = pairs = 0.
        for v in groups.values():
            p, n = rng.choice(v[1.], len(v[1.])), rng.choice(v[0.], len(v[0.]))
            wins += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum()
            pairs += len(p) * len(n)
        values.append(wins / pairs)
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def spearman(points):
    data = [(l, s) for _, l, s in points if l is not None and s is not None]
    if len(data) < 3:
        return None
    rank = lambda v: np.argsort(np.argsort(v, kind='stable'), kind='stable').astype(float)
    def average_rank(values):
        values = np.asarray(values, float)
        ranks = rank(values)
        for value in np.unique(values):
            ranks[values == value] = ranks[values == value].mean()
        return ranks
    labels_, scores = zip(*data)
    return float(np.corrcoef(average_rank(labels_), average_rank(scores))[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    selection = json.loads(SELECTION.read_text())
    label = {link: labels(WORKBOOK, selection, column) for link, column in COLUMNS.items()}
    tables = {}
    for name, (path, method) in RUNS.items():
        tables[name] = {(r['video_id'], r['link']): r for r in csv.DictReader(path.open()) if r['method'] == method}
    videos = sorted({v for v, _ in tables['frame0']})
    assert len(videos) == 45 and all((v, l) in tables['frame0'] for v in videos for l in COLUMNS)
    out = dict(definition='label 1 vs 0; 0.5 excluded; ties half; higher score predicts deformation',
               workbook=str(WORKBOOK.relative_to(REPO)), workbook_sha256=sha(WORKBOOK), columns=COLUMNS,
               selection_sha256=sha(SELECTION), videos=videos,
               score_sources={n: dict(path=str(p.relative_to(REPO)), sha256=sha(p), method=m) for n, (p, m) in RUNS.items()},
               results={})
    join = []
    for link in COLUMNS:
        for v in videos:
            join.append(dict(video_id=v, generator=v.rsplit('_', 1)[0], link=link, label=label[link][v],
                             **{f'{run}_{agg}': float(tables[run][(v, link)][agg]) for run in RUNS for agg in AGGREGATIONS}))
        for run in RUNS:
            for agg in AGGREGATIONS:
                points = [(r['video_id'], r['label'], r[f'{run}_{agg}']) for r in join if r['link'] == link]
                result = dict(metrics(points), ci95=interval(points), spearman_rho=spearman(points),
                              within_generator=dict(within_generator(points), ci95=within_interval(points)),
                              by_generator={g: metrics([p for p in points if p[0].startswith(g + '_')])
                                            for g in GENERATORS if any(p[0].startswith(g + '_') for p in points)})
                out['results'][f'{link}/{run}/{agg}'] = result
    results = RESULTS
    (results / 'metadata/label_evaluation.json').write_text(json.dumps(out, indent=2) + '\n')
    with (results / 'metadata/label_evaluation_join.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(join[0]))
        writer.writeheader()
        writer.writerows(join)
    for key, r in out['results'].items():
        w = r['within_generator']
        print(f"{key:28s} AUROC {r['auroc']:.3f} [{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}] pairs {r['valid_pairs']:3d}"
              f" | within-gen {w['auroc']:.3f} [{w['ci95'][0]:.2f}-{w['ci95'][1]:.2f}] pairs {w['pairs']:3d}"
              f" | rho {r['spearman_rho']:+.3f} | counts {r['counts']}")


if __name__ == '__main__':
    main()
