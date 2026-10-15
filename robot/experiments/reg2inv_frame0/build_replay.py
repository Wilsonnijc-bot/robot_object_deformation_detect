"""Build the Reg2Inv frame-0 error replay from collected results (CPU).

Reuses the MoSE3 error-signal replay: its packer and quantizer, its stage, clock,
picker and packed-array decoding (``common3d.js``), its stylesheet and the vendored
three.js are used unchanged. Only the page (``replay/index.html``), its script
(``replay/replay.js``) and the packed data are specific to Reg2Inv. The input-video
inset reuses the MoSE3 replay's display copies after checking their frame counts.

Each frame's observed Link2/Link5 cloud is drawn over that video's frame-0 normal
reference, aligned with Reg2Inv's own registration, and every point is coloured by
its saved per-point anomaly score: white is normal, red is error. Display copies
keep one depth-grid pixel in each 2 x 2 block and quantize positions to 16 bits
inside the video's box; the scientific arrays stay in ``cases/``.
"""
import argparse
import base64
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from robot.replay.mose3.build_replay import (Packer, quantize, frame_count, THREE, THREE_FILES,
                                                          TEMPLATE as MOSE3_TEMPLATE)
from robot.workflows.mose3.inputs import REPO, sha

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / 'replay'
LINKS = ('link2', 'link5')
REUSED = ('common3d.js', 'site.css')
PAGES = ('index.html', 'replay.js')
STRIDE = 2
MISSING = 65535
MOSE3_VIDEOS = REPO / 'robot/results/mose3/20261012/replay/videos'


def display(pixels):
    return (pixels[:, 0] % STRIDE == 0) & (pixels[:, 1] % STRIDE == 0)


def load_case(results, entry):
    """Per frame and link: display points, their scores, the registration and the frame score."""
    case, links = entry['video_id'], {}
    for link in LINKS:
        with np.load(results / 'inputs' / entry['references'][link]['path'], allow_pickle=False) as ref:
            reference = ref['points_m'][display(ref['pixels_yx'])]
            intrinsics = ref['intrinsics']
        frames = []
        for index in range(entry['frame_count']):
            with np.load(results / 'cases' / case / f'{index:05d}_{link}.npz', allow_pickle=False) as a:
                keep = display(a['pixels_yx'])
                ok = str(a['score_status']) == 'ok'
                frames.append(dict(points=a['points_camera_m'][keep], scores=a['point_scores'][keep] if ok else None,
                                   all_scores=a['point_scores'] if ok else None, ok=ok, status=str(a['score_status']),
                                   transform=a['transform_camera_to_reference_m'].astype(np.float64),
                                   link_score=float(a['link_score']) if ok else None,
                                   fitness=float(a['registration_fitness'])))
        links[link] = dict(reference=reference, frames=frames, intrinsics=intrinsics)
    return links


def colour_scale(results, entries):
    """One white-to-red scale for every video: the 95th percentile of all per-point scores after frame 0."""
    values = []
    for entry in entries:
        for link in LINKS:
            for index in range(1, entry['frame_count']):
                with np.load(results / 'cases' / entry['video_id'] / f'{index:05d}_{link}.npz', allow_pickle=False) as a:
                    if str(a['score_status']) == 'ok':
                        values.append(a['point_scores'])
    return float(np.percentile(np.concatenate(values), 95))


def pack_case(case, links, rows, labels, fps, destination):
    every = [l['reference'] for l in links.values()] + [f['points'] for l in links.values() for f in l['frames']]
    stacked = np.concatenate(every).astype(np.float64)
    low, high = stacked.min(0), stacked.max(0)
    reference = np.concatenate([l['reference'] for l in links.values()])
    intrinsics = links['link2']['intrinsics'].astype(np.float64)
    packer = Packer()
    header = dict(case=case, frames=len(links['link2']['frames']), fps=fps,
                  grid_hw=[int(round(2 * intrinsics[1, 2])), int(round(2 * intrinsics[0, 2]))],
                  intrinsics=intrinsics.tolist(), median_depth=float(np.median(reference[:, 2])),
                  target=reference.mean(0).tolist(), replay_stride=STRIDE, low=low.tolist(), high=high.tolist(), links={})
    for link, data in links.items():
        frames = data['frames']
        counts = [len(f['points']) for f in frames]
        points = np.concatenate([f['points'] for f in frames])
        scores = np.concatenate([f['scores'] if f['ok'] else np.full(len(f['points']), np.nan) for f in frames])
        top = float(np.nanmax(scores)) if np.isfinite(scores).any() else 1.0
        quantized = np.where(np.isfinite(scores), np.rint(np.clip(scores / max(top, 1e-12), 0, 1) * 65534), MISSING)
        transforms = np.stack([f['transform'] if f['ok'] else np.full((4, 4), np.nan) for f in frames])
        row = rows[(case, link)]
        header['links'][link] = dict(
            reference=packer.add(quantize(data['reference'], low, high)), reference_count=len(data['reference']),
            offsets=packer.add(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)),
            position=packer.add(quantize(points, low, high)), score=packer.add(quantized.astype(np.uint16)), top=top,
            transform=packer.add(transforms.astype(np.float32)),
            series=[f['link_score'] for f in frames], status=[f['status'] for f in frames],
            fitness=[round(f['fitness'], 4) for f in frames],
            mean=float(row['score']), p90=float(row['score_p90']), max=float(row['score_max']),
            max_frame=int(np.nanargmax([np.nan if s is None else s for s in (f['link_score'] for f in frames)])),
            label=labels[link].get(case), points_per_frame=int(np.median([np.sum(np.isfinite(f['all_scores'])) if f['ok']
                                                                           else 0 for f in frames])))
    destination.write_text('window.REG2INV_CASE(' + json.dumps(header, separators=(',', ':')) + ',"'
                           + base64.b64encode(packer.bytes()).decode('ascii') + '");\n')
    return header


def fps_of(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                   'stream=avg_frame_rate', '-of', 'csv=p=0', str(path)], text=True).strip()
    numerator, denominator = out.split('/')
    return float(numerator) / float(denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results/20261014-45')
    args = parser.parse_args()
    results, output = args.results, args.results / 'replay'
    manifest = json.loads((results / 'inputs/manifest.json').read_text())['entries']
    evaluation = json.loads((results / 'metadata/label_evaluation.json').read_text())
    rows = {(r['video_id'], r['link']): r for r in csv.DictReader((results / 'scores.csv').open())}
    join = list(csv.DictReader((results / 'metadata/label_evaluation_join.csv').open()))
    labels = {link: {r['video_id']: (None if r['label'] in ('', 'None') else float(r['label']))
                     for r in join if r['link'] == link} for link in LINKS}
    for name in ('cases', 'videos', 'assets'):
        (output / name).mkdir(parents=True, exist_ok=True)
    for name in THREE_FILES:
        shutil.copy2(THREE / name, output / 'assets' / name)
    for name in REUSED:
        shutil.copy2(MOSE3_TEMPLATE / name, output / name)
    for name in PAGES:
        shutil.copy2(TEMPLATE / name, output / name)
    scale = colour_scale(results, manifest)
    index = []
    for entry in manifest:
        case = entry['video_id']
        video = output / 'videos' / f'{case}.mp4'
        source = MOSE3_VIDEOS / f'{case}.mp4'
        if frame_count(source) != entry['frame_count']:
            raise ValueError(f'Display video frame count differs: {case}')
        shutil.copy2(source, video)
        header = pack_case(case, load_case(results, entry), rows, labels, fps_of(video), output / 'cases' / f'{case}.js')
        index.append(dict(case=case, frames=header['frames'], fps=header['fps'], warning=entry['quality_warning'],
                          labels={l: labels[l].get(case) for l in LINKS},
                          stats={l: {k: header['links'][l][k] for k in ('mean', 'p90', 'max', 'max_frame')} for l in LINKS}))
        print('REPLAY_CASE', case, flush=True)
    primary = {link: evaluation['results'][f'{link}/frame0/score'] for link in LINKS}
    data = dict(title='Reg2Inv with a frame-0 normal reference', links=list(LINKS), scale=scale, stride=STRIDE,
                auroc={l: dict(pooled=primary[l]['auroc'], ci95=primary[l]['ci95'],
                               within=primary[l]['within_generator']['auroc'],
                               within_pairs=primary[l]['within_generator']['pairs'], counts=primary[l]['counts'])
                       for l in LINKS},
                label_sources={l: f"{Path(evaluation['workbook']).name} column {evaluation['columns'][l]}" for l in LINKS},
                scores_sha256=sha(results / 'scores.csv'), cases=index)
    (output / 'data.js').write_text('window.REG2INV_INDEX=' + json.dumps(data, separators=(',', ':')) + ';\n')
    print(json.dumps(dict(output=str(output), videos=len(index), colour_scale_p95=scale)))


if __name__ == '__main__':
    main()
