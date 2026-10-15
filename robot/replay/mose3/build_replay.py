"""Build the MoSE3-style interactive replay from collected tracking outputs (CPU).

Each video becomes one script that the viewer loads on demand (works from
``file://``): the moving dense cloud of query-frame pixels with RGB and
rigidity-embedding colours, and for Link2/Link5 points their 3D trajectories,
per-point SE(3) tripods and visibility. Positions are quantized to 16 bits inside
the video's bounding box and axis directions to int8, for display only; the
scientific arrays stay in ``cases/``. Browser videos are re-encoded copies with
their frame count checked against the tracked frames.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from robot.workflows.mose3.inputs import LINKS, REPO, sha
from robot.workflows.mose3.point_fields import quat_to_matrix
from robot.scoring.mose3.rigidity import rigid_fit_residual

TEMPLATE = Path(__file__).resolve().parent / 'templates'
THREE = REPO / 'infrastructure/shared/replay/assets/three-r147'
THREE_FILES = ('three.min.js', 'OrbitControls.js', 'LineSegmentsGeometry.js', 'LineMaterial.js',
               'LineSegments2.js', 'LICENSE')
PAGES = ('index.html', 'signals.html', 'signals.js', 'videos.html', 'videos.js', 'video.html', 'viewer.js', 'common3d.js', 'site.css')
# Signals replayed per point: label, unit, and which rigid prediction (if any) the residual is measured from.
SIGNALS = {
    'rigid_residual': ('Rigid residual', 'fraction of link length', 'prediction',
                       "Distance from where MoSE3's own SE(3) puts the point."),
    'rigid_residual_refit': ('Rigid residual, visible refit', 'fraction of link length', 'prediction_refit',
                             "Distance from one rigid motion fitted to the link's visible points."),
    'rigid_residual_dense': ('Rigid residual, dense mask fit', 'fraction of link length', 'prediction_dense',
                             "Distance from one robust rigid motion fitted to every visible pixel of the link's mask."),
    'rigidity_discrepancy': ('Rigidity discrepancy', '1 − cos', None,
                             "How far the point's rigidity vector is from the link's mean."),
}
MISSING = 65535
# Label and note for each 2D view, shown by both pages (wording follows the MoSE3 project page).
VIEWS = {
    'input': ('Input', 'Input video.'),
    'tracks': ('Tracks', 'MoSE3 3D tracks of the PDI-initialized Link2 and Link5 points, projected into each frame. '
                         'Hollow heads are occluded.'),
    'tracks_world': ('Tracks, camera fixed', "The same world-space tracks re-rendered from the first camera over the "
                     "first frame: the camera's own motion is removed, so only what moves in the world moves on screen."),
    'se3': ('SE(3)', "Each point's SE(3) as a moving x/y/z frame, on 24 spread-out points per link. At frame 0, blue z "
                     'is the outward surface normal; dashed frames are occluded.'),
    'combo': ('Tracking + SE(3)', "Every point's trail over the SE(3) frames."),
    'pca': ('Rigidity embedding', "MoSE3's per-pixel rigidity embedding projected to RGB by PCA. Parts that move "
                                  'together share a colour; no segmentation supervision is used.'),
    'se3_on_pca': ('Frames over the embedding', 'SE(3) frames drawn over the rigidity embedding they were fitted from.'),
}
SCORES = ('rigid_fit_residual', 'pair_ratio_mad', 'se3_rotation_spread_deg', 'embedding_dispersion')
TRIPODS_PER_LINK = 24   # upstream draws tripods on a sparse 48 px grid; trails and heads use every point


class Packer:
    """Concatenate 4-byte-aligned arrays and remember where each one lives."""

    def __init__(self):
        self.chunks, self.size = [], 0

    def add(self, array):
        data = np.ascontiguousarray(array).tobytes()
        offset = self.size
        padding = (-len(data)) % 4
        self.chunks.append(data + b'\0' * padding)
        self.size += len(data) + padding
        return [offset, int(np.asarray(array).size)]

    def bytes(self):
        return b''.join(self.chunks)


def quantize(points, low, high):
    scaled = (np.nan_to_num(points.astype(np.float64), nan=0.0) - low) / np.maximum(high - low, 1e-9)
    return np.rint(np.clip(scaled, 0, 1) * 65535).astype(np.uint16)


def spread_subset(xy, count):
    """Deterministic farthest-point subset of query positions for display tripods."""
    chosen = [int(np.argmin(xy[:, 0] + xy[:, 1]))]
    distance = np.linalg.norm(xy - xy[chosen[0]], axis=1)
    while len(chosen) < min(count, len(xy)):
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(xy - xy[chosen[-1]], axis=1))
    mask = np.zeros(len(xy), bool)
    mask[chosen] = True
    return mask


def link_arrays(arrays):
    """Display tripods: upstream's surface-normal start frame moved by each point's SE(3)."""
    rotation = quat_to_matrix(arrays['se3_quat_wxyz']) @ arrays['display_frame'][None].astype(np.float64)
    axes = np.swapaxes(rotation, -1, -2)                                     # [T, N, 3 axes, 3]
    ok = arrays['se3_valid'] & arrays['display_frame_ok'][None]
    return axes, ok, spread_subset(arrays['query_xy_source'].astype(np.float64), TRIPODS_PER_LINK)


def pack_case(case, folder, entry, destination):
    with np.load(folder / 'scene.npz', allow_pickle=False) as archive:
        scene = {name: archive[name] for name in archive.files}
    links = {}
    for link in LINKS:
        with np.load(folder / f'{link}.npz', allow_pickle=False) as archive:
            links[link] = {name: archive[name] for name in archive.files}
    receipt = json.loads((folder / 'receipt.json').read_text())
    rigidity = json.loads((folder / 'rigidity.json').read_text())
    cloud = scene['points'].astype(np.float32)
    every = [cloud.reshape(-1, 3)] + [links[l][k].reshape(-1, 3) for l in LINKS
                                      for k in ('trajectories_world', 'se3_origin')]
    stacked = np.concatenate(every)
    stacked = stacked[np.isfinite(stacked).all(1)]
    low, high = np.percentile(stacked, 0.1, axis=0), np.percentile(stacked, 99.9, axis=0)
    low, high = np.minimum(low, stacked.min(0)), np.maximum(high, stacked.max(0))
    packer = Packer()
    header = dict(
        case=case, frames=int(receipt['frame_count']), fps=receipt['fps'], grid_hw=receipt['grid_hw'],
        source_hw=receipt['source_hw'], intrinsics=receipt['intrinsics'],
        median_depth=float(scene['median_depth']), low=low.tolist(), high=high.tolist(),
        replay_stride=int(receipt['settings']['replay_stride']),
        camera_poses=np.round(scene['camera_poses'], 6).tolist(),
        cloud=dict(count=int(cloud.shape[1]), position=packer.add(quantize(cloud, low, high)),
                   rgb=packer.add(scene['rgb'].astype(np.uint8)), pca=packer.add(scene['pca'].astype(np.uint8))),
        links={}, masks=rigidity['masks'])
    for link in LINKS:
        arrays = links[link]
        axes, ok, tripod = link_arrays(arrays)
        # Point inspector: rotation angle from the query frame, and each point's own rigid-fit residual.
        quat = arrays['se3_quat_wxyz'].astype(np.float64)
        angle = np.degrees(2 * np.arccos(np.clip(np.abs(quat[..., 0]), 0, 1)))
        _, per_point = rigid_fit_residual(arrays['trajectories_world'], arrays['visible'])
        later = per_point[1:]
        seen = np.isfinite(later)
        residual = np.where(seen.any(0), np.where(seen, later, 0).sum(0) / np.maximum(seen.sum(0), 1), -1.0)
        header['links'][link] = dict(
            count=int(len(arrays['point_ids'])),
            trajectory=packer.add(quantize(arrays['trajectories_world'], low, high)),
            origin=packer.add(quantize(arrays['se3_origin'], low, high)),
            axes=packer.add(np.rint(np.clip(axes, -1, 1) * 127).astype(np.int8)),
            visible=packer.add(arrays['visible'].astype(np.uint8)),
            axes_ok=packer.add(ok.astype(np.uint8)),
            tripod=packer.add(tripod.astype(np.uint8)),
            trail_rgb=packer.add(arrays['trail_rgb'].astype(np.uint8)),
            pca_rgb=packer.add(arrays['pca_rgb'].astype(np.uint8)),
            point_ids=packer.add(arrays['point_ids'].astype(np.int32)),
            query_xy=packer.add(arrays['query_xy_source'].astype(np.float32)),
            rigidity_vector=packer.add(arrays['rigidity_embedding_query'].astype(np.float32)),
            residual=packer.add(residual.astype(np.float32)),     # -1: never available after frame 0
            rotation_centideg=packer.add(np.rint(np.clip(angle, 0, 180) * 100).astype(np.uint16)),
            on_edge=packer.add(arrays['on_depth_edge'].astype(np.uint8)),
            scores=rigidity['links'][link]['scores'],
            series={name: rigidity['links'][link]['series'][name] for name in SCORES},
            interior=dict(points=rigidity['links'][link]['interior']['points'],
                          scores=rigidity['links'][link]['interior']['scores'],
                          series={name: rigidity['links'][link]['interior']['series'][name] for name in SCORES}),
            visible_fraction=rigidity['links'][link]['visible_fraction'],
            on_depth_edge=rigidity['links'][link]['on_depth_edge'],
            frame0_reprojection_px=rigidity['links'][link]['frame0_reprojection_px'])
    payload = base64.b64encode(packer.bytes()).decode('ascii')
    destination.write_text('window.MOSE3_SCENE(' + json.dumps(header, separators=(',', ':'))
                           + ',"' + payload + '");\n')
    return header


def signal_scales(run):
    """One colour scale per signal for every video: the 95th percentile of all per-point values."""
    values = {key: [] for key in SIGNALS}
    for path in sorted((run / 'signals/points').glob('*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            for key in SIGNALS:
                for link in LINKS:
                    v = archive[f'{link}_{key}']
                    values[key].append(v[np.isfinite(v)])
    return {key: float(np.percentile(np.concatenate(v), 95)) for key, v in values.items()}


def pack_signals(case, run, scene, labels, rows, destination):
    """Per-point signal values (16-bit, 65535 = not scored) and rigid predictions in the scene's box."""
    frames = json.loads((run / 'signals/frames' / f'{case}.json').read_text())
    low, high = np.asarray(scene['low']), np.asarray(scene['high'])
    packer = Packer()
    header = dict(case=case, links={})
    with np.load(run / 'signals/points' / f'{case}.npz', allow_pickle=False) as archive:
        for link in LINKS:
            row = rows[(case, link)]
            block = dict(selected=packer.add(archive[f'{link}_selected'].astype(np.uint8)), label=labels[link].get(case),
                         signals={})
            for key, (_, _, prediction, _) in SIGNALS.items():
                values = archive[f'{link}_{key}']
                top = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
                quantized = np.where(np.isfinite(values), np.rint(np.clip(values / max(top, 1e-12), 0, 1) * 65534), MISSING)
                entry = dict(values=packer.add(quantized.astype(np.uint16)), top=top, series=frames[link][key],
                             max=row[f'{key}_max'], max_frame=row[f'{key}_max_frame'], p90=row[f'{key}_p90'])
                if prediction:
                    entry['prediction'] = packer.add(quantize(archive[f'{link}_{prediction}'], low, high))
                block['signals'][key] = entry
            header['links'][link] = block
    destination.write_text('window.MOSE3_SIGNALS(' + json.dumps(header, separators=(',', ':')) + ',"'
                           + base64.b64encode(packer.bytes()).decode('ascii') + '");\n')


def frame_count(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                   '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(path)], text=True)
    return int(out.strip())


def display_video(source, destination, frames, fps):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-frames:v', str(frames),
                    '-vf', 'scale=640:-2:flags=area', '-r', f'{fps:.6f}', '-c:v', 'libx264', '-preset', 'slow',
                    '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(destination)],
                   check=True)
    if frame_count(destination) != frames:
        raise ValueError(f'Display copy frame count differs: {destination}')


def thumbnail(source, destination):
    capture = cv2.VideoCapture(str(source))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f'Cannot read thumbnail frame: {source}')
    height, width = frame.shape[:2]
    cv2.imwrite(str(destination), cv2.resize(frame, (208, round(208 * height / width)), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Run folder with inputs/ and cases/')
    parser.add_argument('--sources', type=Path, required=True, help='Local copy of the inputs/videos root')
    parser.add_argument('--output', type=Path, help='Default: <run>/replay')
    args = parser.parse_args()
    output = args.output or args.run / 'replay'
    manifest = json.loads((args.run / 'inputs/manifest.json').read_text())
    for name in ('cases', 'videos', 'thumbs', 'assets', 'signals'):
        (output / name).mkdir(parents=True, exist_ok=True)
    for name in THREE_FILES:
        shutil.copy2(THREE / name, output / 'assets' / name)
    for name in PAGES:
        shutil.copy2(TEMPLATE / name, output / name)
    signal_run = (args.run / 'signals/label_auroc.json').is_file()
    if signal_run:
        auroc = json.loads((args.run / 'signals/label_auroc.json').read_text())['links']
        labels = {link: block['labels'] for link, block in auroc.items()}
        number = lambda v: None if v in ('', 'None', None) else float(v)
        rows = {(r['case'], r['link']): {k: (number(v) if k.endswith(('_max', '_p90')) else
                                            (int(v) if k.endswith('_max_frame') and v not in ('', 'None') else v))
                                         for k, v in r.items()}
                for r in csv.DictReader((args.run / 'signals/signals.csv').open())}
        scales = signal_scales(args.run)
    index = []
    for entry in manifest['entries']:
        case = entry['video_id']
        folder = args.run / 'cases' / case
        source = args.sources / entry['video']
        if sha(source) != entry['source_video_sha256']:
            raise ValueError(f'Local source copy differs from the tracked video: {case}')
        thumbnail(source, output / 'thumbs' / f'{case}.jpg')
        if not (folder / 'receipt.json').is_file():
            index.append(dict(case=case, status='missing', masks={l: entry['masks'][l]['review'] for l in LINKS}))
            continue
        header = pack_case(case, folder, entry, output / 'cases' / f'{case}.js')
        if signal_run:
            pack_signals(case, args.run, header, labels, rows, output / 'signals' / f'{case}.js')
        video = output / 'videos' / f'{case}.mp4'
        if not video.exists():
            display_video(source, video, header['frames'], header['fps'])
        views = [v for v in VIEWS if (output / 'views' / case / f'{v}.mp4').is_file()]
        index.append(dict(case=case, status='complete', frames=header['frames'], fps=header['fps'], views=views,
                          visible_last={l: header['links'][l]['visible_fraction'][-1] for l in LINKS},
                          masks=header['masks'], points={l: header['links'][l]['count'] for l in LINKS},
                          scores={l: header['links'][l]['scores'] for l in LINKS},
                          interior_scores={l: header['links'][l]['interior']['scores'] for l in LINKS},
                          **(dict(labels={l: labels[l].get(case) for l in LINKS},
                                  signals={l: {k: dict(max=rows[(case, l)][f'{k}_max'], max_frame=rows[(case, l)][f'{k}_max_frame'])
                                               for k in SIGNALS} for l in LINKS}) if signal_run else {})))
        print('REPLAY_CASE', case, flush=True)
    data = dict(title='MoSE3 3D tracking of PDI-initialized Link2 and Link5 points', links=list(LINKS),
                scores=list(SCORES), views=[dict(key=k, label=l, note=n) for k, (l, n) in VIEWS.items()],
                default_view='combo', cases=index)
    if signal_run:
        data['signals'] = [dict(key=k, label=l, unit=u, residual=bool(p), note=n, scale=scales[k],
                                auroc={link: auroc[link]['results'][f'{k}_max']['auroc'] for link in LINKS},
                                ordinal={link: auroc[link]['results'][f'{k}_max']['with_moderate']['ordinal_c_index']['auroc']
                                         for link in LINKS})
                           for k, (l, u, p, n) in SIGNALS.items()]
        data['label_sources'] = {link: f"{auroc[link]['workbook']} column {auroc[link]['column']}" for link in LINKS}
    (output / 'data.js').write_text('window.MOSE3_INDEX=' + json.dumps(data, separators=(',', ':')) + ';\n')
    print(json.dumps(dict(output=str(output), complete=sum(c['status'] == 'complete' for c in index),
                          missing=[c['case'] for c in index if c['status'] != 'complete'])))


if __name__ == '__main__':
    main()
