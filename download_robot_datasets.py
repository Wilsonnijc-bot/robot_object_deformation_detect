#!/usr/bin/env python3
"""Download untouched official robot videos, investigate source quality, and audit.

Usage: python download_robot_datasets.py --phase discover|research|pilot|full|audit
Source revisions are pinned on first discovery. No video encoding is performed.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from urllib.parse import urljoin, urlparse, unquote, parse_qsl, urlencode, urlunparse

import json5
import requests
from huggingface_hub import HfApi, hf_hub_url
from PIL import Image

GALLERY = 'https://humanoideveryday.github.io/data_viewer/data_viewer_page.html'
GH_REPO = 'HumanoidEveryday/humanoideveryday.github.io'
HF_REPO = 'nvidia/PhysicalAI-Robotics-GR00T-GR1'
HE_HF = 'USC-PSI-Lab/humanoid-everyday'
SHEET = 'https://docs.google.com/spreadsheets/d/158Wzf8Xywky3aHJSCfp3OZxf4bkhzAJdcG94eHf8gVc/export?format=csv&gid=1307250382'
DATA_ROOT = Path(__file__).resolve().parent
DATASETS = ('Humanoid_Everyday', 'NVIDIA_GR1_100')
LOCAL = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{threading.get_ident()}.{time.time_ns()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.headers.update({'User-Agent': 'RobotDatasetSourceAudit/1.0', 'Accept-Encoding': 'identity'})
    return LOCAL.session


def fetch(url, binary=False):
    for attempt in range(5):
        try:
            r = session().get(url, timeout=(20, 120))
            r.raise_for_status()
            return r.content if binary else r.text
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(30, 2 ** attempt) + random.random())


def run(cmd, timeout=600):
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr.decode('utf-8', errors='replace')[-6000:])
    return p.stdout


def hashes(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(('blob %d\0' % size).encode())
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(b)
            git.update(b)
    return {'sha256': sha.hexdigest(), 'git_blob_sha1': git.hexdigest(), 'file_size_bytes': size}


def check_identity(path, expected):
    with path.open('rb') as f:
        prefix = f.read(256)
    if b'ftyp' not in prefix[:64]:
        raise ValueError('Missing MP4 ftyp signature; possible HTML, LFS pointer, or invalid file')
    h = hashes(path)
    if expected.get('size') is not None and h['file_size_bytes'] != expected['size']:
        raise ValueError('Source size mismatch')
    for source_key, actual_key in [('sha256', 'sha256'), ('blob_id', 'git_blob_sha1')]:
        if expected.get(source_key) and h[actual_key] != expected[source_key]:
            raise ValueError('Source checksum mismatch: ' + source_key)
    return h


def download(url, path, expected=None):
    """Resume only the same URL/entity; verify range and total; atomically finish."""
    expected = expected or {}
    if path.exists():
        try:
            check_identity(path, expected)
            return {'download_timestamp': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), 'reused': True}
        except Exception:
            # Preserve suspect originals for inspection; never silently overwrite.
            quarantine = path.with_name(path.name + '.invalid-' + str(time.time_ns()))
            path.replace(quarantine)
            logging.warning('Quarantined invalid original: %s', quarantine)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    state_path = path.with_name(path.name + '.part.json')
    last_error = None
    for attempt in range(5):
        try:
            state = read_json(state_path) if state_path.exists() else {}
            if part.exists() and state.get('url') != url:
                part.unlink()
            offset = part.stat().st_size if part.exists() else 0
            headers = {}
            if offset:
                headers['Range'] = f'bytes={offset}-'
                if state.get('validator'):
                    headers['If-Range'] = state['validator']
                elif not (expected.get('sha256') or expected.get('blob_id')):
                    offset = 0
            with session().get(url, headers=headers, stream=True, timeout=(20, 120)) as r:
                if r.status_code == 416 and offset and expected.get('size') == offset:
                    check_identity(part, expected)
                    part.replace(path)
                    state_path.unlink(missing_ok=True)
                    return {'download_timestamp': now(), 'reused': False, 'resumed': True}
                r.raise_for_status()
                if 'text/html' in r.headers.get('Content-Type', '').lower():
                    raise ValueError('HTML response rejected')
                if r.status_code == 206:
                    m = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers.get('Content-Range', ''))
                    if not m or int(m[1]) != offset:
                        raise ValueError('Invalid Content-Range')
                    total = int(m[3])
                    mode = 'ab'
                else:
                    offset = 0
                    total = int(r.headers['Content-Length']) if r.headers.get('Content-Length') else None
                    mode = 'wb'
                validator = r.headers.get('ETag') or r.headers.get('Last-Modified')
                write_json(state_path, {'url': url, 'validator': validator, 'total': total})
                with part.open(mode) as f:
                    for b in r.iter_content(1024 * 1024):
                        if b:
                            f.write(b)
                    f.flush()
                    os.fsync(f.fileno())
                if total is not None and part.stat().st_size != total:
                    raise ValueError('Incomplete HTTP response')
                check_identity(part, expected)
                part.replace(path)
                state_path.unlink(missing_ok=True)
                return {'download_timestamp': now(), 'reused': False, 'resumed': offset > 0,
                        'http_etag': r.headers.get('ETag'), 'http_content_length': total}
        except Exception as e:
            last_error = e
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            logging.warning('Download attempt %d for %s: %s', attempt + 1, url, e)
            if isinstance(e, ValueError):
                part.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
            if status in (403, 404) or attempt == 4:
                break
            retry_after = getattr(getattr(e, 'response', None), 'headers', {}).get('Retry-After', '')
            delay = min(60, int(retry_after)) if retry_after.isdigit() else min(30, 2 ** attempt)
            time.sleep(delay + random.random())
    raise RuntimeError(str(last_error))


def probe(path):
    raw = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))
    streams = [s for s in raw['streams'] if s['codec_type'] == 'video' and not s.get('disposition', {}).get('attached_pic')]
    if not streams or 'mp4' not in raw['format']['format_name']:
        raise ValueError('Not a valid MP4 with a video stream')
    s = streams[0]
    rotation = float(next((d['rotation'] for d in s.get('side_data_list', []) if 'rotation' in d), s.get('tags', {}).get('rotate', 0)))
    w, h = s['width'], s['height']
    dw, dh = (h, w) if abs(rotation) % 180 == 90 else (w, h)
    fps = s.get('avg_frame_rate', '0/1')
    return {'width': w, 'height': h, 'display_width': dw, 'display_height': dh,
            'rotation_degrees': rotation, 'fps': float(Fraction(fps)), 'fps_rational': fps,
            'nominal_fps_rational': s.get('r_frame_rate'),
            'duration_seconds': float(s.get('duration', raw['format'].get('duration', 0))),
            'codec': s['codec_name'], 'pixel_format': s.get('pix_fmt'),
            'video_bitrate': int(s['bit_rate']) if s.get('bit_rate') else None,
            'frame_count': int(s['nb_frames']) if s.get('nb_frames', '').isdigit() else None,
            'sample_aspect_ratio': s.get('sample_aspect_ratio'),
            'color_space': s.get('color_space'), 'color_range': s.get('color_range'),
            'file_size_bytes': path.stat().st_size, 'stream_index': s['index'], 'ffprobe_raw': raw}


def validate(path, expected):
    h = check_identity(path, expected)
    props = probe(path)
    progress = run(['ffmpeg', '-hide_banner', '-v', 'error', '-xerror', '-err_detect', 'explode',
                    '-threads', '2', '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-dn',
                    '-progress', 'pipe:1', '-nostats', '-fps_mode', 'passthrough', '-f', 'null', '-'])
    frames = re.findall(rb'^frame=(\d+)', progress, re.M)
    count = int(frames[-1]) if frames else None
    if not count or props['duration_seconds'] <= 0:
        raise ValueError('No decoded video frames or invalid duration')
    if props['frame_count'] is not None and count != props['frame_count']:
        raise ValueError(f'Decoded frame count differs from container: {count} vs {props["frame_count"]}')
    props.update(h)
    props['decoded_frame_count'] = count
    return props


def first_frame(video, png, props):
    cmd = ['ffmpeg', '-hide_banner', '-v', 'error', '-threads', '2', '-i', str(video), '-map', '0:v:0',
           '-an', '-vf', 'select=eq(n\\,0)', '-frames:v', '1', '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24']
    if not png.exists():
        temp = png.with_name('first_frame.part.png')
        run(cmd + ['-y', '-update', '1', str(temp)])
        temp.replace(png)
    with Image.open(png) as im:
        im.load()
        if im.format != 'PNG' or im.size != (props['display_width'], props['display_height']):
            raise ValueError('PNG format or displayed dimensions mismatch')
        # Independently decode frame zero to verify content, not just file existence.
        decoded = run(cmd + ['-f', 'rawvideo', 'pipe:1'])
        if im.convert('RGB').tobytes() != decoded:
            raise ValueError('PNG pixels differ from actual first decoded video frame')
    return {'success': True, 'frame_index': 0, 'lossless_png': True, 'autorotation_applied': True,
            'width': props['display_width'], 'height': props['display_height'],
            'pixel_equality_to_independent_frame_zero_decode': True, 'sha256': hashes(png)['sha256']}


def report_dir(dataset):
    p = DATA_ROOT / dataset / 'metamanifest'
    p.mkdir(parents=True, exist_ok=True)
    return p


def source_artifact_dir(folder):
    base = DATA_ROOT / folder
    return base / 'original' if (base / 'original').is_dir() else base


def source_dir(dataset='Humanoid_Everyday'):
    p = report_dir(dataset) / '_sources'
    p.mkdir(parents=True, exist_ok=True)
    return p


def discover():
    p = source_dir()
    nv = source_dir('NVIDIA_GR1_100')
    if (p / 'discovery.json').exists():
        return read_json(p / 'discovery.json')
    tree = json.loads(fetch(f'https://api.github.com/repos/{GH_REPO}/git/trees/master?recursive=1'))
    if tree.get('truncated'):
        raise ValueError('GitHub tree is truncated')
    gh_rev = tree['sha']
    raw = f'https://raw.githubusercontent.com/{GH_REPO}/{gh_rev}/data_viewer/data_viewer_page.html'
    html = fetch(raw)
    (p / 'gallery_pinned.html').write_text(html, encoding='utf-8')
    tasks = json5.loads(re.search(r'const\s+taskData\s*=\s*({.*?})\s*;', html, re.S).group(1))
    required = {'task_id', 'task_name', 'category', 'text_description', 'third_view_video', 'egocentric_video'}
    if len(tasks) != 49 or any(not required <= t.keys() or k != t['task_id'] for k, t in tasks.items()):
        raise ValueError('Gallery task structure changed; manual review needed')
    write_json(p / 'gallery_tasks.json', tasks)
    write_json(p / 'github_tree_pinned.json', tree)
    info = HfApi().dataset_info(HF_REPO, files_metadata=True)
    hf_files = []
    for f in info.siblings:
        hf_files.append({'path': f.rfilename, 'size': f.size, 'blob_id': f.blob_id,
                         'lfs': dict(f.lfs.__dict__) if f.lfs else None})
    write_json(nv / 'nvidia_files.json', {'revision': info.sha, 'files': hf_files})
    metadata_url = hf_hub_url(HF_REPO, 'metadata.csv', repo_type='dataset', revision=info.sha)
    (nv / 'nvidia_metadata.csv').write_bytes(fetch(metadata_url, True))
    (nv / 'nvidia_README.md').write_bytes(fetch(hf_hub_url(HF_REPO, 'README.md', repo_type='dataset', revision=info.sha), True))
    rows = list(csv.DictReader((nv / 'nvidia_metadata.csv').open(encoding='utf-8', newline='')))
    names = [r['file_name'] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate NVIDIA metadata filenames; cannot map unambiguously')
    available = [f['path'] for f in hf_files if f['path'].lower().endswith('.mp4')]
    basenames = [Path(f).name for f in available]
    if len(set(basenames)) != len(basenames):
        raise ValueError('Duplicate video basenames; cannot map unambiguously')
    d = {'created_at': now(), 'github_revision': gh_rev, 'nvidia_revision': info.sha,
         'gallery_source_url': raw, 'nvidia_metadata_url': metadata_url,
         'humanoid_task_count': len(tasks), 'nvidia_metadata_rows': len(rows), 'nvidia_video_count': len(available),
         'metadata_without_video': sorted(set(names) - set(basenames)),
         'video_without_metadata': sorted(set(basenames) - set(names))}
    write_json(p / 'discovery.json', d)
    write_json(nv / 'discovery.json', d)
    logging.info('Discovered %s', d)
    return d


def slug(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')[:140]


def records():
    d = discover()
    p = source_dir()
    nv = source_dir('NVIDIA_GR1_100')
    tree = {x['path']: x for x in read_json(p / 'github_tree_pinned.json')['tree'] if x['type'] == 'blob'}
    tasks = read_json(p / 'gallery_tasks.json')
    result = []
    for t in tasks.values():
        task_folder = Path('Humanoid_Everyday') / (t['task_id'] + '__' + slug(t['task_name']))
        for field, view in [('third_view_video', 'third_person'), ('egocentric_video', 'first_person')]:
            url = urljoin(GALLERY, t[field])
            repo_path = unquote(urlparse(url).path).lstrip('/')
            blob = tree.get(repo_path)
            result.append({'dataset': 'Humanoid_Everyday', 'original_id': t['task_id'], 'task_id': t['task_id'],
                'original_filename': Path(repo_path).name, 'instruction': t['text_description'], 'task_name': t['task_name'],
                'category': t['category'], 'viewpoint': view, 'folder': str(task_folder / view), 'task_folder': str(task_folder),
                'gallery_asset_url': url, 'download_url': f'https://raw.githubusercontent.com/{GH_REPO}/{d["github_revision"]}/{repo_path}',
                'repository_file_path': repo_path, 'repository': GH_REPO, 'revision': d['github_revision'],
                'metadata_source_url': d['gallery_source_url'], 'official_dataset_url': 'https://humanoideveryday.github.io/',
                'expected': {'size': blob['size'], 'blob_id': blob['sha']} if blob else {}, 'source_record': t,
                'robot_model': None, 'robot_model_evidence': 'Gallery has no per-task robot field; corpus contains Unitree G1 and H1. Filename hints retained without asserting identity.'})
    annotations = {r['file_name']: r['text'] for r in csv.DictReader((nv / 'nvidia_metadata.csv').open(encoding='utf-8', newline=''))}
    for f in sorted(read_json(nv / 'nvidia_files.json')['files'], key=lambda f: f['path']):
        if not f['path'].lower().endswith('.mp4'):
            continue
        name = Path(f['path']).name
        vid = Path(name).stem
        exp = {'size': f['size']}
        if f.get('lfs'):
            exp['sha256'] = f['lfs']['sha256']
        else:
            exp['blob_id'] = f['blob_id']
        result.append({'dataset': 'NVIDIA_GR1_100', 'original_id': vid, 'task_id': vid, 'original_filename': name,
            'instruction': annotations.get(name), 'task_name': None, 'category': None, 'viewpoint': 'third_person',
            'folder': str(Path('NVIDIA_GR1_100') / vid), 'download_url': hf_hub_url(HF_REPO, f['path'], repo_type='dataset', revision=d['nvidia_revision']),
            'repository_file_path': f['path'], 'repository': HF_REPO, 'revision': d['nvidia_revision'],
            'metadata_source_url': d['nvidia_metadata_url'], 'official_dataset_url': f'https://huggingface.co/datasets/{HF_REPO}',
            'expected': exp, 'robot_model': 'Fourier GR1-T2', 'robot_model_evidence': 'Official NVIDIA dataset card'})
    return result


def dropbox_candidates(rec, sheet_rows):
    # Filename normalization only identifies sources to inspect, never establishes episode identity.
    def normalized(s):
        s = Path(unquote(urlparse(s).path)).name.lower().replace('lyft', 'lift')
        s = re.sub(r'(?:_no_sound|_nosound|_no_audio)', '', s)
        s = re.sub(r'_(?:g1|h1)(?=\.mp4$)', '', s)
        return s
    matches = []
    for row in sheet_rows:
        if len(row) < 7:
            continue
        for u in re.findall(r'https://www\.dropbox\.com/[^\s,]+?(?=https://|$)', row[6]):
            if normalized(u) == normalized(rec['original_filename']):
                z = urlparse(u)
                q = dict(parse_qsl(z.query)); q['dl'] = '1'
                matches.append({'url': urlunparse(z._replace(query=urlencode(q))), 'sheet_task_name': row[2],
                                'sheet_robot': row[3], 'sheet_description': row[4]})
    return matches


def refine_quality_pixel_evidence(summary):
    """Compare different containers at decoded native pixel level, not bitrate."""
    p = source_dir()
    frame_hash_cache = {}

    def pixel_hashes(path):
        key = str(path)
        if key not in frame_hash_cache:
            output = run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path),
                          '-map', '0:v:0', '-an', '-pix_fmt', 'rgb24', '-fps_mode', 'passthrough',
                          '-f', 'framemd5', '-']).decode()
            frame_hash_cache[key] = [line.rsplit(',', 1)[-1].strip() for line in output.splitlines()
                                     if line and not line.startswith('#')]
        return frame_hash_cache[key]

    for folder, assessment in summary['assessments'].items():
        task_id = Path(folder).parent.name.split('__')[0]
        view = Path(folder).name
        ref = p / 'quality_candidates' / (task_id + '__' + view + '.mp4')
        if not ref.exists():
            ref = source_artifact_dir(folder) / 'original.mp4'
        for variant in assessment['variants_considered']:
            if variant.get('byte_identical') or 'video_properties' not in variant:
                continue
            if variant['source'].startswith('Publisher'):
                alt = p / 'quality_candidates' / ('dropbox_' + hashlib.sha256(variant['url'].encode()).hexdigest()[:20] + '.mp4')
            else:
                alt = p / 'quality_candidates' / ('alternate_' + variant['git_blob_sha1'] + '.mp4')
            if not (ref.exists() and alt.exists()):
                continue
            ref_hashes, alt_hashes = pixel_hashes(ref), pixel_hashes(alt)
            identical = ref_hashes == alt_hashes
            variant['decoded_pixel_comparison'] = {'same_decoded_pixel_sequence': identical,
                'reference_frame_count': len(ref_hashes), 'alternate_frame_count': len(alt_hashes),
                'method': 'Every displayed RGB frame hashed at native dimensions; compare frame sequence while ignoring container timestamps.'}
            if identical:
                variant['same_recording_verified'] = True
                variant['assessment'] = 'Different container bytes but every native decoded frame is pixel-identical. No genuine pixel-quality improvement.'
            else:
                variant['assessment'] += ' Full native decoded pixel sequences differ; no pixel-quality upgrade established.'
        write_json(p / 'assessments' / (task_id + '__' + view + '.json'), assessment)
        metadata_path = source_artifact_dir(folder) / 'metadata.json'
        if metadata_path.exists():
            m = read_json(metadata_path)
            m['quality'] = assessment
            write_json(metadata_path, m)
    write_json(p / 'quality_research.json', summary)
    return summary


def research(workers):
    d = discover()
    p = source_dir()
    # Retrieve only metadata about raw/LeRobot sources; never download the full corpus.
    if not (p / 'he_task_download_index.csv').exists():
        (p / 'he_task_download_index.csv').write_bytes(fetch(SHEET, True))
    he_info = HfApi().dataset_info(HE_HF)
    for remote, local in [('README.md', 'he_hf_card.md'), ('meta/info.json', 'he_hf_info.json'), ('meta/episodes.jsonl', 'he_hf_episodes.jsonl')]:
        (p / local).write_bytes(fetch(hf_hub_url(HE_HF, remote, repo_type='dataset', revision=he_info.sha), True))
    code_url = 'https://raw.githubusercontent.com/physical-superintelligence-lab/Humanoid-Everyday/master/README.md'
    (p / 'he_code_readme.md').write_bytes(fetch(code_url, True))
    sheet_rows = list(csv.reader((p / 'he_task_download_index.csv').open(encoding='utf-8', newline='')))[3:]
    tree = [x for x in read_json(p / 'github_tree_pinned.json')['tree'] if x['path'].endswith('.mp4')]
    he = [r for r in records() if r['dataset'] == 'Humanoid_Everyday']
    assessments = {}

    def assess(r):
        # Fetch each gallery reference into a reusable source cache for native ffprobe assessment.
        cache = p / 'quality_candidates' / (r['task_id'] + '__' + r['viewpoint'] + '.mp4')
        download(r['download_url'], cache, r['expected'])
        props = probe(cache)
        variants = []
        for x in tree:
            if Path(x['path']).name == r['original_filename'] and x['path'] != r['repository_file_path']:
                v = {'source': 'official GitHub alternate asset', 'repository_file_path': x['path'], 'size': x['size'], 'git_blob_sha1': x['sha']}
                if x['sha'] == r['expected'].get('blob_id'):
                    v.update({'same_recording_verified': True, 'byte_identical': True, 'video_properties': props,
                              'assessment': 'Identical Git blob: same bytes, pixels and quality.'})
                else:
                    other = p / 'quality_candidates' / ('alternate_' + x['sha'] + '.mp4')
                    url = f'https://raw.githubusercontent.com/{GH_REPO}/{d["github_revision"]}/{x["path"]}'
                    download(url, other, {'size': x['size'], 'blob_id': x['sha']})
                    v.update({'url': url, 'same_recording_verified': False, 'byte_identical': False,
                              'video_properties': probe(other), 'sha256': hashes(other)['sha256'],
                              'assessment': 'Different bytes; filename alone does not verify recording identity or native pixel quality. Not substituted.'})
                variants.append(v)
        if r['viewpoint'] == 'third_person':
            for candidate in dropbox_candidates(r, sheet_rows):
                url = candidate['url']
                out = p / 'quality_candidates' / ('dropbox_' + hashlib.sha256(url.encode()).hexdigest()[:20] + '.mp4')
                v = dict(candidate, source='Publisher-linked Dropbox MP4')
                try:
                    download(url, out)
                    v['video_properties'] = probe(out)
                    v['sha256'] = hashes(out)['sha256']
                    identical = v['sha256'] == hashes(cache)['sha256']
                    v.update({'byte_identical': identical, 'same_recording_verified': identical,
                              'assessment': 'Byte-identical to gallery MP4; no quality upgrade.' if identical else
                              'Different file; cannot establish exact recording match from filename alone. Kept for review, not substituted.'})
                    if identical:
                        out.unlink()  # assessment retained; avoid redundant duplicate bytes
                except Exception as e:
                    v.update({'available': False, 'same_recording_verified': False, 'assessment': str(e)})
                variants.append(v)
        a = {'selected_quality_source': 'Untouched original official gallery MP4 at pinned GitHub revision',
             'selected_video_properties': props, 'gallery_asset_url': r['gallery_asset_url'],
             'variants_considered': variants, 'quality_upgrade_verified': False,
             'selection_explanation': 'Highest native pixel quality identifiable for this exact gallery recording among inspected published assets. Identical copies provide no upgrade. No verified gallery-to-raw/LeRobot episode mapping; no bitrate-only quality inference.',
             'raw_option': {'available_in_published_corpus': True, 'format': 'lossless 640x480 RGB PNG image sequences in task archives',
                            'exact_gallery_recording_match_verified': False, 'downloaded': False,
                            'source_url': code_url, 'task_archive_index_url': SHEET},
             'lerobot_option': {'repository': HE_HF, 'revision': he_info.sha, 'published_resolution': [640, 480],
                                'published_fps': 30, 'exact_gallery_recording_match_verified': False,
                                'evidence': 'Episodes use numeric episode_index and instructions; gallery has task IDs and MP4 names, without episode IDs. Task similarity alone cannot map a demonstration.'}}
        write_json(p / 'assessments' / (r['task_id'] + '__' + r['viewpoint'] + '.json'), a)
        logging.info('Quality assessed %s %s: %dx%d, %.4f fps, %d alternatives', r['task_id'], r['viewpoint'], props['width'], props['height'], props['fps'], len(variants))
        return r['folder'], a

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for folder, a in pool.map(assess, he):
            assessments[folder] = a
    summary = {'completed_at': now(), 'github_revision': d['github_revision'], 'he_hf_revision': he_info.sha,
        'github_mp4_assets_inspected': len(tree), 'gallery_video_sources_assessed': len(assessments),
        'quality_upgrades_verified': 0, 'raw_images': 'Published lossless 640x480 PNG; no exact gallery recording mapping established.',
        'license': {'gallery_repository': None, 'published_hf_dataset': 'Apache-2.0', 'code_readme_dataset_claim': 'MIT',
                    'limitation': 'Official sources disagree; gallery repository does not specify an asset license. Preserve both claims; do not infer one applies to gallery videos.'},
        'assessments': assessments}
    return refine_quality_pixel_evidence(summary)


def quality_for(r):
    if r['dataset'] == 'Humanoid_Everyday':
        p = source_dir() / 'assessments' / (r['task_id'] + '__' + r['viewpoint'] + '.json')
        if not p.exists():
            raise RuntimeError('Quality research required before downloading gallery datasets')
        return read_json(p)
    return {'selected_quality_source': 'Original MP4 in official NVIDIA Hugging Face gr1/ repository directory',
            'variants_considered': [{'source': 'Hugging Face viewer previews', 'selected': False, 'assessment': 'Derived viewer representations are not original assets.'}],
            'quality_upgrade_verified': False, 'selection_explanation': 'Original repository MP4 at pinned revision with publisher LFS SHA-256 verification; no re-encoding, upscaling, or interpolation.'}


def issues_for(r):
    issues = []
    if r['dataset'] == 'Humanoid_Everyday':
        if r['task_id'] == 'human-robot-4' and r['viewpoint'] == 'first_person':
            issues.append('Original gallery book-handover annotation references a stack-of-plates egocentric filename. Potential task/video mismatch; original association preserved.')
        if r['task_id'] in ('pick-place-7', 'high-precision-8'):
            issues.append('Same third-person Git blob appears under two gallery task IDs; egocentric files differ. Cross-task duplicate and pairing caution; no synchronization verified.')
        if 'lyft' in r['original_filename']:
            issues.append('Original filename spells lyft; task_name spells Lift. Preserved without correction.')
    return issues


def process_record(r):
    folder = source_artifact_dir(r['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / 'original.mp4'
    quality = quality_for(r)
    prior = read_json(folder / 'metadata.json') if (folder / 'metadata.json').exists() else {}
    reused_cache = False
    cache = source_dir() / 'quality_candidates' / (r['task_id'] + '__' + r['viewpoint'] + '.mp4')
    if r['dataset'] == 'Humanoid_Everyday' and not video.exists() and cache.exists():
        check_identity(cache, r['expected'])
        cache.replace(video)
        reused_cache = True
    transport = download(r['download_url'], video, r['expected'])
    if prior.get('download_timestamp'):
        transport['download_timestamp'] = prior['download_timestamp']
    # Skip full repeat decode only after byte checksum matches previously validated file.
    current = check_identity(video, r['expected'])
    if prior.get('validation', {}).get('full_decode_success') and prior.get('video_properties', {}).get('sha256') == current['sha256']:
        props = prior['video_properties']
        validation = dict(prior['validation'], reused_after_checksum_validation=True)
    else:
        props = validate(video, r['expected'])
        validation = {'download_success': True, 'source_integrity_success': True, 'ffprobe_success': True,
                      'full_decode_success': True, 'validated_at': now(), 'decoded_frame_count': props['decoded_frame_count'],
                      'source_integrity_evidence': r['expected'], 'reused_after_checksum_validation': False}
    frame = first_frame(video, folder / 'first_frame.png', props)
    if r['instruction'] is not None:
        (folder / 'task_prompt.txt').write_bytes(r['instruction'].encode('utf-8'))
    if r.get('task_folder'):
        task_folder = DATA_ROOT / r['task_folder']
        (task_folder / 'task_prompt.txt').write_bytes(r['instruction'].encode('utf-8'))
        write_json(task_folder / 'task_metadata.json', {'dataset': r['dataset'], 'task_id': r['task_id'],
            'task_name': r['task_name'], 'category': r['category'], 'text_description': r['instruction'],
            'original_gallery_record': r['source_record'], 'official_gallery_url': GALLERY,
            'metadata_source_url': r['metadata_source_url'], 'repository': r['repository'], 'revision': r['revision'],
            'associated_viewpoints': {'third_person': r['source_record']['third_view_video'], 'first_person': r['source_record']['egocentric_video']},
            'official_gallery_associates_views': True, 'same_physical_demonstration_independently_verified': False,
            'temporal_synchronization_verified': False,
            'annotation_discrepancies': sorted(set(issues_for(dict(r, viewpoint='first_person')) + issues_for(dict(r, viewpoint='third_person'))))})
    metadata = {k: v for k, v in r.items() if k not in ('expected', 'source_record', 'folder', 'task_folder')}
    metadata.update({'exact_task_instruction': r['instruction'], 'official_gallery_url': GALLERY if r['dataset'] == 'Humanoid_Everyday' else None,
        'viewpoint_evidence': 'Official gallery labels this viewpoint' if r['dataset'] == 'Humanoid_Everyday' else
                             'Official NVIDIA dataset card describes third person perspective; physical camera placement is not independently verified.',
        'download_timestamp': transport['download_timestamp'], 'download_transport': transport,
        'quality_assessment_cache_used': reused_cache, 'video_properties': props,
        'published_dataset_license': 'CC-BY-4.0' if r['dataset'] == 'NVIDIA_GR1_100' else read_json(source_dir() / 'quality_research.json')['license'],
        'quality': quality, 'validation': validation, 'first_frame': frame,
        'task_video_mapping': {'confidence': 'exact original annotation lookup',
            'evidence': 'Same taskData entry; original description and view URL fields' if r['dataset'] == 'Humanoid_Everyday' else 'CSV file_name exactly equals repository MP4 basename; never row order',
            'annotation_available': r['instruction'] is not None, 'semantic_content_verified': False},
        'annotation_discrepancies': issues_for(r),
        'official_gallery_associates_views': r['dataset'] == 'Humanoid_Everyday',
        'same_physical_demonstration_independently_verified': False if r['dataset'] == 'Humanoid_Everyday' else None,
        'temporal_synchronization_verified': False if r['dataset'] == 'Humanoid_Everyday' else None})
    write_json(folder / 'metadata.json', metadata)
    logging.info('Validated %s/%s %s %dx%d frames=%d', r['dataset'], r['original_id'], r['viewpoint'], props['width'], props['height'], props['decoded_frame_count'])
    return metadata


def write_csv(path, rows, fields):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def execute(phase, workers):
    all_records = records()
    if phase == 'pilot':
        he_ids = list(dict.fromkeys(r['task_id'] for r in all_records if r['dataset'] == 'Humanoid_Everyday'))[:2]
        nv = sorted((r for r in all_records if r['dataset'] == 'NVIDIA_GR1_100'), key=lambda r: int(r['original_id']))[:2]
        selected = [r for r in all_records if r['dataset'] == 'Humanoid_Everyday' and r['task_id'] in he_ids] + nv
    else:
        selected = all_records
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_records = {pool.submit(process_record, r): r for r in selected}
        for future in concurrent.futures.as_completed(future_records):
            r = future_records[future]
            try:
                future.result()
            except Exception as e:
                logging.exception('Failed %s', r['folder'])
                failures.append({'dataset': r['dataset'], 'task_id': r['task_id'], 'viewpoint': r['viewpoint'], 'url': r['download_url'], 'error': str(e), 'timestamp': now()})
    for dataset in DATASETS:
        count = sum(r['dataset'] == dataset for r in selected)
        dataset_failures = [f for f in failures if f['dataset'] == dataset]
        write_json(source_dir(dataset) / (phase + '_run.json'), {'timestamp': now(), 'dataset': dataset,
            'selected': count, 'success': count-len(dataset_failures), 'failures': dataset_failures})
    audit(pilot=(phase == 'pilot'))
    if failures:
        raise RuntimeError(f'{len(failures)} videos failed; details retained in audit and log')


def audit(pilot=False):
    d = discover()
    all_records = records()
    manifests = collections.defaultdict(list)
    errors = []
    quality_rows = []
    issues = []
    completed_tasks = collections.defaultdict(set)
    hashes_to_records = collections.defaultdict(list)
    for r in all_records:
        folder = source_artifact_dir(r['folder'])
        if not (folder / 'metadata.json').exists():
            if not pilot:
                errors.append({'dataset': r['dataset'], 'task_id': r['task_id'], 'viewpoint': r['viewpoint'], 'url': r['download_url'], 'error': 'Video/metadata not successfully completed', 'timestamp': now()})
            continue
        try:
            m = read_json(folder / 'metadata.json')
            h = check_identity(folder / 'original.mp4', r['expected'])
            if h['sha256'] != m['video_properties']['sha256']:
                raise ValueError('Metadata checksum mismatch')
            if r['instruction'] is not None and (folder / 'task_prompt.txt').read_bytes() != r['instruction'].encode('utf-8'):
                raise ValueError('Prompt bytes differ from original field')
            if m['exact_task_instruction'] != r['instruction'] or m['repository_file_path'] != r['repository_file_path']:
                raise ValueError('Metadata instruction/source identity mismatch')
            if r.get('task_folder'):
                t = DATA_ROOT / r['task_folder']
                if (t / 'task_prompt.txt').read_bytes() != r['instruction'].encode('utf-8'):
                    raise ValueError('Task-level prompt differs from original field')
                if read_json(t / 'task_metadata.json')['original_gallery_record'] != r['source_record']:
                    raise ValueError('Task-level original entry differs from source')
            props = probe(folder / 'original.mp4')
            if (props['width'], props['height']) != (m['video_properties']['width'], m['video_properties']['height']):
                raise ValueError('Video dimensions differ from metadata')
            first_frame(folder / 'original.mp4', folder / 'first_frame.png', props)
            if not m['validation']['full_decode_success']:
                raise ValueError('Missing full decode validation')
            row = {'dataset': r['dataset'], 'task_id': r['task_id'], 'original_video_id': r['original_id'],
                'task_name': r['task_name'], 'category': r['category'], 'viewpoint': r['viewpoint'], 'instruction': r['instruction'],
                'relative_file_path': Path(os.path.relpath(folder / 'original.mp4', report_dir(r['dataset']))).as_posix(), 'original_url': r['download_url'],
                'original_filename': r['original_filename'], 'width': props['width'], 'height': props['height'],
                'display_width': props['display_width'], 'display_height': props['display_height'], 'fps': props['fps'],
                'duration_seconds': props['duration_seconds'], 'codec': props['codec'], 'pixel_format': props['pixel_format'],
                'video_bitrate': props['video_bitrate'], 'frame_count': m['video_properties']['decoded_frame_count'],
                'file_size_bytes': h['file_size_bytes'], 'sha256': h['sha256'], 'validation_status': 'passed',
                'first_frame_status': 'passed', 'annotation_discrepancies': ' | '.join(m['annotation_discrepancies']),
                'temporal_synchronization_verified': m['temporal_synchronization_verified']}
            manifests[r['dataset']].append(row)
            hashes_to_records[h['sha256']].append(row['relative_file_path'])
            if r['dataset'] == 'Humanoid_Everyday':
                completed_tasks[r['task_id']].add(r['viewpoint'])
            for issue in m['annotation_discrepancies']:
                issues.append({'dataset': r['dataset'], 'task_id': r['task_id'], 'viewpoint': r['viewpoint'], 'issue': issue})
            q = m['quality']
            quality_rows.append({**row, 'selected_quality_source': q['selected_quality_source'],
                'quality_upgrade_verified': q['quality_upgrade_verified'], 'selection_explanation': q['selection_explanation'],
                'variants_considered': json.dumps(q['variants_considered'], ensure_ascii=False),
                'higher_quality_raw_exact_match_verified': q.get('raw_option', {}).get('exact_gallery_recording_match_verified')})
        except Exception as e:
            errors.append({'dataset': r['dataset'], 'task_id': r['task_id'], 'viewpoint': r['viewpoint'], 'url': r['download_url'], 'error': str(e), 'timestamp': now()})
    for name in d['metadata_without_video']:
        errors.append({'dataset': 'NVIDIA_GR1_100', 'task_id': Path(name).stem, 'viewpoint': 'third_person', 'url': '', 'error': 'Original metadata row exists but no repository MP4: ' + name, 'timestamp': now()})
    for name in d['video_without_metadata']:
        errors.append({'dataset': 'NVIDIA_GR1_100', 'task_id': Path(name).stem, 'viewpoint': 'third_person', 'url': '', 'error': 'Repository video lacks original annotation: ' + name, 'timestamp': now()})
    fields = ['dataset', 'task_id', 'original_video_id', 'task_name', 'category', 'viewpoint', 'instruction', 'relative_file_path', 'original_url', 'original_filename', 'width', 'height', 'display_width', 'display_height', 'fps', 'duration_seconds', 'codec', 'pixel_format', 'video_bitrate', 'frame_count', 'file_size_bytes', 'sha256', 'validation_status', 'first_frame_status', 'annotation_discrepancies', 'temporal_synchronization_verified']
    timestamp = now()
    summaries = {}
    manifest_names = {'Humanoid_Everyday': 'humanoid_everyday_manifest.csv',
                      'NVIDIA_GR1_100': 'nvidia_gr1_100_manifest.csv'}
    for dataset in DATASETS:
        reports = report_dir(dataset)
        rows = manifests[dataset]
        dataset_errors = [e for e in errors if e['dataset'] == dataset]
        dataset_issues = [i for i in issues if i['dataset'] == dataset]
        write_csv(reports / manifest_names[dataset], rows, fields)
        write_csv(reports / 'download_failures.csv', dataset_errors,
                  ['dataset', 'task_id', 'viewpoint', 'url', 'error', 'timestamp'])
        write_csv(reports / 'quality_audit.csv', [q for q in quality_rows if q['dataset'] == dataset],
                  fields + ['selected_quality_source', 'quality_upgrade_verified', 'selection_explanation',
                            'variants_considered', 'higher_quality_raw_exact_match_verified'])
        write_csv(reports / 'annotation_discrepancies.csv', dataset_issues,
                  ['dataset', 'task_id', 'viewpoint', 'issue'])
        local_hash_groups = collections.defaultdict(list)
        for row in rows:
            local_hash_groups[row['sha256']].append(row['relative_file_path'])
        quality_summary = {
            'native_resolutions': dict(collections.Counter(f'{r["width"]}x{r["height"]}' for r in rows)),
            'displayed_resolutions': dict(collections.Counter(f'{r["display_width"]}x{r["display_height"]}' for r in rows)),
            'frame_rates': dict(collections.Counter(str(r['fps']) for r in rows)),
            'codecs': dict(collections.Counter(r['codec'] for r in rows))}
        limitations = ['No quality upgrade verified; source-native resolutions retained.',
                       'Semantic instruction correctness is not certified; original annotations are preserved.']
        if dataset == 'Humanoid_Everyday':
            limitations += [
                'Raw lossless PNGs exist in the larger Humanoid corpus but no exact gallery episode correspondence verified.',
                'Views are associated by the original gallery; physical episode equivalence and temporal synchronization are unverified.',
                'Published license claims conflict (HF Apache-2.0 versus code README MIT); gallery MP4 asset license is not specified.']
        summary = {'dataset': dataset, 'audited_at': timestamp, 'pilot_only': pilot,
            'output_directory': str(DATA_ROOT / dataset), 'reports_directory': str(reports),
            'manifest_paths_relative_to': str(reports), 'video_count': len(rows),
            'task_count': len({r['task_id'] for r in rows}),
            'download_or_validation_failures': [e for e in dataset_errors
                if not e['error'].startswith(('Original metadata row', 'Repository video lacks'))],
            'annotation_discrepancies': dataset_issues,
            'byte_identical_video_groups': [v for v in local_hash_groups.values() if len(v) > 1],
            'original_video_storage_bytes': sum(r['file_size_bytes'] for r in rows),
            'quality_summary': quality_summary, 'limitations': limitations,
            'total_output_storage_bytes': sum(f.stat().st_size for f in (DATA_ROOT / dataset).rglob('*') if f.is_file())}
        if dataset == 'Humanoid_Everyday':
            summary.update({'humanoid_tasks_with_both_views': sum(v == {'first_person', 'third_person'} for v in completed_tasks.values()),
                            'humanoid_tasks_with_any_view': len(completed_tasks), 'humanoid_videos': len(rows)})
            source = GALLERY
            revision = d['github_revision']
            metadata_url = d['gallery_source_url']
            layout = '`<official-task-id>__<sanitized-task-name>/` contains the task prompt and task metadata, plus `third_person/` and `first_person/`. Reviewed third-person clips store unchanged source artifacts under `third_person/original/` and full generation references plus visually authored action prompts under `third_person/generation prompt/`. First-person source artifacts remain intact in `first_person/`. The downloader supports both direct and original/ source layouts.'
            license_text = 'The Humanoid HF card says Apache-2.0 and the official code README says MIT; the gallery repository has no explicit video asset license. Both claims are retained without inferring that either applies to the gallery MP4s.'
            mapping = 'Prompts use the exact text_description field. Both viewpoints are associated in the same original taskData entry. Each view has its own unchanged copy of the instruction.'
            count_details = f"Tasks with both views: {summary['humanoid_tasks_with_both_views']}."
            quality_text = 'Official GitHub alternatives and publisher-linked Dropbox MP4s were inspected with native ffprobe properties, source hashes and decoded pixel comparisons. Identical files or decoded frames provide no pixel-quality upgrade. Different files are not substituted on filename or task similarity. Published raw lossless 640x480 RGB PNG sequences and LeRobot episode metadata were investigated, but no exact gallery-to-episode mapping was verified. The enormous full corpus was not downloaded. The book/plates annotation conflict and duplicate paper-towel third-person video are preserved and flagged in annotation_discrepancies.csv.'
        else:
            summary.update({'nvidia_videos': len(rows), 'nvidia_metadata_rows': d['nvidia_metadata_rows'],
                            'nvidia_metadata_without_video': d['metadata_without_video'],
                            'nvidia_video_without_metadata': d['video_without_metadata']})
            source = f'https://huggingface.co/datasets/{HF_REPO}'
            revision = d['nvidia_revision']
            metadata_url = d['nvidia_metadata_url']
            layout = '`<original-video-id>/original/` contains original.mp4, task_prompt.txt, first_frame.png and metadata.json. The sibling `generation prompt/` contains the complete unchanged generation video, first frame, visually authored actiongeneration_prompt.txt, action_review.json and generation metadata. IDs remain unchanged, including numbering gaps. The downloader supports both direct and original/ source layouts.'
            license_text = 'The official NVIDIA dataset license is CC-BY-4.0.'
            mapping = 'Prompts use the exact text field whose file_name equals the original repository MP4 basename. CSV row order is never used for matching.'
            count_details = f"Metadata rows: {d['nvidia_metadata_rows']}. Metadata without repository video: {', '.join(d['metadata_without_video']) or 'none'}. Videos without instructions: {', '.join(d['video_without_metadata']) or 'none'}."
            quality_text = 'All selected files are original MP4 assets in the official gr1/ repository at the pinned revision. Hugging Face viewer previews were excluded. File sizes and publisher LFS SHA-256 hashes are verified. The published camera viewpoint is attributed to the dataset card; physical camera placement has not been independently calibrated.'
        write_json(reports / 'final_report.json', summary)
        readme = f"""# {dataset}

This dataset is a first-level project folder. Its manifests, audits and provenance records are under `metamanifest/`.
Audit: {timestamp}. {'Pilot only.' if pilot else 'Complete downloaded dataset audit.'}

## Counts and layout

Downloaded and validated MP4s: {len(rows)}. {count_details}
Download or validation failures: {len(summary['download_or_validation_failures'])}.
Original MP4 storage: {summary['original_video_storage_bytes']:,} bytes.
Dataset and metadata logical storage at audit: {summary['total_output_storage_bytes']:,} bytes.

{layout}

`metamanifest/{manifest_names[dataset]}` has one row per validated downloaded video. Each relative_file_path resolves relative to `metamanifest/` and points to that video's unchanged original MP4. `quality_audit.csv`, `download_failures.csv` and `annotation_discrepancies.csv` contain only this dataset's rows. `final_report.json` contains this dataset's counts, storage, quality distribution and limitations. Source snapshots, inventories, assessments and execution evidence are under `_sources/`.

## Original instructions and provenance

{mapping} Original `task_prompt.txt` files contain only the original field encoded in UTF-8, without an added newline. Original author spelling, punctuation and wording are retained.

Official source: {source}
Pinned revision: `{revision}`
Original metadata source: {metadata_url}

{license_text}

## Humanoid world-model action prompts

Every reviewed third-person Humanoid clip and every available GR1 clip has a dedicated `generation prompt/` folder. `actiongeneration_prompt.txt` is the visually authored `humanoid_action_v1` instruction to pair with `first_frame.png`; it is the sole generation prompt file. `video.mp4` retains the complete original bytes. `action_review.json` records observed hand roles, coordination, terminal state, inspection coverage and uncertainties. Original task annotations remain under `original/`.

`world_model_prompts.csv` indexes completed prompts, and `world_model_prompt_report.json` records completed and pending reviews. `bimanual_second_review_report.json` records the focused second pass on coordinated or potentially coordinated hand actions, including prompt corrections and detailed contact evidence. The project `AGENTS.md` and this folder's `REVIEW_WORKFLOW.md` define the protocol. No cropping or shortening is used for these single-task references.

## Quality and verification

{quality_text}

Published MP4 bytes remain untouched. No recompression, upscaling, interpolation, cropping, trimming or normalization is performed. Higher bitrate alone is not used to rank pixel quality. No higher-quality exact recording variant was verified, so no quality upgrade is claimed. Each successful MP4 passed source checksum/size checks, ffprobe and full FFmpeg decoding. The lossless PNG is decoded frame zero with displayed orientation retained, native dimensions checked and pixels verified against an independent frame-zero decode. Semantic instruction correctness and cross-view synchronization are not assumed.

```json
{json.dumps(quality_summary, indent=2)}
```

## Reproduce or resume

From the project root, install FFmpeg/ffprobe (`brew install ffmpeg` on macOS), create `.venv` with `python3 -m venv .venv`, and install `requirements-robot-datasets.txt`. The shared downloader manages both datasets and stores each one's outputs under its own metamanifest:

```
.venv/bin/python -m pip install -r requirements-robot-datasets.txt
.venv/bin/python download_robot_datasets.py --phase discover
.venv/bin/python download_robot_datasets.py --phase research --workers 3
.venv/bin/python download_robot_datasets.py --phase pilot --workers 3
.venv/bin/python download_robot_datasets.py --phase full --workers 3
.venv/bin/python download_robot_datasets.py --phase audit
```

Use `--output <project-directory>` for another project root. Discovery pins revisions for safe resumes; use a new output directory for independent upstream versions. Downloads stream to temporary .part files with range/entity validation, exponential retry and atomic completion. Validated originals are skipped after checksum verification; corrupt originals are quarantined. Each dataset has its own `_sources/pilot_run.json` and `_sources/full_run.json`. Historical shared logs are retained as execution evidence.
"""
        (reports / 'README.md').write_text(readme, encoding='utf-8')
        summaries[dataset] = summary
        logging.info('AUDIT %s videos=%d failures=%d', dataset, len(rows), len(summary['download_or_validation_failures']))
    return summaries


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['discover', 'research', 'pilot', 'full', 'audit'], required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, default=DATA_ROOT, help='Project directory containing each dataset and its metamanifest')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error('Use 1-8 parallel workers; default 3')
    DATA_ROOT = args.output.resolve()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for exe in ['ffmpeg', 'ffprobe']:
        if not shutil.which(exe):
            raise SystemExit(f'{exe} is missing. On macOS run: brew install ffmpeg')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[*(logging.FileHandler(source_dir(dataset) / 'pipeline.log') for dataset in DATASETS), logging.StreamHandler()])
    if args.phase == 'discover':
        discover()
    elif args.phase == 'research':
        research(args.workers)
    elif args.phase in ('pilot', 'full'):
        execute(args.phase, args.workers)
    else:
        audit()


if __name__ == '__main__':
    main()
