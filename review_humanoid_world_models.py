#!/usr/bin/env python3
"""Prepare complete source-video inspection and validate visually authored prompts.

This utility never authors actions. A dedicated review agent must inspect each video.
Temporary inspection sheets are outside dataset deliverables. Original videos are
copied byte-for-byte into complete world-model references, never transcoded.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parent
JOBS_PATH = PROJECT / 'humanoid_world_model_review_jobs.json'
CONSTRAINT = 'Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.'
TEMP_ROOT = Path(tempfile.gettempdir()) / 'robotvideo_humanoid_action_review'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, timeout=600)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-4000:])
    return p.stdout


def jobs():
    return json.loads(JOBS_PATH.read_text())['jobs']


def init():
    import download_robot_datasets as downloader
    records = [r for r in downloader.records() if r['dataset'] == 'NVIDIA_GR1_100' or r['viewpoint'] == 'third_person']
    result = []
    for r in records:
        base = PROJECT / r['folder']
        original = base / 'original'
        original.mkdir(exist_ok=True)
        for name in ['original.mp4', 'task_prompt.txt', 'metadata.json']:
            source = base / name
            destination = original / name
            if source.exists():
                if destination.exists():
                    if sha(source) != sha(destination):
                        raise RuntimeError('Conflicting source artifacts: ' + str(source))
                    source.unlink()
                else:
                    source.replace(destination)
            if not destination.exists():
                raise RuntimeError('Missing original artifact: ' + str(destination))
        metadata = json.loads((original / 'metadata.json').read_text())
        assert sha(original / 'original.mp4') == metadata['video_properties']['sha256']
        key = ('he_' + r['task_id']) if r['dataset'] == 'Humanoid_Everyday' else ('gr1_' + r['original_id'])
        world = base / 'generation input'
        world.mkdir(exist_ok=True)
        image = world / 'first_frame.png'
        for prior_image in [base / 'first_frame.png', original / 'first_frame.png']:
            if prior_image.exists():
                if image.exists():
                    if sha(prior_image) != sha(image):
                        raise RuntimeError('Conflicting first frames: ' + str(prior_image))
                    prior_image.unlink()
                else:
                    prior_image.replace(image)
        downloader.first_frame(original / 'original.mp4', image, metadata['video_properties'])
        if sha(image) != metadata['first_frame']['sha256']:
            raise RuntimeError('First frame differs from source metadata: ' + str(image))
        for source, target in [('original.mp4', 'video.mp4')]:
            src, dst = original / source, world / target
            if dst.exists():
                if sha(src) != sha(dst):
                    raise RuntimeError('Existing generation reference differs: ' + str(dst))
            else:
                shutil.copy2(src, dst)
        result.append({'key': key, 'dataset': r['dataset'], 'original_id': r['original_id'],
            'task_name': r['task_name'], 'base': str(base.relative_to(PROJECT)),
            'source_video': str((original / 'original.mp4').relative_to(PROJECT)),
            'source_first_frame': str(image.relative_to(PROJECT)),
            'source_metadata': str((original / 'metadata.json').relative_to(PROJECT)),
            'original_task_prompt': r['instruction'], 'frame_count': metadata['video_properties']['decoded_frame_count'],
            'fps': metadata['video_properties']['fps'], 'duration_seconds': metadata['video_properties']['duration_seconds'],
            'source_video_sha256': metadata['video_properties']['sha256'], 'world_model': str(world.relative_to(PROJECT)),
            'inspection_directory': str(TEMP_ROOT / key), 'status': 'pending'})
    write_json(JOBS_PATH, {'protocol': 'humanoid_action_v1', 'constraint_sentence': CONSTRAINT, 'jobs': result})
    print(json.dumps({'jobs': len(result), 'humanoid_third_person': sum(r['dataset'] == 'Humanoid_Everyday' for r in result),
                      'nvidia': sum(r['dataset'] == 'NVIDIA_GR1_100' for r in result)}, indent=2))


def get_job(key):
    return next(j for j in jobs() if j['key'] == key)


def prepare(key):
    job = get_job(key)
    directory = Path(job['inspection_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'inspection.json'
    if manifest_path.exists():
        inspection = json.loads(manifest_path.read_text())
        inspection.update({'world_model_directory': str(PROJECT / job['world_model']),
                           'source_first_frame': str(PROJECT / job['source_first_frame']),
                           'original_task_instruction': job['original_task_prompt']})
        print(json.dumps(inspection, indent=2))
        return
    video = PROJECT / job['source_video']
    props = json.loads((PROJECT / job['source_metadata']).read_text())['video_properties']
    dw, dh = props['display_width'], props['display_height']
    tw = 320 if dw >= dh else 200
    th = max(1, round(tw * dh / dw))
    if th > 320:
        th = 320
        tw = max(1, round(th * dw / dh))
    # Thumbnails are exclusively for inspection. Published media and generation
    # references remain untouched, and native transition frames are read separately.
    raw = run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video), '-map', '0:v:0',
               '-vf', f'select=not(mod(n\\,2)),scale={tw}:{th}', '-fps_mode', 'passthrough',
               '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
    frame_bytes = tw * th * 3
    sample_count = len(raw) // frame_bytes
    expected = math.ceil(job['frame_count'] / 2)
    if len(raw) % frame_bytes or sample_count != expected:
        raise RuntimeError(f'Incomplete sample coverage: {sample_count} vs {expected}')
    sheets = []
    frames_per_sheet = 64
    cols = 8
    cell_w, cell_h = tw + 8, th + 24
    for first in range(0, sample_count, frames_per_sheet):
        count = min(frames_per_sheet, sample_count - first)
        rows = math.ceil(count / cols)
        board = Image.new('RGB', (cols * cell_w, rows * cell_h), '#eeeeee')
        draw = ImageDraw.Draw(board)
        for i in range(count):
            sampled_index = first + i
            frame = Image.frombytes('RGB', (tw, th), raw[sampled_index*frame_bytes:(sampled_index+1)*frame_bytes])
            x, y = (i % cols) * cell_w + 4, (i // cols) * cell_h
            draw.text((x, y + 3), f'f{sampled_index*2}  {sampled_index*2/job["fps"]:.2f}s', fill='black')
            board.paste(frame, (x, y + 22))
        name = f'sheet_{first//frames_per_sheet+1:03d}.jpg'
        board.save(directory / name, quality=93)
        sheets.append({'path': str(directory / name), 'first_frame': first*2, 'last_frame': (first+count-1)*2, 'sample_count': count})
    inspection = {'key': key, 'source_video': str(video), 'source_first_frame': str(PROJECT / job['source_first_frame']),
                  'world_model_directory': str(PROJECT / job['world_model']), 'original_task_instruction': job['original_task_prompt'],
                  'frame_count': job['frame_count'], 'fps': job['fps'], 'frame_stride': 2,
                  'sample_count': sample_count, 'sheets': sheets,
                  'native_dimensions': [dw, dh], 'native_detail_frames': []}
    write_json(manifest_path, inspection)
    print(json.dumps(inspection, indent=2))


def native(key, indices):
    job = get_job(key)
    directory = Path(job['inspection_directory'])
    manifest_path = directory / 'inspection.json'
    inspection = json.loads(manifest_path.read_text())
    paths = []
    for frame in indices:
        if frame < 0 or frame >= job['frame_count']:
            raise ValueError('Frame outside source sequence')
        path = directory / f'native_{frame:06d}.png'
        if not path.exists():
            run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(PROJECT / job['source_video']),
                 '-vf', f'select=eq(n\\,{frame})', '-frames:v', '1', '-fps_mode', 'passthrough',
                 '-pix_fmt', 'rgb24', '-update', '1', '-y', str(path)])
        paths.append(str(path))
    inspection['native_detail_frames'] = sorted(set(inspection['native_detail_frames'] + indices))
    write_json(manifest_path, inspection)
    print(json.dumps({'native_frames': paths}, indent=2))


def finalize(key):
    job = get_job(key)
    world = PROJECT / job['world_model']
    review = json.loads((world / 'action_review.json').read_text())
    prompt_path = world / 'actiongeneration_prompt.txt'
    prompt = prompt_path.read_text().strip()
    inspection = json.loads((Path(job['inspection_directory']) / 'inspection.json').read_text())
    if not prompt.endswith(CONSTRAINT) or len(prompt.split()) < 26:
        raise ValueError('Missing complete prompt or protocol constraint')
    if review.get('prompt_protocol') != 'humanoid_action_v1' or review.get('prompt_provenance') != 'visually_authored_from_video_and_initial_frame':
        raise ValueError('Wrong review protocol or provenance')
    expected_sheets = {Path(s['path']).name for s in inspection['sheets']}
    if set(review.get('reviewed_contact_sheets', [])) != expected_sheets:
        raise ValueError('Full-video sheet inspection coverage not recorded')
    if review.get('inspected_frame_stride') != 2 or review.get('inspected_sampled_frame_count') != inspection['sample_count']:
        raise ValueError('Wrong frame sampling coverage')
    for field in ['observed_sequence', 'arm_hand_roles', 'anatomical_side_evidence', 'terminal_state', 'uncertainties']:
        if field not in review:
            raise ValueError('Missing review field: ' + field)
    if not review['observed_sequence'] or not review['arm_hand_roles']:
        raise ValueError('Missing visual action evidence')
    if not inspection['native_detail_frames'] or job['frame_count'] - 1 not in inspection['native_detail_frames']:
        raise ValueError('Native transition or final-frame inspection missing')
    if sha(world / 'video.mp4') != job['source_video_sha256']:
        raise ValueError('Generation reference was altered')
    if sha(world / 'first_frame.png') != json.loads((PROJECT / job['source_metadata']).read_text())['first_frame']['sha256']:
        raise ValueError('Initial PNG differs from source')
    prompt_path.write_text(prompt + '\n')
    review.update({'video_id': job['original_id'], 'dataset': job['dataset'],
        'dedicated_review_agent': 'review_' + key.replace('-', '_'),
        'review_source_video': job['source_video'], 'source_frame_count': job['frame_count'], 'source_fps': job['fps'],
        'native_detail_frames': inspection['native_detail_frames'], 'world_model_prompt': prompt,
        'original_task_instruction': job['original_task_prompt'],
        'selected_segment': {'start_frame': 0, 'end_frame_exclusive': job['frame_count'],
                             'reason': 'Complete original single-task reference; no cropping or trimming requested.'},
        'source_video_sha256': job['source_video_sha256'], 'review_completed_at': datetime.now(timezone.utc).isoformat()})
    write_json(world / 'action_review.json', review)
    source_metadata = json.loads((PROJECT / job['source_metadata']).read_text())
    props = source_metadata['video_properties']
    metadata = {'dataset': job['dataset'], 'video_id': job['original_id'], 'variant': 'full_sequence',
        'generation_prompt': prompt, 'prompt_protocol': 'humanoid_action_v1',
        'prompt_provenance': 'visually_authored_from_video_and_initial_frame', 'source_video': job['source_video'],
        'source_video_sha256': job['source_video_sha256'], 'source_url': source_metadata['download_url'],
        'source_repository_commit': source_metadata['revision'], 'view': source_metadata['viewpoint'],
        'crop_rectangle': None, 'spatial_resize': False, 'encoding': 'original_bytes_copied',
        'source_frame_start': 0, 'source_frame_end_exclusive': job['frame_count'],
        'frame_count': job['frame_count'], 'first_frame_index': 0, 'fps': props['fps'],
        'fps_rational': props['fps_rational'], 'duration_seconds': props['duration_seconds'],
        'resolution': {'width': props['width'], 'height': props['height']},
        'displayed_resolution': {'width': props['display_width'], 'height': props['display_height']},
        'codec': props['codec'], 'file_size_bytes': (world / 'video.mp4').stat().st_size,
        'video_sha256': sha(world / 'video.mp4'), 'first_frame_sha256': sha(world / 'first_frame.png'),
        'observed_terminal_state': review['terminal_state'], 'arm_hand_roles': review['arm_hand_roles'],
        'uncertainties': review['uncertainties'], 'annotation_discrepancies': review.get('annotation_discrepancies', []),
        'relative_video_path': 'video.mp4', 'relative_first_frame_path': 'first_frame.png',
        'relative_prompt_path': 'actiongeneration_prompt.txt', 'action_review_path': 'action_review.json',
        'validation': {'original_bytes_unchanged': True, 'first_frame_source_copy_exact': True,
            'full_sequence_retained': True, 'full_source_decode_previously_passed': source_metadata['validation']['full_decode_success'],
            'complete_sampled_frame_coverage_recorded': True, 'prompt_constraint_exact': True},
        'validation_status': 'passed'}
    write_json(world / 'metadata.json', metadata)
    print(json.dumps({'key': key, 'words': len(prompt.split()), 'status': 'complete', 'prompt': prompt}, indent=2))


def audit():
    all_jobs = jobs()
    complete = []
    missing = []
    coverage = {dataset: {'source_frames': 0, 'sampled_frames': 0, 'native_detail_frames': 0, 'reviewers': set()}
                for dataset in ['Humanoid_Everyday', 'NVIDIA_GR1_100']}
    for job in all_jobs:
        world = PROJECT / job['world_model']
        required = ['video.mp4', 'first_frame.png', 'actiongeneration_prompt.txt', 'action_review.json', 'metadata.json']
        if not all((world / f).is_file() for f in required):
            missing.append(job['key'])
            continue
        m = json.loads((world / 'metadata.json').read_text())
        review = json.loads((world / 'action_review.json').read_text())
        prompt = (world / 'actiongeneration_prompt.txt').read_text().strip()
        if m['validation_status'] != 'passed' or sha(world / 'video.mp4') != job['source_video_sha256']:
            raise ValueError('Invalid world-model reference: ' + job['key'])
        if not prompt.endswith(CONSTRAINT) or prompt != m['generation_prompt'] or prompt != review['world_model_prompt']:
            raise ValueError('Prompt protocol or recorded prompt differs: ' + job['key'])
        if sha(world / 'first_frame.png') != json.loads((PROJECT / job['source_metadata']).read_text())['first_frame']['sha256']:
            raise ValueError('First frame differs: ' + job['key'])
        if m['frame_count'] != job['frame_count'] or m['source_frame_start'] != 0 or m['source_frame_end_exclusive'] != job['frame_count']:
            raise ValueError('Complete source sequence not retained: ' + job['key'])
        if review['inspected_frame_stride'] != 2 or review['inspected_sampled_frame_count'] != math.ceil(job['frame_count'] / 2):
            raise ValueError('Sample coverage differs: ' + job['key'])
        if job['frame_count'] - 1 not in review['native_detail_frames']:
            raise ValueError('Native final frame missing: ' + job['key'])
        evidence = coverage[job['dataset']]
        evidence['source_frames'] += job['frame_count']
        evidence['sampled_frames'] += review['inspected_sampled_frame_count']
        evidence['native_detail_frames'] += len(review['native_detail_frames'])
        evidence['reviewers'].add(review['dedicated_review_agent'])
        manifest_root = PROJECT / job['dataset'] / 'metamanifest'
        complete.append({'dataset': job['dataset'], 'video_id': job['original_id'], 'key': job['key'],
            'prompt': m['generation_prompt'], 'relative_prompt_path': os.path.relpath(world / 'actiongeneration_prompt.txt', manifest_root),
            'relative_first_frame_path': os.path.relpath(world / 'first_frame.png', manifest_root),
            'relative_video_path': os.path.relpath(world / 'video.mp4', manifest_root),
            'relative_review_path': os.path.relpath(world / 'action_review.json', manifest_root),
            'source_video_sha256': job['source_video_sha256'], 'protocol': 'humanoid_action_v1', 'status': 'passed'})
    completed_keys = {r['key'] for r in complete}
    job_document = json.loads(JOBS_PATH.read_text())
    for job in job_document['jobs']:
        job['status'] = 'complete' if job['key'] in completed_keys else 'pending'
    write_json(JOBS_PATH, job_document)
    for dataset in ['Humanoid_Everyday', 'NVIDIA_GR1_100']:
        records = [r for r in complete if r['dataset'] == dataset]
        destination = PROJECT / dataset / 'metamanifest'
        with (destination / 'world_model_prompts.csv').open('w', newline='') as f:
            fields = ['dataset', 'video_id', 'key', 'prompt', 'relative_prompt_path', 'relative_first_frame_path', 'relative_video_path', 'relative_review_path', 'source_video_sha256', 'protocol', 'status']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
        write_json(destination / 'world_model_prompt_report.json', {'dataset': dataset,
            'expected': sum(j['dataset'] == dataset for j in all_jobs), 'completed': len(records),
            'pending': [j['key'] for j in all_jobs if j['dataset'] == dataset and j['key'] in missing],
            'completed_source_frame_count': coverage[dataset]['source_frames'],
            'inspected_sampled_frame_count': coverage[dataset]['sampled_frames'],
            'inspected_native_detail_frame_count': coverage[dataset]['native_detail_frames'],
            'distinct_dedicated_review_agents': len(coverage[dataset]['reviewers']),
            'protocol': 'humanoid_action_v1', 'cropping_or_trimming_performed': False})
    print(json.dumps({'expected': len(all_jobs), 'completed': len(complete), 'pending': missing}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['init', 'prepare', 'native', 'finalize', 'audit', 'dispatch'])
    parser.add_argument('--key')
    parser.add_argument('--frames', nargs='+', type=int)
    parser.add_argument('--slots', type=int, default=3)
    args = parser.parse_args()
    if args.phase == 'init': init()
    elif args.phase == 'prepare': prepare(args.key)
    elif args.phase == 'native': native(args.key, args.frames)
    elif args.phase == 'finalize': finalize(args.key)
    elif args.phase == 'audit': audit()
    elif args.phase == 'dispatch':
        path = PROJECT / 'Humanoid_Everyday/metamanifest/world_model_dispatch.json'
        dispatch = json.loads(path.read_text())
        pending = [j for j in jobs() if j['key'] not in dispatch['assigned_keys']]
        selected = pending[:args.slots]
        dispatch['assigned_keys'].extend(j['key'] for j in selected)
        write_json(path, dispatch)
        print(json.dumps([{'key': j['key'], 'task_name': 'review_' + j['key'].replace('-', '_'),
                           'sheets': math.ceil(math.ceil(j['frame_count']/2)/64)} for j in selected], indent=2))


if __name__ == '__main__':
    main()
