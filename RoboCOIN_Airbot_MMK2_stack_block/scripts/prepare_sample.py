"""Build the episode folders from the files downloaded by fetch_source.py.

Usage: ../.venv/bin/python prepare_sample.py /path/to/download_directory [--record download.json]

Only episodes of rc_common.EPISODES that were downloaded and whose folder does not yet exist are built;
existing episode folders (which may carry reviewed prompts and gate blocks) are never rewritten. The
download provenance is saved as metamanifest/metadata/<record> (default download.json, e.g.
download_more.json for a later batch); an existing record is never overwritten.

Originals (Parquet and the four AV1 camera videos) are copied byte for byte. The generation
reference is the complete head-camera video re-encoded as lossless H.264 (the AV1 source is not
decodable by many generation pipelines); every frame's YUV checksum must equal the AV1 decode.
No action prompt or action review is written: those need a dedicated visual review per video.
"""
import argparse
import filecmp
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow.parquet as pq

from rc_common import CAMERAS, EPISODES, GENERATION_CAMERA, REPO, REVISION, ROOT, TASK, frame_hashes, probe, save, sha


def subtask_timeline(table, names):
    """Run-length segments of the per-frame subtask index tuples."""
    segments = []
    for frame, indices in enumerate(table.column('subtask_annotation').to_pylist()):
        key = tuple(indices)
        if segments and segments[-1]['indices'] == list(key):
            segments[-1]['end_frame_inclusive'] = frame
        else:
            segments.append({'start_frame': frame, 'end_frame_inclusive': frame, 'indices': list(key)})
    for segment in segments:
        segment['subtasks'] = [names.get(i) for i in segment['indices']]
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('--record', default='download.json')
    args = parser.parse_args()
    work = args.work.resolve()
    record = json.loads((work / 'download_record.json').read_text())['members']
    build = []
    for index in EPISODES:
        eid = f'episode_{index:06d}'
        if (ROOT / TASK / eid).exists():
            print('skip existing', eid, flush=True)
        elif f'data/chunk-000/{eid}.parquet' in record:
            build.append(index)
    if not build:
        sys.exit('nothing to build')
    episodes = {x['episode_index']: x for x in map(json.loads, (work / 'meta/episodes.jsonl').read_text().splitlines())}
    scenes = (work / 'annotations/scene_annotations.jsonl').read_text().splitlines()
    names = {x['subtask_index']: x['subtask'] for x in map(json.loads,
             (work / 'annotations/subtask_annotations.jsonl').read_text().splitlines())}
    info = json.loads((work / 'meta/info.json').read_text())
    meta_dir = ROOT / 'metamanifest'
    (meta_dir / 'source_static').mkdir(parents=True, exist_ok=True)
    statics = {member: meta_dir / 'source_static' / Path(member).name for member in
               ['info.yaml', 'meta/info.json', 'meta/tasks.jsonl', 'annotations/subtask_annotations.jsonl']}
    statics['README.md'] = meta_dir / 'source_README.md'
    for member, target in statics.items():
        if target.exists():
            assert filecmp.cmp(work / member, target, shallow=False), f'{target} differs from the download'
        else:
            shutil.copy2(work / member, target)
    record_path = meta_dir / 'metadata' / args.record
    assert not record_path.exists(), f'{record_path} exists; pass a new --record name'
    for index in build:
        eid = f'episode_{index:06d}'
        root = ROOT / TASK / eid
        original, generation = root / 'original', root / 'generation input'
        (generation / 'metadata').mkdir(parents=True, exist_ok=True)
        files = {}
        parquet = original / f'data/chunk-000/{eid}.parquet'
        parquet.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / f'data/chunk-000/{eid}.parquet', parquet)
        files['parquet'] = {'path': f'../data/chunk-000/{eid}.parquet', 'size_bytes': parquet.stat().st_size, 'sha256': sha(parquet)}
        camera_info = {}
        for camera in CAMERAS:
            relative = f'videos/chunk-000/observation.images.{camera}/{eid}.mp4'
            stored = original / relative
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work / relative, stored)
            assert sha(stored) == record[relative]['sha256']
            stream = next(s for s in probe(stored)['streams'] if s['codec_type'] == 'video')
            camera_info[camera] = {'path': '../' + relative, 'size_bytes': stored.stat().st_size, 'sha256': sha(stored),
                'codec': stream['codec_name'], 'pixel_format': stream['pix_fmt'], 'frame_count': int(stream['nb_read_frames']),
                'fps_rational': stream['avg_frame_rate'], 'resolution': {'width': stream['width'], 'height': stream['height']}}
        table = pq.read_table(parquet)
        counts = {c['frame_count'] for c in camera_info.values()}
        assert counts == {table.num_rows} == {episodes[index]['length']}, (counts, table.num_rows)
        assert table.column('frame_index').to_pylist() == list(range(table.num_rows))
        task_lines = episodes[index]['tasks']
        (original / 'task_prompt.txt').write_text('\n'.join(task_lines) + '\n')
        scene = json.loads(scenes[index])
        timeline = subtask_timeline(table, names)
        source = original / camera_info[GENERATION_CAMERA]['path'].replace('../', '')
        video = generation / 'video.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-map', '0:v:0', '-fps_mode', 'passthrough',
            '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '0', '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart',
            str(video)], check=True)
        assert frame_hashes(video) == frame_hashes(source), 'lossless frame correspondence failed'
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'], check=True)
        first_frame = generation / 'first_frame.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-frames:v', '1', str(first_frame)], check=True)
        info_video = probe(video)
        stream = next(s for s in info_video['streams'] if s['codec_type'] == 'video')
        assert int(stream['nb_read_frames']) == table.num_rows and stream['avg_frame_rate'] == '30/1'
        width, height = stream['width'], stream['height']
        save(original / 'metadata/metadata.json', {'dataset': REPO, 'source_repository_commit': REVISION,
            'license': 'apache-2.0', 'task': TASK, 'episode_id': eid, 'episode_index': index,
            'robot_type': info['robot_type'], 'source_annotation': episodes[index],
            'scene_annotation': {'scene_index': scene['scene_index'], 'scene': scene['scene']},
            'subtask_timeline': timeline,
            'subtask_timeline_note': 'Run-length encoding of the per-frame subtask_annotation index tuple in the Parquet; '
                                     'names come from annotations/subtask_annotations.jsonl. Index 19 is the dataset\'s null entry.',
            'source_files': files, 'camera_videos': camera_info,
            'parquet_rows': table.num_rows, 'source_bytes_preserved': True})
        save(generation / 'metadata/metadata.json', {'dataset': 'RoboCOIN_Airbot_MMK2_stack_block', 'task': TASK,
            'episode_id': eid, 'source_repository_commit': REVISION, 'source_annotation': episodes[index],
            'view': f'observation.images.{GENERATION_CAMERA}', 'variant': 'full_sequence_pending_visual_review',
            'review_status': 'pending_visual_review', 'prompt_status': 'not_yet_authored',
            'source_video': '../../original/' + camera_info[GENERATION_CAMERA]['path'].replace('../', ''),
            'relative_video_path': '../video.mp4', 'relative_first_frame_path': '../first_frame.png',
            'source_frame_start': 0, 'source_frame_end_exclusive': table.num_rows, 'source_frame_count': table.num_rows,
            'frame_count': table.num_rows, 'duration_seconds': table.num_rows / 30, 'fps_rational': '30/1',
            'fps_source': 'recorded_in_source_info_json',
            'resolution': {'width': width, 'height': height}, 'spatial_resize': False, 'crop_rectangle': None,
            'codec': stream['codec_name'], 'pixel_format': stream['pix_fmt'],
            'encoding': 'frame_accurate_lossless_h264_of_av1_source',
            'source_video_sha256': camera_info[GENERATION_CAMERA]['sha256'],
            'video_sha256': sha(video), 'first_frame_sha256': sha(first_frame),
            'validation': {'generation_complete_decode_passed': True, 'generation_frames_match_source_decode': True,
                           'generation_resolution_matches_source': True}})
        print(TASK, eid, table.num_rows, f'{width}x{height}', len(timeline), 'subtask segments', flush=True)
    save(record_path, {'repository': REPO, 'revision': REVISION, 'members': record,
        'note': 'Every file was verified against the Hub X-Linked-Etag (LFS SHA-256, or git blob SHA-1 for small files).'})


if __name__ == '__main__':
    main()
