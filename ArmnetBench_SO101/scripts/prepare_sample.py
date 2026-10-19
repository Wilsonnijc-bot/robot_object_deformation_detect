"""Build the episode folders from the shards downloaded by fetch_source.py.

Usage: ../.venv/bin/python prepare_sample.py /path/to/download_directory [--record download.json]

LeRobot v3 packs many episodes into one AV1 shard per camera, so no per-episode source bytes exist.
Each episode is therefore a derivative: the Parquet rows of the episode (values untouched) and, per
camera, the episode's frames re-encoded as lossless H.264. Every cut frame's YUV checksum must equal
the corresponding frame of the full AV1 shard decode, so the cuts are frame-accurate and lossless
with respect to the decoded source. The generation reference is the unchanged complete cut of
GENERATION_CAMERA; metadata points to it. No action prompt or review is written: those need a
dedicated visual review per video. Existing episode folders are never rewritten.
"""
import argparse
import json
from pathlib import Path
import subprocess

import pyarrow.compute as pc
import pyarrow.parquet as pq

from ab_common import (CAMERAS, EPISODE_META, EPISODES, GENERATION_CAMERA, REPO, REVISION, ROOT, data_member,
                       episode_id, episode_rows, probe, save, sha, video_member)

FPS = 20


def shard_hashes(path, cache={}):
    """Per-frame MD5 of the full AV1 shard decode (cached: one shard serves many episodes)."""
    if path not in cache:
        out = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0', '-fps_mode',
            'passthrough', '-pix_fmt', 'yuv420p', '-f', 'framemd5', '-']).decode()
        cache[path] = [line.rsplit(',', 1)[1].strip() for line in out.splitlines() if line and not line.startswith('#')]
    return cache[path]


def cut_hashes(path):
    out = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0', '-fps_mode',
        'passthrough', '-pix_fmt', 'yuv420p', '-f', 'framemd5', '-']).decode()
    return [line.rsplit(',', 1)[1].strip() for line in out.splitlines() if line and not line.startswith('#')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('--record', default='download.json')
    args = parser.parse_args()
    work = args.work.resolve()
    record = json.loads((work / 'download_record.json').read_text())['members']
    rows = episode_rows(work / EPISODE_META, EPISODES)
    info = json.loads((work / 'meta/info.json').read_text())
    meta_dir = ROOT / 'metamanifest'
    (meta_dir / 'source_static').mkdir(parents=True, exist_ok=True)
    for member, target in {'README.md': meta_dir / 'source_README.md', 'meta/info.json': meta_dir / 'source_static/info.json',
                           'meta/tasks.parquet': meta_dir / 'source_static/tasks.parquet'}.items():
        target.write_bytes((work / member).read_bytes())
    record_path = meta_dir / 'metadata' / args.record
    assert not record_path.exists(), f'{record_path} exists; pass a new --record name'
    for index, task in EPISODES.items():
        eid = episode_id(index)
        root = ROOT / task / eid
        if root.exists():
            print('skip existing', eid, flush=True)
            continue
        row = rows[index]
        length = row['length']
        original, generation = root / 'original', root / 'generation input'
        data_path = work / data_member(row['data/chunk_index'], row['data/file_index'])
        table = pq.read_table(data_path)
        episode = table.filter(pc.equal(table.column('episode_index'), index))
        assert episode.num_rows == length and episode.column('frame_index').to_pylist() == list(range(length))
        parquet = original / f'data/chunk-000/{eid}.parquet'
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(episode, parquet)
        camera_info = {}
        for camera in CAMERAS:
            prefix = f'videos/observation.images.{camera}/'
            shard_member = video_member(camera, row[prefix + 'chunk_index'], row[prefix + 'file_index'])
            shard = work / shard_member
            start = round(row[prefix + 'from_timestamp'] * FPS)
            assert round((row[prefix + 'to_timestamp'] - row[prefix + 'from_timestamp']) * FPS) == length
            reference = shard_hashes(shard)[start:start + length]
            assert len(reference) == length, (eid, camera, len(reference))
            relative = f'videos/chunk-000/observation.images.{camera}/{eid}.mp4'
            stored = original / relative
            stored.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(shard), '-map', '0:v:0', '-fps_mode', 'passthrough',
                '-vf', f'select=between(n\\,{start}\\,{start + length - 1}),setpts=N/({FPS}*TB)', '-c:v', 'libx264',
                '-preset', 'veryslow', '-crf', '0', '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart',
                str(stored)], check=True)
            assert cut_hashes(stored) == reference, f'lossless frame correspondence failed: {eid} {camera}'
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(stored), '-f', 'null', '-'], check=True)
            stream = next(s for s in probe(stored)['streams'] if s['codec_type'] == 'video')
            assert int(stream['nb_read_frames']) == length and stream['avg_frame_rate'] == f'{FPS}/1'
            camera_info[camera] = {'path': '../' + relative, 'size_bytes': stored.stat().st_size, 'sha256': sha(stored),
                'codec': stream['codec_name'], 'pixel_format': stream['pix_fmt'], 'frame_count': length,
                'fps_rational': stream['avg_frame_rate'], 'resolution': {'width': stream['width'], 'height': stream['height']},
                'source_shard': shard_member, 'source_shard_sha256': record[shard_member]['sha256'],
                'source_frame_start': start, 'source_frame_end_exclusive': start + length,
                'encoding': 'frame_accurate_lossless_h264_of_av1_shard'}
            print(eid, camera, length, stream['width'], stream['height'], flush=True)
        first_frame = generation / 'first_frame.png'
        first_frame.parent.mkdir(parents=True, exist_ok=True)
        generation_video = original / camera_info[GENERATION_CAMERA]['path'].replace('../', '')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(generation_video), '-frames:v', '1', str(first_frame)], check=True)
        (original / 'task_prompt.txt').write_text('\n'.join(row['tasks']) + '\n')
        source_row = {k: v for k, v in row.items() if not k.startswith('stats/')}
        data_info = {'path': f'../data/chunk-000/{eid}.parquet', 'size_bytes': parquet.stat().st_size, 'sha256': sha(parquet),
                     'rows': length, 'source_shard': data_member(row['data/chunk_index'], row['data/file_index']),
                     'source_shard_sha256': record[data_member(row['data/chunk_index'], row['data/file_index'])]['sha256']}
        save(original / 'metadata/metadata.json', {'dataset': REPO, 'source_repository_commit': REVISION,
            'license': 'apache-2.0', 'task': task, 'episode_id': eid, 'episode_index': index,
            'robot_type': info['robot_type'], 'fps': FPS, 'source_annotation': source_row,
            'outcome_label': {'success': row['success'], 'success_class': row['success_class'],
                              'policy_type': row['policy_type'], 'policy_repo_id': row['policy_repo_id']},
            'outcome_label_note': 'Publisher label for the whole rollout (teleoperated episodes are always successful); '
                                  'it is not a visual verification.',
            'parquet': data_info, 'camera_videos': camera_info,
            'source_bytes_preserved': False,
            'derivative_note': 'The source packs many episodes per AV1 shard; these files are per-episode lossless cuts '
                               '(Parquet rows copied unchanged). Frame checksums equal the shard decode.'})
        stream_info = camera_info[GENERATION_CAMERA]
        save(generation / 'metadata/metadata.json', {'dataset': 'ArmnetBench_SO101', 'task': task, 'episode_id': eid,
            'source_repository_commit': REVISION, 'source_annotation': source_row,
            'view': f'observation.images.{GENERATION_CAMERA}', 'variant': 'full_sequence_pending_visual_review',
            'review_status': 'pending_visual_review', 'prompt_status': 'not_yet_authored',
            'source_video': '../../original/' + stream_info['path'].replace('../', ''),
            'relative_video_path': '../../original/' + stream_info['path'].replace('../', ''),
            'generation_video_is_source': True, 'relative_first_frame_path': '../first_frame.png',
            'source_frame_start': 0, 'source_frame_end_exclusive': length, 'source_frame_count': length,
            'frame_count': length, 'duration_seconds': length / FPS, 'fps_rational': f'{FPS}/1',
            'resolution': stream_info['resolution'], 'spatial_resize': False, 'crop_rectangle': None,
            'codec': stream_info['codec'], 'pixel_format': stream_info['pixel_format'],
            'encoding': stream_info['encoding'], 'video_sha256': stream_info['sha256'], 'first_frame_sha256': sha(first_frame)})
    save(record_path, {'repository': REPO, 'revision': REVISION, 'members': record,
        'note': 'Every file was verified against the Hub X-Linked-Etag (LFS SHA-256, or git blob SHA-1 for small files).'})


if __name__ == '__main__':
    main()
