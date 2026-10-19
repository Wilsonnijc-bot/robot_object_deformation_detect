"""Shared helpers for the ArmnetBench SO-101 sample scripts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
REPO = 'armnet/armnetbench_v01_lerobot_so101'
REVISION = '1db69c82b589c7172ee31a6b8f7ef58d1fc2c663'
CAMERAS = ['front']  # only the front camera is retained (the source also has top and wrist)
GENERATION_CAMERA = 'front'
# Teleoperated (human, successful) episodes: the first of each of the eight tasks, plus the
# second ring_insert and block_stack demonstrations.
EPISODES = {255: 'ring_insert', 256: 'ring_insert', 560: 'block_stack', 561: 'block_stack', 910: 'tool_insert',
            1200: 'tool_removal', 1549: 'eye_drops_to_shelf', 1839: 'eye_drops_to_basket',
            2159: 'cable_clip', 2449: 'cable_unclip'}
EPISODE_META = 'meta/episodes/chunk-000/file-000.parquet'
STATIC = ['README.md', 'meta/info.json', 'meta/tasks.parquet', EPISODE_META]


def episode_id(index):
    return f'episode_{index:06d}'


def video_member(camera, chunk, file):
    return f'videos/observation.images.{camera}/chunk-{chunk:03d}/file-{file:03d}.mp4'


def data_member(chunk, file):
    return f'data/chunk-{chunk:03d}/file-{file:03d}.parquet'


def episode_rows(meta_path, indices):
    import pyarrow.parquet as pq
    table = pq.read_table(meta_path)
    wanted = set(indices)
    rows = {}
    for row in table.to_pylist():
        if row['episode_index'] in wanted:
            rows[row['episode_index']] = row
    assert set(rows) == wanted, wanted - set(rows)
    return rows


def members_for(rows):
    found = set()
    for row in rows.values():
        found.add(data_member(row['data/chunk_index'], row['data/file_index']))
        for camera in CAMERAS:
            found.add(video_member(camera, row[f'videos/observation.images.{camera}/chunk_index'],
                                   row[f'videos/observation.images.{camera}/file_index']))
    return sorted(found)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(path)]))
