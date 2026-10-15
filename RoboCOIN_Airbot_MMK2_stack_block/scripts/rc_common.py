"""Shared helpers for the RoboCOIN Airbot MMK2 sample scripts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
REPO = 'RoboCOIN/Airbot_MMK2_stack_block'
REVISION = 'ee1901cfd4a45185b71b571d9fdd8608ab43c934'
TASK = 'stack_block'
EPISODES = list(range(10))
CAMERAS = ['cam_front_rgb', 'cam_head_rgb', 'cam_left_wrist_rgb', 'cam_right_wrist_rgb']
GENERATION_CAMERA = 'cam_head_rgb'

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


def frame_hashes(video, pix_fmt='yuv420p'):
    """Per-frame MD5 of a full decode, the reference every re-encode must match."""
    out = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(video), '-map', '0:v:0', '-fps_mode', 'passthrough',
        '-pix_fmt', pix_fmt, '-f', 'framemd5', '-']).decode()
    return [line.rsplit(',', 1)[1].strip() for line in out.splitlines() if line and not line.startswith('#')]

