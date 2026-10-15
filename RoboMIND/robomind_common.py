"""Shared helpers for the RoboMIND sample scripts."""
import hashlib
import json
from pathlib import Path
import subprocess
import threading

import h5py

ROOT = Path(__file__).resolve().parent
REPO = 'x-humanoid-robomind/RoboMIND'
REVISION = 'e7ffe31d1fe983a42c3d7b79d192f554fc05b86e'
# RoboMIND does not record a frame rate for these files; 30 FPS is a nominal playback rate.
NOMINAL_FPS = '30/1'
CONSTRAINT = 'Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.'
EPISODES = [
    {'key': 'franka_3rgb', 'folder': 'franka_3rgb__pick_apple_into_drawer', 'episode': '0923_164719',
     'task': 'pick_apple_into_drawer', 'embodiment': 'h5_franka_3rgb', 'camera': 'camera_right'},
    {'key': 'tienkung', 'folder': 'tienkung_gello_1rgb__close_the_drawer_under_the_combination_cabinet',
     'episode': '1211_104950', 'task': 'close_the_drawer_under_the_combination_cabinet',
     'embodiment': 'h5_tienkung_gello_1rgb', 'camera': 'camera_top'},
    {'key': 'agilex', 'folder': 'agilex_3rgb__13_packbowl',
     'episode': '2024_09_29-10_29_40-172857804939817056.00', 'task': '13_packbowl',
     'embodiment': 'h5_agilex_3rgb', 'camera': 'camera_front'},
]


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


def jpeg_frames(hdf5_path, camera, limit=None):
    """Yield the stored JPEG bytes of one RGB camera, in frame order."""
    with h5py.File(hdf5_path, 'r') as handle:
        dataset = handle['observations/rgb_images'][camera]
        for index in range(len(dataset) if limit is None else limit):
            yield bytes(dataset[index])


def ffmpeg_jpeg_pipe(frames, output_args, tail):
    """Feed JPEG frames to ffmpeg's mjpeg image2pipe demuxer; return captured stdout if `tail` is '-'."""
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'image2pipe', '-framerate', NOMINAL_FPS.split('/')[0],
               '-c:v', 'mjpeg', '-i', '-'] + output_args + [tail]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE if tail == '-' else None)
    stdout_chunks = []
    reader = None
    if tail == '-':
        reader = threading.Thread(target=lambda: stdout_chunks.append(process.stdout.read()))
        reader.start()
    for frame in frames:
        process.stdin.write(frame)
    process.stdin.close()
    process.wait()
    if reader:
        reader.join()
    if process.returncode:
        raise RuntimeError('ffmpeg failed')
    return b''.join(stdout_chunks)


def source_frame_hashes(hdf5_path, camera, count):
    """Per-frame MD5 of the YUV decode of the stored JPEGs (the reference every encode must match)."""
    out = ffmpeg_jpeg_pipe(jpeg_frames(hdf5_path, camera, count),
        ['-pix_fmt', 'yuvj420p', '-f', 'framemd5'], '-')
    return [line.rsplit(',', 1)[1].strip() for line in out.decode().splitlines() if line and not line.startswith('#')]


def video_frame_hashes(video, count):
    out = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(video), '-map', '0:v:0',
        '-frames:v', str(count), '-pix_fmt', 'yuvj420p', '-f', 'framemd5', '-']).decode()
    return [line.rsplit(',', 1)[1].strip() for line in out.splitlines() if line and not line.startswith('#')]
