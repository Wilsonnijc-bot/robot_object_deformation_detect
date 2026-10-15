"""Build the episode folders from the HDF5 files extracted by fetch_source.py.

Usage: ../.venv/bin/python prepare_sample.py /path/to/work_directory

For every selected episode this copies the HDF5 byte for byte, renders every RGB camera to a
preview MP4 (frame count and order equal the HDF5 arrays), decodes frame zero of the generation
camera to first_frame.png, and writes initial metadata. Prompts and reviews are authored
afterwards; finalize_sample.py then builds and verifies the generation reference videos.
"""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image

from robomind_common import (EPISODES, NOMINAL_FPS, REPO, REVISION, ROOT, STANDARD_RGB_EMBODIMENTS, color_args,
                             ffmpeg_jpeg_pipe, jpeg_frames, probe, save, sha)

CHANNEL_NOTE = {
    True: 'Standard-decoder colours are natural (RoboMIND lists this embodiment as BGR for its OpenCV loader); no swap applied.',
    False: 'Stored JPEGs hold red and blue swapped relative to a standard decoder (RoboMIND converts BGR->RGB only for '
           'Franka/UR embodiments). All decoded derivatives swap red and blue back; the HDF5 is unchanged.',
}
CAMERA_NOTE = {'camera_left': 'left external', 'camera_right': 'right external', 'camera_top': 'top',
               'camera_front': 'front', 'camera_left_wrist': 'left wrist', 'camera_right_wrist': 'right wrist'}


def describe_hdf5(path):
    with h5py.File(path, 'r') as handle:
        datasets = {}
        handle.visititems(lambda name, obj: datasets.__setitem__(name, {'shape': list(obj.shape), 'dtype': str(obj.dtype)})
                          if isinstance(obj, h5py.Dataset) else None)
        attrs = {k: bool(v) if isinstance(v, (bool, np.bool_)) else str(v) for k, v in handle.attrs.items()}
        language = None
        if 'language_raw' in handle:
            language = [x.decode() if isinstance(x, bytes) else str(x) for x in handle['language_raw'][()]]
        cameras = list(handle['observations/rgb_images'])
        counts = {c: len(handle['observations/rgb_images'][c]) for c in cameras}
    return datasets, attrs, language, cameras, counts


def build_episode(episode, source, member):
    root = ROOT / episode['folder'] / episode['episode']
    original, generation = root / 'original', root / 'generation input'
    (generation / 'metadata').mkdir(parents=True, exist_ok=True)
    (original / 'data').mkdir(parents=True, exist_ok=True)
    stored = original / 'data/trajectory.hdf5'
    shutil.copy2(source, stored)
    assert sha(stored) == member['sha256']
    datasets, attrs, language, cameras, counts = describe_hdf5(stored)
    assert attrs.get('sim') is False, 'sample must be real-robot data'
    frame_count = counts[episode['camera']]
    assert len(set(counts.values())) == 1, counts
    if language:
        (original / 'task_prompt.txt').write_text('\n'.join(language) + '\n')
    previews = {}
    for camera in cameras:
        video = original / 'decoded_videos' / f'{camera}.mp4'
        video.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_jpeg_pipe(jpeg_frames(stored, camera), color_args(episode['embodiment']) + ['-c:v', 'libx264', '-preset', 'medium', '-crf', '14',
            '-pix_fmt', 'yuvj420p', '-color_range', 'pc', '-movflags', '+faststart'], str(video))
        info = probe(video)
        stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
        assert int(stream['nb_read_frames']) == counts[camera]
        previews[camera] = {'path': f'../decoded_videos/{camera}.mp4', 'view': CAMERA_NOTE.get(camera, camera),
            'frame_count': counts[camera], 'resolution': {'width': stream['width'], 'height': stream['height']},
            'encoding': 'lossy_preview_h264_crf14_yuvj420p', 'sha256': sha(video)}
    # Frame zero of the generation camera, decoded from the stored JPEG with no resizing.
    first_frame = generation / 'first_frame.png'
    ffmpeg_jpeg_pipe(jpeg_frames(stored, episode['camera'], 1), color_args(episode['embodiment']) + ['-frames:v', '1'],
                     str(first_frame))
    width, height = Image.open(io.BytesIO(next(jpeg_frames(stored, episode['camera'], 1)))).size
    assert Image.open(first_frame).size == (width, height)
    save(original / 'metadata/metadata.json', {
        'dataset': REPO, 'source_repository_commit': REVISION, 'license': 'apache-2.0',
        'embodiment': episode['embodiment'], 'task': episode['task'], 'episode_id': episode['episode'],
        'source_url': f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{member['source_archive']}",
        'source_archive': member['source_archive'], 'archive_member': member['archive_member'],
        'source_extraction': member.get('note', 'Extracted unchanged from the official example zip; the zip SHA-256 matches the Hugging Face LFS hash.'),
        'hdf5': {'path': '../data/trajectory.hdf5', 'size_bytes': stored.stat().st_size, 'sha256': sha(stored),
                 'attributes': attrs, 'datasets': datasets},
        'source_language_instruction': language,
        'source_language_instruction_note': None if language else
            'No language instruction is stored in this HDF5 and the task is absent from RoboMIND_v1_2_instr.csv.',
        'image_encoding': 'JPEG per frame (RGB, 4:2:0) in observations/rgb_images/<camera>; no frame rate or timestamps are stored.',
        'channel_order_check': CHANNEL_NOTE[episode['embodiment'] in STANDARD_RGB_EMBODIMENTS],
        'decoded_preview_videos': previews, 'source_bytes_preserved': True})
    save(generation / 'metadata/metadata.json', {
        'dataset': 'RoboMIND', 'embodiment': episode['embodiment'], 'task': episode['task'],
        'episode_id': episode['episode'], 'source_repository_commit': REVISION,
        'view': episode['camera'], 'variant': 'full_sequence_pending_visual_review',
        'source_hdf5': '../../original/data/trajectory.hdf5', 'source_camera_dataset': f"observations/rgb_images/{episode['camera']}",
        'relative_video_path': '../video.mp4', 'relative_first_frame_path': '../first_frame.png',
        'relative_prompt_path': '../actiongeneration_prompt.txt', 'action_review_path': 'action_review.json',
        'source_frame_start': 0, 'source_frame_count': frame_count,
        'fps_rational': NOMINAL_FPS, 'fps_source': 'nominal_assumed_not_recorded_in_source',
        'resolution': {'width': width, 'height': height}, 'spatial_resize': False, 'crop_rectangle': None,
        'first_frame_sha256': sha(first_frame), 'validation': {}})
    print(episode['folder'], episode['episode'], frame_count, f'{width}x{height}', flush=True)



def main():
    work = Path(sys.argv[1]).resolve()
    record = json.loads((work / 'download_record.json').read_text())
    for episode in EPISODES:
        source = work / 'extracted' / episode['key'] / 'trajectory.hdf5'
        build_episode(episode, source, record['members'][episode['key']])


if __name__ == '__main__':
    main()
