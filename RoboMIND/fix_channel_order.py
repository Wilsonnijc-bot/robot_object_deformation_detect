"""Regenerate colour-corrected derivatives for embodiments whose stored JPEGs hold red/blue swapped.

Usage: ../.venv/bin/python fix_channel_order.py
       ../.venv/bin/python finalize_sample.py <episode ids printed here>

For every episode whose embodiment is not in STANDARD_RGB_EMBODIMENTS (here: Tien Kung), rebuilds
original/decoded_videos/<camera>.mp4 and generation input/first_frame.png with red and blue swapped
back, and updates both metadata files. The HDF5 is not modified. finalize_sample.py then rebuilds the
generation video from the same corrected decode. Episodes already corrected are skipped.
"""
import json
import subprocess

from PIL import Image

from prepare_sample import CHANNEL_NOTE
from robomind_common import ROOT, STANDARD_RGB_EMBODIMENTS, color_args, ffmpeg_jpeg_pipe, jpeg_frames, probe, save, sha


def main():
    fixed = []
    for generation in sorted(ROOT.glob('*/*/generation input')):
        meta_path = generation / 'metadata/metadata.json'
        meta = json.loads(meta_path.read_text())
        embodiment = meta['embodiment']
        original = generation.parent / 'original'
        orig_path = original / 'metadata/metadata.json'
        orig = json.loads(orig_path.read_text())
        note = CHANNEL_NOTE[embodiment in STANDARD_RGB_EMBODIMENTS]
        if embodiment in STANDARD_RGB_EMBODIMENTS or orig.get('channel_order_check') == note:
            if orig.get('channel_order_check') != note:
                orig['channel_order_check'] = note
                save(orig_path, orig)
            continue
        source = original / 'data/trajectory.hdf5'
        for camera, preview in orig['decoded_preview_videos'].items():
            video = original / 'decoded_videos' / f'{camera}.mp4'
            ffmpeg_jpeg_pipe(jpeg_frames(source, camera), color_args(embodiment) + ['-c:v', 'libx264', '-preset', 'medium',
                '-crf', '14', '-pix_fmt', 'yuvj420p', '-color_range', 'pc', '-movflags', '+faststart'], str(video))
            stream = next(s for s in probe(video)['streams'] if s['codec_type'] == 'video')
            assert int(stream['nb_read_frames']) == preview['frame_count']
            preview['sha256'] = sha(video)
        first_frame = generation / 'first_frame.png'
        size = Image.open(first_frame).size
        ffmpeg_jpeg_pipe(jpeg_frames(source, meta['view'], 1), color_args(embodiment) + ['-frames:v', '1'], str(first_frame))
        assert Image.open(first_frame).size == size
        orig['channel_order_check'] = note
        meta['first_frame_sha256'] = sha(first_frame)
        meta['channel_order'] = 'red_blue_swapped_back_from_stored_jpeg'
        save(orig_path, orig)
        save(meta_path, meta)
        fixed.append(generation.parent.name)
        print('corrected', generation.parent.relative_to(ROOT), flush=True)
    print(' '.join(fixed))


if __name__ == '__main__':
    main()
