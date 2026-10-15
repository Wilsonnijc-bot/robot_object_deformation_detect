"""Create and verify the generation references chosen by the visual reviewers.

Each video.mp4 is a lossless H.264 (yuvj420p, full range) encoding of the first N stored JPEG
frames of the generation camera, with no cropping, resizing or FPS conversion. Every frame's YUV
checksum must equal the checksum of the stored JPEG decoded directly.
"""
import json
import subprocess

from PIL import Image

from robomind_common import CONSTRAINT, NOMINAL_FPS, ROOT, ffmpeg_jpeg_pipe, jpeg_frames, probe, save, sha, \
    source_frame_hashes, video_frame_hashes



def main():
    for generation in sorted(ROOT.glob('*/*/generation input')):
        metadata_path = generation / 'metadata/metadata.json'
        review_path = generation / 'metadata/action_review.json'
        metadata = json.loads(metadata_path.read_text())
        review = json.loads(review_path.read_text())
        prompt_path = generation / 'actiongeneration_prompt.txt'
        prompt = prompt_path.read_text().strip()
        assert prompt.endswith(CONSTRAINT)
        assert review['prompt_protocol'] in ('concise_action_v1', 'humanoid_action_v1')
        assert review['prompt_provenance'] == 'visually_authored_from_video_and_initial_frame'
        source = (metadata_path.parent / metadata['source_hdf5']).resolve()
        camera = metadata['view']
        total = metadata['source_frame_count']
        end = review['selected_source_frame_end_exclusive']
        assert 0 < end <= total and review['selected_source_frame_start'] == 0
        full_prompt = generation / 'full_sequence_actiongeneration_prompt.txt'
        if end < total:
            assert full_prompt.read_text().strip().endswith(CONSTRAINT)
            metadata['variant'] = 'first_complete_manipulation'
            metadata['relative_full_sequence_prompt_path'] = '../full_sequence_actiongeneration_prompt.txt'
        else:
            assert not full_prompt.exists(), 'full-sequence prompt only belongs with a shortened reference'
            metadata['variant'] = 'full_sequence'
        video = generation / 'video.mp4'
        candidate = generation / 'video.reviewed.mp4'
        ffmpeg_jpeg_pipe(jpeg_frames(source, camera, end), ['-c:v', 'libx264', '-preset', 'fast', '-crf', '0',
            '-pix_fmt', 'yuvj420p', '-color_range', 'pc', '-movflags', '+faststart'], str(candidate))
        expected = source_frame_hashes(source, camera, end)
        actual = video_frame_hashes(candidate, end + 1)
        assert len(expected) == end and actual == expected, 'frame correspondence failed'
        candidate.replace(video)
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'], check=True)
        info = probe(video)
        stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
        assert int(stream['nb_read_frames']) == end and stream['avg_frame_rate'] == NOMINAL_FPS
        assert (stream['width'], stream['height']) == (metadata['resolution']['width'], metadata['resolution']['height'])
        initial_rgb = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(video), '-frames:v', '1',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
        assert Image.open(generation / 'first_frame.png').convert('RGB').tobytes() == initial_rgb
        metadata.update({'source_frame_end_exclusive': end, 'frame_count': end,
            'duration_seconds': float(info['format']['duration']),
            'duration_note': 'Nominal: frame count at an assumed 30 FPS; the source records no frame rate.',
            'codec': stream['codec_name'], 'pixel_format': stream['pix_fmt'],
            'encoding': 'frame_accurate_lossless_h264_of_stored_jpeg_prefix',
            'video_sha256': sha(video), 'first_frame_sha256': sha(generation / 'first_frame.png'),
            'prompt_sha256': sha(prompt_path), 'prompt_protocol': review['prompt_protocol'],
            'prompt_provenance': review['prompt_provenance'], 'generation_prompt': prompt,
            'review_status': 'visually_reviewed', 'full_source_hdf5_retained_in_original': True})
        metadata['validation'].update({'generation_complete_decode_passed': True,
            'generation_frames_match_stored_jpeg_decode': True,
            'generation_resolution_matches_source': True, 'conditioning_png_matches_generation_frame_zero': True,
            'prompt_constraint_exact': True})
        review['generation_reference_validation'] = {'source_frame_start': 0, 'source_frame_end_exclusive': end,
            'frame_count': end, 'lossless_source_frame_correspondence_passed': True,
            'original_source_hdf5_retained_unchanged': True}
        save(metadata_path, metadata)
        save(review_path, review)
        print(metadata['task'], metadata['episode_id'], 'selected frames', end, 'of', total, flush=True)


if __name__ == '__main__':
    main()
