"""Build episode folders for the twelve episodes fetched by fetch_more.py.

Usage: ../.venv/bin/python prepare_more.py /path/to/work_directory

Produces the same layout and metadata as prepare_sample.py. Source members are copied unchanged;
the complete RGB reference is a byte copy of the source video.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from fetch_more import REPO, REVISION, SELECTION
from prepare_sample import ROOT, save, sha

ANNOTATIONS = ROOT / 'metamanifest' / 'metadata' / 'source_annotations.json'


def main():
    work = Path(sys.argv[1]).resolve()
    record = json.loads((work / 'download_record.json').read_text())
    annotations = {item['id']: item for item in json.loads(ANNOTATIONS.read_text())}
    meta = ROOT / 'metamanifest' / 'metadata'
    for eid, info in record['episodes'].items():
        original, generation = ROOT / eid / 'original', ROOT / eid / 'generation input'
        (original / 'metadata').mkdir(parents=True, exist_ok=True)
        (generation / 'metadata').mkdir(parents=True, exist_ok=True)
        files = []
        for member, details in info['files'].items():
            relative = member if member == eid + '.hdf5' else member[len(eid) + 1:]
            destination = original / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work / 'extracted' / member, destination)
            assert sha(destination) == details['sha256']
            files.append({'archive_member': member, 'path': '../' + relative,
                          'size_bytes': destination.stat().st_size, 'sha256': details['sha256']})
        annotation = annotations[eid]
        (original / 'task_prompt.txt').write_text(annotation['prompt'] + '\n')
        shutil.copy2(original / 'top' / 'rgb.mp4', generation / 'video.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(generation / 'video.mp4'),
                        '-frames:v', '1', str(generation / 'first_frame.png')], check=True)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                            '-show_streams', '-show_format', '-of', 'json', str(generation / 'video.mp4')]))
        stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        archive = info['archive']
        save(original / 'metadata' / 'metadata.json', {
            'dataset': REPO, 'episode_id': eid, 'source_repository_commit': REVISION,
            'source_url': f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{archive}',
            'archive_name': archive, 'annotation': annotation, 'source_files': files,
            'source_bytes_preserved': True,
            'fetch_note': 'Members read with HTTP Range requests from the uncompressed tar; the rest of the archive was not downloaded.'})
        save(generation / 'metadata' / 'metadata.json', {
            'dataset': 'ActionNet', 'episode_id': eid, 'variant': 'full_sequence', 'view': 'top',
            'source_repository_commit': REVISION, 'source_video': '../../original/top/rgb.mp4',
            'relative_video_path': '../video.mp4', 'relative_first_frame_path': '../first_frame.png',
            'relative_prompt_path': '../actiongeneration_prompt.txt', 'action_review_path': 'action_review.json',
            'encoding': 'original_bytes_copied', 'spatial_resize': False, 'crop_rectangle': None,
            'source_frame_start': 0, 'source_frame_end_exclusive': int(stream['nb_read_frames']),
            'frame_count': int(stream['nb_read_frames']), 'fps_rational': stream['avg_frame_rate'],
            'resolution': {'width': stream['width'], 'height': stream['height']},
            'duration_seconds': float(probe['format']['duration']), 'codec': stream['codec_name'],
            'video_sha256': sha(generation / 'video.mp4'), 'first_frame_sha256': sha(generation / 'first_frame.png'),
            'source_annotation': annotation,
            'validation': {'source_and_generation_bytes_identical': sha(generation / 'video.mp4') == sha(original / 'top' / 'rgb.mp4'),
                           'full_sequence_retained': True}})
        print(eid, probe['format']['duration'], stream['nb_read_frames'], stream['width'], stream['height'], flush=True)
    download = json.loads((meta / 'download.json').read_text())
    download['additional_episodes'] = {
        'method': 'range_requests_on_uncompressed_tar (fetch_more.py); archives were not downloaded in full',
        'selection': 'Twelve episodes with distinct source prompts, three from each of four archives (a 10%, 50% and 90% '
                     'position scan). Convenience sample; not representative of ActionNet.',
        'archives': {name: ids for name, ids in SELECTION.items()},
        'members': {eid: info['files'] for eid, info in record['episodes'].items()}}
    save(meta / 'download.json', download)


if __name__ == '__main__':
    main()
