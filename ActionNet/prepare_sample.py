"""Extract the three selected episodes from an already downloaded official tar.

Usage: ../.venv/bin/python prepare_sample.py ARCHIVE ANNOTATIONS
Credentials are neither needed nor stored by this script.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
REVISION = 'eb3b5bb16f3a5a7052c1498c1072e78ac99042d0'
EPISODES = ['01JN0Y324H2ZFDTSYD', '01JN0YDDF5ZYHXSSFO', '01JN0YMV54BPXU4PLB']


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    archive, annotations = map(Path, sys.argv[1:])
    assert archive.stat().st_size == 1067192320, 'Unexpected archive size'
    annotation_map = {item['id']: item for item in json.loads(annotations.read_text())}
    meta = ROOT / 'metamanifest' / 'metadata'
    meta.mkdir(parents=True, exist_ok=True)
    shutil.copy2(annotations, meta / 'source_annotations.json')
    with tarfile.open(archive, 'r:') as tar:
        for eid in EPISODES:
            original = ROOT / eid / 'original'
            generation = ROOT / eid / 'generation input'
            (original / 'metadata').mkdir(parents=True, exist_ok=True)
            (generation / 'metadata').mkdir(parents=True, exist_ok=True)
            files = []
            for member in tar.getmembers():
                if not member.isfile() or not (member.name == eid + '.hdf5' or member.name.startswith(eid + '/')):
                    continue
                relative = member.name if member.name == eid + '.hdf5' else member.name[len(eid) + 1:]
                destination = original / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(destination, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                files.append({'archive_member': member.name, 'path': '../' + relative,
                              'size_bytes': destination.stat().st_size, 'sha256': sha(destination)})
            (original / 'task_prompt.txt').write_text(annotation_map[eid]['prompt'] + '\n')
            shutil.copy2(original / 'top' / 'rgb.mp4', generation / 'video.mp4')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(generation / 'video.mp4'),
                            '-frames:v', '1', str(generation / 'first_frame.png')], check=True)
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                                '-show_streams', '-show_format', '-of', 'json', str(generation / 'video.mp4')]))
            stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            save(original / 'metadata' / 'metadata.json', {
                'dataset': 'FourierIntelligence/ActionNet', 'episode_id': eid,
                'source_repository_commit': REVISION,
                'source_url': f'https://huggingface.co/datasets/FourierIntelligence/ActionNet/resolve/{REVISION}/{archive.name}',
                'archive_name': archive.name, 'annotation': annotation_map[eid],
                'source_files': files, 'source_bytes_preserved': True})
            metadata = {
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
                'source_annotation': annotation_map[eid],
                'validation': {'source_and_generation_bytes_identical': sha(generation / 'video.mp4') == sha(original / 'top' / 'rgb.mp4'),
                               'full_sequence_retained': True}}
            save(generation / 'metadata' / 'metadata.json', metadata)
            print(eid, metadata['duration_seconds'], metadata['frame_count'], metadata['resolution'], flush=True)
    save(meta / 'download.json', {
        'dataset': 'FourierIntelligence/ActionNet', 'revision': REVISION, 'license': 'cc-by-nc-sa-4.0',
        'download_method': 'authenticated_existing_browser_session', 'archive_name': archive.name,
        'archive_size_bytes': archive.stat().st_size, 'archive_sha256': sha(archive),
        'selected_episode_ids': EPISODES,
        'selection': 'First, annotation variant, and last episode of smallest source archive. Convenience sample of one task; not representative of the whole dataset.',
        'source_annotation_sha256': sha(meta / 'source_annotations.json')})


if __name__ == '__main__':
    main()
