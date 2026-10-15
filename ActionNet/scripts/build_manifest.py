"""Build the CSV manifest from reviewed episodes. Run ../quality_gate.py first for the gate columns."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def gate_columns(metadata):
    gate = metadata.get('data_quality_gate', {})
    return {'gate_view': gate.get('view', ''), 'gate_sharpness': gate.get('sharpness', ''),
            'gate_threshold': gate.get('threshold', ''), 'gate_result': gate.get('result', '')}


def main():
    rows = []
    for folder in sorted(ROOT.glob('*/generation input')):
        metadata = json.loads((folder / 'metadata' / 'metadata.json').read_text())
        eid = folder.parent.name
        prefix = f'{eid}/generation input/'
        video = (folder / 'metadata' / metadata['relative_video_path']).resolve().relative_to(ROOT).as_posix()
        rows.append({'episode_id': eid, 'video': video,
               'first_frame': prefix + 'first_frame.png', 'prompt': prefix + 'actiongeneration_prompt.txt',
               'metadata': prefix + 'metadata/metadata.json', 'action_review': prefix + 'metadata/action_review.json',
               'source_video': f'{eid}/original/top/rgb.mp4', 'duration_seconds': metadata['duration_seconds'],
               'frame_count': metadata['frame_count'], 'width': metadata['resolution']['width'],
               'height': metadata['resolution']['height'], 'fps': metadata['fps_rational'],
               'video_sha256': metadata['video_sha256'], 'first_frame_sha256': metadata['first_frame_sha256'],
               'prompt_protocol': metadata['prompt_protocol'], **gate_columns(metadata)})
    with open(ROOT / 'metamanifest' / 'world_model_prompts.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Built manifest for', len(rows), 'episodes.')


if __name__ == '__main__':
    main()
