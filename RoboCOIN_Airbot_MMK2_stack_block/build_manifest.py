"""Build the generation-input manifest. Run ../quality_gate.py first for the gate columns."""
import csv
import json

from rc_common import ROOT


def gate_columns(metadata):
    gate = metadata.get('data_quality_gate', {})
    return {'gate_view': gate.get('view', ''), 'gate_sharpness': gate.get('sharpness', ''),
            'gate_threshold': gate.get('threshold', ''), 'gate_result': gate.get('result', '')}


def main():
    rows = []
    for generation in sorted(ROOT.glob('*/*/generation input')):
        metadata = json.loads((generation / 'metadata/metadata.json').read_text())
        original_metadata = json.loads((generation.parent / 'original/metadata/metadata.json').read_text())
        task, eid = metadata['task'], metadata['episode_id']
        base = generation.relative_to(ROOT).as_posix() + '/'
        original = generation.parent.relative_to(ROOT).as_posix() + '/original/'
        rows.append({'task': task, 'episode_id': eid, 'camera': metadata['view'].replace('observation.images.', ''),
            'video': base + 'video.mp4', 'first_frame': base + 'first_frame.png', 'metadata': base + 'metadata/metadata.json',
            'original_metadata': original + 'metadata/metadata.json', 'source_video': original + metadata['source_video'].split('original/')[1],
            'source_parquet': original + original_metadata['source_files']['parquet']['path'].replace('../', ''),
            'frame_count': metadata['frame_count'], 'duration_seconds': round(metadata['duration_seconds'], 3),
            'width': metadata['resolution']['width'], 'height': metadata['resolution']['height'], 'fps': metadata['fps_rational'],
            'prompt': base + 'actiongeneration_prompt.txt', 'action_review': base + 'metadata/action_review.json',
            'prompt_protocol': metadata.get('prompt_protocol', ''),
            'review_status': metadata['review_status'], 'publisher_instruction': ' '.join(metadata['source_annotation']['tasks']),
            'video_sha256': metadata['video_sha256'], 'first_frame_sha256': metadata['first_frame_sha256'],
            **gate_columns(metadata)})
    with (ROOT / 'metamanifest/world_model_prompts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Built manifest for', len(rows), 'episodes.')


if __name__ == '__main__':
    main()
