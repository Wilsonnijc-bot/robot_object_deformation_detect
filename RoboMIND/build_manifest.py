"""Build the generation-input manifest. Run ../quality_gate.py first for the gate columns."""
import csv
import json

from robomind_common import ROOT


def gate_columns(metadata):
    gate = metadata.get('data_quality_gate', {})
    return {'gate_view': gate.get('view', ''), 'gate_sharpness': gate.get('sharpness', ''),
            'gate_threshold': gate.get('threshold', ''), 'gate_result': gate.get('result', '')}


def main():
    rows = []
    for generation in sorted(ROOT.glob('*/*/generation input')):
        metadata = json.loads((generation / 'metadata/metadata.json').read_text())
        base = generation.relative_to(ROOT).as_posix() + '/'
        original = generation.parent.relative_to(ROOT).as_posix() + '/original/'
        rows.append({'embodiment': metadata['embodiment'], 'task': metadata['task'], 'episode_id': metadata['episode_id'],
            'camera': metadata['view'], 'video': base + 'video.mp4', 'first_frame': base + 'first_frame.png',
            'prompt': base + 'actiongeneration_prompt.txt', 'metadata': base + 'metadata/metadata.json',
            'action_review': base + 'metadata/action_review.json', 'source_hdf5': original + 'data/trajectory.hdf5',
            'source_frame_count': metadata['source_frame_count'], 'source_frame_start': 0,
            'source_frame_end_exclusive': metadata['source_frame_end_exclusive'], 'frame_count': metadata['frame_count'],
            'nominal_duration_seconds': round(metadata['duration_seconds'], 3),
            'width': metadata['resolution']['width'], 'height': metadata['resolution']['height'],
            'fps_nominal': metadata['fps_rational'], 'prompt_protocol': metadata['prompt_protocol'],
            'variant': metadata['variant'], 'video_sha256': metadata['video_sha256'],
            'first_frame_sha256': metadata['first_frame_sha256'], **gate_columns(metadata)})
    with (ROOT / 'metamanifest/world_model_prompts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Built manifest for', len(rows), 'episodes.')


if __name__ == '__main__':
    main()
