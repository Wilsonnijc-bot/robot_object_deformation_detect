"""Build episode folders for the fifteen episodes fetched by fetch_more.py.

Usage: ../.venv/bin/python prepare_more.py /path/to/work_directory [download_record_file]

Uses the same per-episode builder as prepare_sample.py. Embodiment, generation camera and folder
naming follow the first three episodes.
"""
import json
from pathlib import Path, PurePosixPath
import sys

from prepare_sample import build_episode

EMBODIMENTS = {
    'franka_3rgb': ('franka_3rgb', 'h5_franka_3rgb', 'camera_right'),
    'tienkung': ('tienkung_gello_1rgb', 'h5_tienkung_gello_1rgb', 'camera_top'),
    'agilex': ('agilex_3rgb', 'h5_agilex_3rgb', 'camera_front'),
}


def main():
    work = Path(sys.argv[1]).resolve()
    record = json.loads((work / (sys.argv[2] if len(sys.argv) > 2 else 'download_record_more.json')).read_text())
    for entry in record['entries']:
        prefix, embodiment, camera = EMBODIMENTS[entry['embodiment_key']]
        parts = PurePosixPath(entry['archive_member']).parts
        task = parts[0].removesuffix('_h5')
        episode = {'key': entry['embodiment_key'], 'folder': f'{prefix}__{task}', 'episode': parts[-3],
                   'task': task, 'embodiment': embodiment, 'camera': camera}
        build_episode(episode, work / entry['path'], entry)


if __name__ == '__main__':
    main()
