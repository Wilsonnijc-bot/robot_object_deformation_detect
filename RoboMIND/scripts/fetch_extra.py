"""Fetch six more real-robot episodes: the first two complete episodes of three further tasks.

Usage: HF_TOKEN=... ../.venv/bin/python fetch_extra.py /path/to/work_directory
Then:  ../.venv/bin/python prepare_more.py /path/to/work_directory download_record_extra.json

Same method as fetch_more.py: each first split part is range-downloaded as a 250-700 MB prefix and
only complete trajectory.hdf5 members are written. Writes <work>/download_record_extra.json.
"""
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

from fetch_more import RANGE_STEPS, REPO, REVISION, extract_members, range_download, safe, sha

PLAN = [
    ('franka_3rgb', 'benchmark1_0_compressed/h5_franka_3rgb/place_in_bread_in_plate.tar.gz.part-aa', 2, 'franka_bread_plate.part'),
    ('tienkung', 'benchmark1_0_compressed/h5_tienkung_gello_1rgb/upright_white_cup.tar.gz.part-aa', 2, 'tienkung_upright_cup.part'),
    ('tienkung', 'benchmark1_0_compressed/h5_tienkung_gello_1rgb/place_yellow_banana_on_plastic_plate.tar.gz.part-aa', 2, 'tienkung_banana.part'),
]


def main():
    work = Path(sys.argv[1]).resolve()
    out = work / 'extracted_extra'
    (work / 'partial').mkdir(parents=True, exist_ok=True)
    record = {'repository': REPO, 'revision': REVISION, 'entries': []}
    for key, archive_path, count, prefix_name in PLAN:
        prefix = work / 'partial' / prefix_name
        done = {}
        for size in RANGE_STEPS:
            range_download(archive_path, prefix, size)
            shutil.rmtree(out / key / PurePosixPath(archive_path).name.split('.tar')[0], ignore_errors=True)
            done = extract_members(prefix, out / key, None, count)
            if len(done) >= count:
                break
            prefix.unlink()
        assert len(done) >= count, (archive_path, done)
        for name in done:
            destination = out / key / Path(*safe(name).parts)
            record['entries'].append({'embodiment_key': key, 'source_archive': archive_path, 'archive_member': name,
                'path': str(destination.relative_to(work)), 'size_bytes': destination.stat().st_size,
                'sha256': sha(destination), 'range_downloaded_bytes': prefix.stat().st_size,
                'range_prefix_sha256': sha(prefix),
                'note': 'Member extracted from a truncated prefix of the first split part; the whole-part hash was not verified.'})
            print(key, name, flush=True)
    (work / 'download_record_extra.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
