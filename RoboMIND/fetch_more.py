"""Fetch five more real-robot episodes per embodiment (fifteen in total).

Usage: HF_TOKEN=... ../.venv/bin/python fetch_more.py /path/to/work_directory_used_by_fetch_source

Per embodiment: the second episode of the existing task (from the example zip or the archive
prefix that fetch_source.py already downloaded), plus the first two complete episodes of each of
two further task archives (see PLAN), each range-downloaded as a 250-700 MB prefix. Only complete
.hdf5 members are written. Writes <work>/download_record_more.json.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import urllib.request
import zipfile
import zlib

REPO = 'x-humanoid-robomind/RoboMIND'
REVISION = 'e7ffe31d1fe983a42c3d7b79d192f554fc05b86e'
RANGE_STEPS = [250_000_000, 450_000_000, 700_000_000]
ZIP = 'example_data/pick_apple_into_drawer_h5 2.zip'
ZIP_MEMBER = 'pick_apple_into_drawer_h5/success_episodes/0923_164743/data/trajectory.hdf5'
# (embodiment key, archive part in the repo, members to keep or None for "first N complete", N, prefix file name)
PLAN = [
    ('tienkung', 'example_data/h5_tienkung_gello_1rgb/close_the_drawer_under_the_combination_cabinet.tar.gz.part-aa',
     ['close_the_drawer_under_the_combination_cabinet/success_episodes/train/1211_111642/data/trajectory.hdf5'], 'tienkung.part'),
    ('agilex', 'example_data/h5_agilex_3rgb/13_packbowl.tar.gz.part-aa',
     ['13_packbowl/success_episodes/train/2024_09_28-18_12_39-172857410899899744.00/data/trajectory.hdf5'], 'agilex.part'),
    ('franka_3rgb', 'benchmark1_0_compressed/h5_franka_3rgb/pick_up_strawberry_in_bowl.tar.gz.part-aa', 2, 'franka_strawberry.part'),
    ('franka_3rgb', 'benchmark1_0_compressed/h5_franka_3rgb/slide_close_cabinet.tar.gz.part-aa', 2, 'franka_slide_cabinet.part'),
    ('tienkung', 'benchmark1_0_compressed/h5_tienkung_gello_1rgb/pick_up_tennis_ball_and_place_it_in_box.tar.gz.part-aa', 2, 'tienkung_tennis.part'),
    ('tienkung', 'benchmark1_0_compressed/h5_tienkung_gello_1rgb/press_down_toaster.tar.gz.part-aa', 2, 'tienkung_toaster.part'),
    ('agilex', 'benchmark1_0_compressed/h5_agilex_3rgb/19_takebread.tar.gz.part-aa', 2, 'agilex_takebread.part'),
    ('agilex', 'benchmark1_0_compressed/h5_agilex_3rgb/36_putpepper.tar.gz.part-aa', 2, 'agilex_putpepper.part'),
]


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts, name
    return path


def range_download(path, destination, size):
    if destination.exists() and destination.stat().st_size >= size:
        return
    request = urllib.request.Request(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}',
        headers={'Authorization': 'Bearer ' + os.environ['HF_TOKEN'], 'Range': f'bytes=0-{size - 1}'})
    with urllib.request.urlopen(request, timeout=900) as response, open(destination, 'wb') as out:
        assert response.status == 206
        shutil.copyfileobj(response, out)


def extract_members(prefix_file, destination_root, names=None, count=None):
    """Write complete .hdf5 members from a truncated .tar.gz prefix; return {member name: size}."""
    done = {}
    try:
        with tarfile.open(prefix_file, 'r|gz') as tar:
            for member in tar:
                path = safe(member.name)
                if not member.isfile() or not member.name.endswith('trajectory.hdf5'):
                    continue
                if names is not None and member.name not in names:
                    continue
                destination = destination_root / Path(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with tar.extractfile(member) as src, open(destination, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    if destination.stat().st_size != member.size:
                        raise EOFError('short')
                except (EOFError, zlib.error, tarfile.ReadError):
                    destination.unlink(missing_ok=True)
                    break
                done[member.name] = member.size
                if names is not None and len(done) == len(names) or count is not None and len(done) == count:
                    break
    except (EOFError, zlib.error, tarfile.ReadError):
        pass
    return done


def main():
    work = Path(sys.argv[1]).resolve()
    out = work / 'extracted_more'
    out.mkdir(exist_ok=True)
    record = {'repository': REPO, 'revision': REVISION, 'entries': []}
    # Second Franka example episode from the already downloaded example zip.
    zip_destination = out / 'franka_3rgb' / ZIP_MEMBER
    if not zip_destination.exists():
        with zipfile.ZipFile(work / ZIP) as archive:
            info = archive.getinfo(ZIP_MEMBER)
            safe(info.filename)
            zip_destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(zip_destination, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    record['entries'].append({'embodiment_key': 'franka_3rgb', 'source_archive': ZIP, 'archive_member': ZIP_MEMBER,
        'path': str(zip_destination.relative_to(work)), 'size_bytes': zip_destination.stat().st_size, 'sha256': sha(zip_destination)})
    (work / 'partial').mkdir(exist_ok=True)
    for key, archive_path, selector, prefix_name in PLAN:
        prefix = work / 'partial' / prefix_name
        names = set(selector) if isinstance(selector, list) else None
        count = selector if isinstance(selector, int) else None
        wanted = len(names) if names else count
        done = {}
        for size in RANGE_STEPS:
            range_download(archive_path, prefix, size)
            shutil.rmtree(out / key / PurePosixPath(archive_path).name.split('.tar')[0], ignore_errors=True)
            done = extract_members(prefix, out / key, names, count)
            if len(done) >= wanted:
                break
            prefix.unlink()
        assert len(done) >= wanted, (archive_path, done)
        for name, member_size in done.items():
            destination = out / key / Path(*safe(name).parts)
            record['entries'].append({'embodiment_key': key, 'source_archive': archive_path, 'archive_member': name,
                'path': str(destination.relative_to(work)), 'size_bytes': destination.stat().st_size,
                'sha256': sha(destination), 'range_downloaded_bytes': prefix.stat().st_size,
                'range_prefix_sha256': sha(prefix),
                'note': 'Member extracted from a truncated prefix of the first split part; the whole-part hash was not verified.'})
            print(key, name, member_size, flush=True)
    (work / 'download_record_more.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
