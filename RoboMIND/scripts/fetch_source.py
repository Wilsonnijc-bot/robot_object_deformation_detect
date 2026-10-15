"""Fetch the three selected RoboMIND episodes from the gated Hugging Face dataset.

Usage: HF_TOKEN=... ../.venv/bin/python fetch_source.py /path/to/new_work_directory

RoboMIND is gated: accept the access terms on the dataset page and export an HF_TOKEN.
The token is read from the environment only and is never written anywhere.

* Franka: one episode from the 180 MB official example zip.
* Tien Kung and AgileX: the official example archives are 12-17 GB split tarballs. Their
  first ~250 MB are range-downloaded and only the first complete .hdf5 member is extracted.
Existing files in the work directory are reused. Extraction is done with explicit path checks
into fresh directories under <work>/extracted/.
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

from huggingface_hub import HfApi, hf_hub_download

REPO = 'x-humanoid-robomind/RoboMIND'
REVISION = 'e7ffe31d1fe983a42c3d7b79d192f554fc05b86e'
RANGE_BYTES = 250_000_000
ZIP = 'example_data/pick_apple_into_drawer_h5 2.zip'
ZIP_MEMBER = 'pick_apple_into_drawer_h5/success_episodes/0923_164719/data/trajectory.hdf5'
PARTS = {
    'tienkung': ('example_data/h5_tienkung_gello_1rgb/close_the_drawer_under_the_combination_cabinet.tar.gz.part-aa',
                 'close_the_drawer_under_the_combination_cabinet/success_episodes/train/1211_104950/data/trajectory.hdf5'),
    'agilex': ('example_data/h5_agilex_3rgb/13_packbowl.tar.gz.part-aa',
               '13_packbowl/success_episodes/train/2024_09_29-10_29_40-172857804939817056.00/data/trajectory.hdf5'),
}
STATIC = ['README.md', 'static/RoboMIND_v1_2_instr.csv', 'static/all_robot_h5_info_v1.2.md', 'static/robomind.yaml']


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


def range_download(path, destination):
    if destination.exists() and destination.stat().st_size == RANGE_BYTES:
        return
    request = urllib.request.Request(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}',
        headers={'Authorization': 'Bearer ' + os.environ['HF_TOKEN'], 'Range': f'bytes=0-{RANGE_BYTES - 1}'})
    with urllib.request.urlopen(request, timeout=600) as response, open(destination, 'wb') as out:
        assert response.status == 206
        shutil.copyfileobj(response, out)


def extract_tar_member(prefix_file, member_name, destination):
    """Stream a truncated .tar.gz prefix and write one complete regular member."""
    with tarfile.open(prefix_file, 'r|gz') as tar:
        for member in tar:
            if member.name == member_name and member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(destination, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                if destination.stat().st_size != member.size:
                    raise EOFError('member truncated by the range prefix')
                return member.size
    raise KeyError(member_name)


def main():
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    extracted = work / 'extracted'
    extracted.mkdir(exist_ok=True)
    api = HfApi()
    lfs = {x.path: x for x in api.get_paths_info(REPO, [ZIP] + [p for p, _ in PARTS.values()],
                                                 repo_type='dataset', revision=REVISION)}
    for name in STATIC + [ZIP]:
        hf_hub_download(REPO, name, repo_type='dataset', revision=REVISION, local_dir=work)
    record = {'static_files': {n: {'sha256': sha(work / n)} for n in STATIC},
              'zip': {'path': ZIP, 'size_bytes': (work / ZIP).stat().st_size, 'sha256': sha(work / ZIP),
                      'hf_lfs_sha256': lfs[ZIP].lfs.sha256 if lfs[ZIP].lfs else None}, 'members': {}}
    destination = extracted / 'franka_3rgb' / 'trajectory.hdf5'
    if not destination.exists():
        with zipfile.ZipFile(work / ZIP) as archive:
            info = archive.getinfo(ZIP_MEMBER)
            safe(info.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(destination, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    record['members']['franka_3rgb'] = {'source_archive': ZIP, 'archive_member': ZIP_MEMBER,
        'size_bytes': destination.stat().st_size, 'sha256': sha(destination)}
    (work / 'partial').mkdir(exist_ok=True)
    for key, (path, member) in PARTS.items():
        prefix = work / 'partial' / f'{key}.part'
        range_download(path, prefix)
        destination = extracted / key / 'trajectory.hdf5'
        if not destination.exists():
            safe(member)
            extract_tar_member(prefix, member, destination)
        record['members'][key] = {'source_archive': path, 'archive_member': member,
            'archive_part_size_bytes': lfs[path].size, 'archive_part_hf_lfs_sha256': lfs[path].lfs.sha256 if lfs[path].lfs else None,
            'range_downloaded_bytes': RANGE_BYTES, 'range_prefix_sha256': sha(prefix),
            'size_bytes': destination.stat().st_size, 'sha256': sha(destination),
            'note': 'Member extracted from a truncated prefix of the first split part; the whole-part hash was not verified.'}
    info = api.dataset_info(REPO, revision=REVISION)
    record.update({'repository': REPO, 'revision': info.sha, 'license': 'apache-2.0'})
    (work / 'download_record.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v['size_bytes'] for k, v in record['members'].items()}, indent=2))


if __name__ == '__main__':
    main()
