"""Fetch twelve additional ActionNet episodes with HTTP Range requests.

Usage: HF_TOKEN=... ../.venv/bin/python fetch_more.py /path/to/new_work_directory

The official archives are uncompressed .tar files of ~100 episodes each. This walks an archive's
512-byte tar headers with tiny Range requests, then downloads only the four members of each
selected episode (<id>.hdf5, <id>/top/rgb.mp4, depth.mkv, timestamps.json), so the rest of the
~4 GB archive is never transferred. The token is read from the environment only.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import requests

REPO = 'FourierIntelligence/ActionNet'
REVISION = 'eb3b5bb16f3a5a7052c1498c1072e78ac99042d0'
# Three episodes with different tasks from each of four archives (distinct source prompts).
SELECTION = {
    '01JGJCNDDF-01JGJY03Q1.tar': ['01JGJQAJP3GQBKPCED', '01JGJS4GWCL7A5XCXC', '01JGJXRGT7E3ULU66F'],
    '01JHF7ZEBF-01JHFAFHR7.tar': ['01JHF84WJCMA7S2S3S', '01JHF8YGPVVKAXY7FK', '01JHFA7YZJTTECHOFB'],
    '01JJ130F0A-01JJ15BREB.tar': ['01JJ136C6DH2EZB42B', '01JJ141D4HW47GWFAL', '01JJ15873JBIEVYT62'],
    '01JMC4WT0H-01JMCBC79M.tar': ['01JMC54RMAVQ5E7IKP', '01JMCA5VKE4FOEQ7FA', '01JMCB8Z194OS4IEC5'],
}
SUFFIXES = ['.hdf5', '/top/rgb.mp4', '/top/depth.mkv', '/top/timestamps.json']


class Remote:
    def __init__(self, archive):
        self.archive = archive
        self.url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{archive}'
        self.session = requests.Session()
        self.auth = {'Authorization': 'Bearer ' + os.environ['HF_TOKEN']}
        self.signed = None

    def _resolve(self):
        response = self.session.get(self.url, headers={**self.auth, 'Range': 'bytes=0-0'}, allow_redirects=False, timeout=60)
        self.signed = response.headers['Location'] if response.status_code in (301, 302, 307) else self.url

    def request(self, start, length, stream=False):
        for _ in range(3):
            if not self.signed:
                self._resolve()
            headers = {'Range': f'bytes={start}-{start + length - 1}'}
            if self.signed == self.url:
                headers.update(self.auth)
            response = self.session.get(self.signed, headers=headers, timeout=300, stream=stream)
            if response.status_code == 206:
                return response
            self.signed = None
        raise RuntimeError(f'{self.archive}: HTTP {response.status_code}')


def parse_header(block):
    name = block[:100].split(b'\0')[0].decode()
    size = int(block[124:136].split(b'\0')[0].strip() or b'0', 8)
    prefix = block[345:500].split(b'\0')[0].decode()
    return (prefix + '/' + name if prefix else name), size, block[156:157]


def locate(remote, wanted):
    """Walk tar headers until every wanted member name has an (offset, size)."""
    found, offset = {}, 0
    while len(found) < len(wanted):
        block = remote.request(offset, 512).content
        if block == b'\0' * 512:
            break
        name, size, kind = parse_header(block)
        if name in wanted and kind in (b'0', b'\0'):
            found[name] = (offset + 512, size)
        offset += 512 + ((size + 511) // 512) * 512
    missing = set(wanted) - set(found)
    assert not missing, (remote.archive, sorted(missing))
    return found


def download(remote, offset, size, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with remote.request(offset, size, stream=True) as response, open(destination, 'wb') as handle:
        for block in response.iter_content(1 << 20):
            handle.write(block)
            digest.update(block)
            written += len(block)
    assert written == size, (destination, written, size)
    return digest.hexdigest()


def main():
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    record = {'repository': REPO, 'revision': REVISION, 'method': 'range_requests_on_uncompressed_tar', 'episodes': {}}
    for archive, episode_ids in SELECTION.items():
        remote = Remote(archive)
        wanted = {eid + suffix for eid in episode_ids for suffix in SUFFIXES}
        located = locate(remote, wanted)
        for eid in episode_ids:
            files = {}
            for suffix in SUFFIXES:
                member = eid + suffix
                offset, size = located[member]
                destination = work / 'extracted' / member
                files[member] = {'archive_offset': offset, 'size_bytes': size,
                                 'sha256': download(remote, offset, size, destination)}
            record['episodes'][eid] = {'archive': archive, 'files': files}
            print(archive, eid, sum(f['size_bytes'] for f in files.values()), flush=True)
    (work / 'download_record.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
