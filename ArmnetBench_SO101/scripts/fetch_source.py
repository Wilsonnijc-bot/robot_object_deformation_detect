"""Download the shards that hold the selected ArmnetBench episodes at the pinned revision.

Usage: ../.venv/bin/python -I fetch_source.py /path/to/new_empty_download_directory

The repository is public. LeRobot v3 packs many episodes into each Parquet/MP4 shard, so whole
shards are fetched (front camera only). Every file is verified against its X-Linked-Etag
(SHA-256 for LFS files, git blob SHA-1 for small non-LFS files).
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.request

from ab_common import EPISODE_META, EPISODES, REPO, REVISION, STATIC, episode_rows, members_for, sha

BASE = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def headers_of(url):
    """Hub response headers (commit, LFS sha256) come from the first hop, before the CDN redirect."""
    request = urllib.request.Request(url, method='HEAD')
    try:
        return urllib.request.build_opener(_NoRedirect).open(request).headers
    except urllib.error.HTTPError as error:
        if error.code in (301, 302, 307):
            return error.headers
        raise


def fetch(member, destination, record):
    target = destination / member
    target.parent.mkdir(parents=True, exist_ok=True)
    head = headers_of(BASE + member)
    commit, linked = head['X-Repo-Commit'], head['X-Linked-Etag'].strip('"')
    with urllib.request.urlopen(BASE + member) as response, open(target, 'wb') as handle:
        shutil.copyfileobj(response, handle, 1024 * 1024)
    assert commit == REVISION, (member, commit)
    digest = sha(target)
    if len(linked) == 64:
        assert digest == linked, member
        method = 'x-linked-etag (LFS sha256)'
    else:
        data = target.read_bytes()
        assert hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest() == linked, member
        method = 'x-linked-etag (git blob sha1)'
    record[member] = {'size_bytes': target.stat().st_size, 'sha256': digest, 'verified_against': method}
    print(member, target.stat().st_size, flush=True)


def main():
    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    record = {}
    for member in STATIC:
        fetch(member, destination, record)
    rows = episode_rows(destination / EPISODE_META, EPISODES)
    for member in members_for(rows):
        fetch(member, destination, record)
    (destination / 'download_record.json').write_text(json.dumps(
        {'repository': REPO, 'revision': REVISION, 'members': record}, indent=2) + '\n')


if __name__ == '__main__':
    main()
