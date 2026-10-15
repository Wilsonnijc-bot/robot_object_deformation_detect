"""Download the selected RoboCOIN episodes at the pinned revision.

Usage: ../.venv/bin/python fetch_source.py /path/to/new_empty_download_directory [EPISODE_INDEX ...]

Without episode indices, every episode in rc_common.EPISODES is fetched; with them, only those
(the static metadata files are always fetched).

The repository is public (not gated on the Hub API). Every file is verified against its X-Linked-Etag
(SHA-256 for LFS files, git blob SHA-1 for small non-LFS files).
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

from rc_common import CAMERAS, EPISODES, REPO, REVISION, sha

BASE = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
STATIC = ['README.md', 'info.yaml', 'meta/info.json', 'meta/tasks.jsonl', 'meta/episodes.jsonl',
          'annotations/subtask_annotations.jsonl', 'annotations/scene_annotations.jsonl']


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


def members(episodes):
    yield from STATIC
    for index in episodes:
        yield f'data/chunk-000/episode_{index:06d}.parquet'
        for camera in CAMERAS:
            yield f'videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4'


def main():
    destination = Path(sys.argv[1])
    episodes = [int(arg) for arg in sys.argv[2:]] or EPISODES
    assert set(episodes) <= set(EPISODES), 'add the episodes to rc_common.EPISODES first'
    destination.mkdir(parents=True, exist_ok=True)
    record = {}
    for member in members(episodes):
        target = destination / member
        target.parent.mkdir(parents=True, exist_ok=True)
        head = headers_of(BASE + member)
        commit, linked = head['X-Repo-Commit'], head['X-Linked-Etag'].strip('"')
        with urllib.request.urlopen(BASE + member) as response, open(target, 'wb') as handle:
            handle.write(response.read())
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
    (destination / 'download_record.json').write_text(json.dumps(
        {'repository': REPO, 'revision': REVISION, 'members': record}, indent=2) + '\n')


if __name__ == '__main__':
    main()
