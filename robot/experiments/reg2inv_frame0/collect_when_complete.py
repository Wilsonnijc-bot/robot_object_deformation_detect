"""Read-only observation and verified local collection of this frame-0 run."""
import argparse
import io
import json
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from common import ROOT, RESULTS, sha, write

REMOTE = '/PHShome/zy992/Wilson/deformationdetection/workspace-reg2inv-frame0-20261013/robot/experiments/reg2inv_frame0/results/20261014-45'
TITLE = '## Reg2Inv frame-0 normal reference, all 45 videos — 2026-10-14'


def ledger(status, detail):
    path = ROOT.parents[2] / 'documentation/gpu/experiment_GPU_record.md'
    text = path.read_text()
    start = text.index(TITLE)
    end = text.find('\n## ', start+3)
    if end < 0:
        end = len(text)
    lines = text[start:end].splitlines()
    for i, line in enumerate(lines):
        if line.startswith('- **Status:**'):
            lines[i] = '- **Status:** ' + status + '; ' + detail
        if line.startswith('- **Results:**'):
            lines[i] = '- **Results:** ' + detail
    path.write_text(text[:start]+'\n'.join(lines)+'\n'+text[end:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    args = parser.parse_args()
    ssh = ['ssh', '-S', args.socket, '-oBatchMode=yes', '-oConnectTimeout=15',
           'zy992@eris2n7.research.partners.org']
    for _ in range(1440):
        try:
            raw = subprocess.check_output(ssh+['cat '+shlex.quote(REMOTE+'/metadata/run_status.json')],
                                          text=True, timeout=30)
            status = json.loads(raw)
            write(RESULTS / 'metadata/run_status.json', status)
            print('REMOTE_STATUS', status['status'], status.get('job_id'), flush=True)
        except (subprocess.SubprocessError, json.JSONDecodeError):
            time.sleep(60)
            continue
        if status['status'] == 'completed':
            break
        if status['status'] in {'submission_failed', 'submission_outcome_uncertain', 'controller_wait_expired',
                'failed', 'failed_without_completion_receipt', 'cancelled', 'cancelled_before_submission'}:
            ledger(status['status'], 'Remote status recorded; no new result accepted. Job '+str(status.get('job_id')))
            return
        time.sleep(60)
    else:
        raise RuntimeError('Local observation window expired; remote run status remains authoritative')
    summary = status['summary']
    archive = RESULTS / '.collection.tar.gz'
    names = ['scores.csv', 'frame_scores.csv', 'cases', 'banks', 'inputs', 'metadata']
    command = 'cd '+shlex.quote(REMOTE)+' && tar -czf - '+' '.join(shlex.quote(n) for n in names)
    with archive.open('wb') as f:
        subprocess.run(ssh+[command], stdout=f, check=True, timeout=1800)
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            target = (RESULTS / member.name).resolve()
            if not target.is_relative_to(RESULTS.resolve()) or member.issym() or member.islnk():
                raise RuntimeError('Unsafe collection archive entry')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as src, target.open('wb') as dst:
                    for block in iter(lambda: src.read(8<<20), b''):
                        dst.write(block)
            else:
                raise RuntimeError('Unexpected archive member')
    assert sha(RESULTS / 'scores.csv') == summary['scores_sha256']
    assert sha(RESULTS / 'frame_scores.csv') == summary['frame_scores_sha256']
    import numpy as np
    checked = 0
    for case in (RESULTS / 'cases').iterdir():
        for path in case.glob('*.npz'):
            with np.load(path, allow_pickle=False) as a:
                assert str(a['manifest_sha256']) == summary['manifest_sha256']
                assert str(a['code_sha256']) == summary['code_sha256']
                assert str(a['method']) == 'reg2inv_frame0' and int(a['reference_frame_index']) == 0
                assert a['point_scores'].shape == (len(a['points_camera_m']),)
                assert np.isfinite(a['points_camera_m']).all()
                if str(a['score_status']) == 'ok':
                    assert np.isfinite(a['point_scores']).all()
                checked += 1
    assert checked == summary['frame_link_rows']
    write(RESULTS / 'metadata/collection.json', dict(status='verified', job_id=status['job_id'],
        archives_verified=checked, summary=summary, local_root=str(RESULTS), remote_root=REMOTE))
    ledger('completed', f"Job {status['job_id']} exited 0; {summary['videos']} videos, {summary['video_link_scores']} video/link scores, {checked:,} frame/link results collected and verified. Available: {summary['available']:,}. Frame-0 normal references only; no CAD or FoundationPose.")
    print('FRAME0_COLLECTION_VERIFIED', checked, flush=True)


if __name__ == '__main__':
    main()
