"""Owned login-node watcher: wait for Slurm recovery, submit once, observe completion.

No GPU calls occur here. stop.request prevents submission and cancels only this
watcher's recorded allocation. A submission with uncertain outcome is never
retried automatically. No dependent allocations are created.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from common import ROOT, RESULTS, manifest, write, sha, code_sha

STATUS = RESULTS / 'metadata/run_status.json'


def main():
    metadata = RESULTS / 'metadata'
    metadata.mkdir(parents=True, exist_ok=True)
    write(metadata / 'watcher.json', dict(pid=os.getpid(), host=os.uname().nodename,
        session='frame0-submit-20261014-45', socket='frame0-submit-20261014-45', root=str(ROOT)))
    videos = len(manifest())
    frame_links = sum(2 * e['frame_count'] for e in manifest())
    validation = json.loads((metadata / 'input_validation.json').read_text())
    assert validation['status'] == 'passed' and validation['manifest_sha256'] == sha(RESULTS / 'inputs/manifest.json')
    assert validation['code_sha256'] == code_sha()
    job = None
    for _ in range(1440):
        if (metadata / 'stop.request').exists():
            write(STATUS, dict(status='cancelled_before_submission', job_id=None))
            return
        ping = subprocess.run(['timeout', '10s', 'scontrol', 'ping'], capture_output=True, text=True)
        status = dict(status='waiting_for_slurm_controller', job_id=None, checked_unix=time.time(),
            controller_response=ping.stdout.strip(), gpu_count=1, planned_video_workers=[5, 8, 12],
            videos=videos, frame_link_outputs=frame_links, reference='per-video per-link frame0', other_jobs_modified=False)
        write(STATUS, status)
        print('CONTROLLER_CHECK', ping.stdout.strip(), flush=True)
        if ' is UP' in ping.stdout:
            try:
                jobs = subprocess.run(['squeue', '-u', 'zy992', '-o', '%i %j %T %N %b'],
                                      capture_output=True, text=True, timeout=15)
                inventory = subprocess.run(['sinfo', '-p', 'gpu-h200,gpu-l40s', '-N',
                    '-o', '%N %t %G %C'], capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired as error:
                print('CONTROLLER_NOT_READY', error.cmd, flush=True)
                time.sleep(30)
                continue
            if jobs.returncode != 0:
                time.sleep(30)
                continue
            if inventory.returncode != 0:
                time.sleep(30)
                continue
            (metadata / 'gpu_inventory_before_submission.txt').write_text(inventory.stdout)
            (metadata / 'other_jobs_before_submission.txt').write_text(jobs.stdout)
            # Input/runtime paths were CPU-validated. The scheduled entry point
            # must pass actual CUDA/scorer preflight before any full experiment.
            command = ['sbatch', '--parsable', '--account=wang', '--partition=gpu-h200',
                '--gres=gpu:nvidia_h200:1', '--cpus-per-task=24', '--mem=128G', '--time=08:00:00',
                f'--job-name=reg2inv-frame0-{videos}', '--output=' + str(metadata / 'slurm-%j.log'),
                str(ROOT / 'gpu_job.sh'), str(ROOT)]
            (metadata / 'submission_command.json').write_text(json.dumps(command, indent=2)+'\n')
            submitted = subprocess.run(command, capture_output=True, text=True, timeout=90)
            (metadata / 'submission.stdout').write_text(submitted.stdout)
            (metadata / 'submission.stderr').write_text(submitted.stderr)
            if submitted.returncode != 0:
                write(STATUS, dict(status='submission_failed', stdout=submitted.stdout, stderr=submitted.stderr))
                raise RuntimeError('Slurm submission failed; no automatic retry')
            job = int(submitted.stdout.strip().split(';')[0])
            write(STATUS, dict(status='submitted', job_id=job, gpu_count=1, videos=videos,
                reference='frame0', other_jobs_modified=False, cancel_on_stop=[job], dependent_jobs=[]))
            print('SUBMITTED_JOB', job, flush=True)
            break
        time.sleep(30)
    if job is None:
        write(STATUS, dict(status='controller_wait_expired', job_id=None))
        raise RuntimeError('Slurm remained down for the bounded waiting window')
    while True:
        if (metadata / 'stop.request').exists():
            subprocess.run(['scancel', str(job)], check=True, timeout=30)
            write(STATUS, dict(status='cancelled', job_id=job))
            return
        exitfile = metadata / f'experiment-{job}.exit'
        if exitfile.exists():
            rc = int(exitfile.read_text().strip())
            complete = metadata / 'completion.json'
            if rc == 0 and complete.exists():
                receipt = json.loads(complete.read_text())
                assert receipt['status'] == 'completed' and receipt['summary']['videos'] == videos
                assert receipt['summary']['frame_link_rows'] == frame_links
                assert sha(RESULTS / 'scores.csv') == receipt['summary']['scores_sha256']
                assert sha(RESULTS / 'frame_scores.csv') == receipt['summary']['frame_scores_sha256']
                write(STATUS, dict(status='completed', job_id=job, exit_status=rc, summary=receipt['summary']))
            else:
                write(STATUS, dict(status='failed', job_id=job, exit_status=rc))
            return
        state = subprocess.run(['timeout', '15s', 'sacct', '-j', str(job), '-X', '-n', '-P', '-o', 'State'],
                               capture_output=True, text=True)
        if state.returncode == 0 and state.stdout.strip():
            terminal = state.stdout.strip().split('|')[0].split()[0]
            write(STATUS, dict(status='allocated_validating' if terminal == 'RUNNING' and not (metadata / 'preflight.json').exists()
                else 'running' if terminal == 'RUNNING' else 'queued', job_id=job, slurm_state=terminal,
                gpu_count=1, videos=videos, reference='frame0', other_jobs_modified=False,
                cancel_on_stop=[job], dependent_jobs=[]))
            if terminal in {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'COMPLETED'}:
                write(STATUS, dict(status='failed_without_completion_receipt', job_id=job, slurm_state=terminal))
                return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(RESULTS / 'metadata/watcher_error.json', dict(error=repr(error), unix_time=time.time()))
        raise
