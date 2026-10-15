"""Schedule MoSE3 video workers on one shared H200 with a measured capacity trial.

GPU count (one) and video-worker concurrency are separate. Workers load MoSE3
once and claim videos longest first. The first ``--initial-workers`` videos give
a baseline throughput and per-worker memory peak; the scheduler then adds the
workers that fit below ``--memory-fraction`` of the card, measures the next
window of equally long clips, and keeps the higher concurrency only if source
frames per second improve. SIGINT/SIGTERM writes the stop file, terminates every
worker process group and launches nothing further.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from robot.workflows.mose3.inputs import REPO, sha
from robot.workflows.mose3.worker import complete, write_json

PER_WORKER_HEADROOM_BYTES = 2 << 30
# Code that shapes the scientific outputs; hashed into every allocation's preflight record.
SOURCES = sorted([*Path(__file__).parent.glob('*.py'), *(REPO / 'robot/scoring/mose3').glob('*.py')])
WORKERS = {'track': 'robot.workflows.mose3.worker',
           'render': 'robot.workflows.mose3.render_worker'}


def nvidia_smi():
    row = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(', ')
    return dict(uuid=row[0], name=row[1], used_mib=int(row[2]), total_mib=int(row[3]), utilization=int(row[4]))


def checkpoint_identity(checkpoint, mose3_root):
    files = {name: sha(checkpoint / name) for name in ('config.json', 'model.safetensors')}
    commit = subprocess.check_output(['git', '-C', str(mose3_root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(mose3_root), 'status', '--porcelain'], text=True).strip()
    return dict(hub_repository='mose3-tracker/MoSE3', hub_revision='5280ca2ea4c8cf41ac174645fd3aada430af7f40',
                files=files, source_commit=commit, source_clean=not dirty, license='CC BY-NC 4.0 weights')


def preflight(args, entries):
    import torch
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    if torch.cuda.device_count() != 1 or args.gpu_model not in torch.cuda.get_device_name(0):
        raise RuntimeError(f'Exactly one visible {args.gpu_model} required')
    if (torch.ones(2, device='cuda') + 1).sum().item() != 4:
        raise RuntimeError('CUDA startup operation failed')
    identity = checkpoint_identity(args.checkpoint, args.mose3_root)
    if identity['source_commit'] != 'c39ef00f7198515d709f40c668688d4f869255ae' or not identity['source_clean']:
        raise RuntimeError('MoSE3 source is not the pinned clean commit')
    for entry in entries:
        case = entry['video_id']
        receipt = json.loads((args.queries / f'{case}.json').read_text())
        if receipt['queries_sha256'] != sha(args.queries / f'{case}.npz'):
            raise ValueError(f'Query archive hash mismatch: {case}')
        if sha(args.sources / entry['video']) != entry['source_video_sha256']:
            raise ValueError(f'Source video hash mismatch: {case}')
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'queries', 'sources', 'output', 'checkpoint', 'mose3-root', 'metadata'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker', choices=WORKERS, default='track',
                        help='track: scientific outputs; render: display-only rigidity videos')
    parser.add_argument('--tracking', type=Path, help='Completed tracking cases (required by --worker render)')
    parser.add_argument('--gpu-model', default='H200',
                        help='Required visible GPU model; the scientific tracking run used H200')
    parser.add_argument('--initial-workers', type=int, default=5)
    parser.add_argument('--max-workers', type=int, default=12)
    parser.add_argument('--memory-fraction', type=float, default=0.85)
    parser.add_argument('--cases', help='Optional comma-separated subset (smoke tests)')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if (args.worker == 'render') != (args.tracking is not None):
        parser.error('--tracking is required by, and only used with, --worker render')
    entries = json.loads((args.inputs / 'manifest.json').read_text())['entries']
    if args.cases:
        wanted = args.cases.split(',')
        entries = [e for e in entries if e['video_id'] in wanted]
        if len(entries) != len(wanted):
            raise ValueError('Unknown case in --cases')
    job = os.environ.get('SLURM_JOB_ID', 'local')
    metadata = args.metadata / f'allocation-{job}'
    metadata.mkdir(parents=True, exist_ok=False)
    identity = preflight(args, entries)
    gpu = nvidia_smi()
    write_json(metadata / 'checkpoint_identity.json', identity)
    write_json(metadata / 'preflight.json', dict(status='passed', worker=args.worker, gpu=gpu, gpu_count=1, cases=len(entries),
               initial_workers=args.initial_workers, max_workers=args.max_workers, interpreter=sys.executable,
               inputs_manifest_sha256=sha(args.inputs / 'manifest.json'),
               source_sha256={str(p.relative_to(REPO)): sha(p) for p in SOURCES}))
    print('MOSE3_PREFLIGHT_PASSED', flush=True)
    if args.preflight_only:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    order = [e['video_id'] for e in sorted(entries, key=lambda e: -e['frame_count'])]
    frames = {e['video_id']: e['frame_count'] for e in entries}
    pending = [case for case in order if not complete(args.output, case)]
    claims, status, stop_file = metadata / 'claims', metadata / 'status.jsonl', metadata / 'stop'
    claims.mkdir()
    status.touch()
    workers, lock, stop = {}, threading.RLock(), threading.Event()
    started = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 8) // args.max_workers)))

    def launch(count):
        with lock:
            for _ in range(count):
                if stop.is_set():
                    return
                worker = f'w{len(workers):02d}'
                command = [sys.executable, '-u', '-m', WORKERS[args.worker],
                           '--worker-id', worker, '--order', ','.join(pending),
                           '--checkpoint-identity', str(metadata / 'checkpoint_identity.json')]
                for name in ('inputs', 'queries', 'sources', 'output', 'checkpoint', 'mose3_root'):
                    command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
                command += ['--claims', str(claims), '--status', str(status), '--stop-file', str(stop_file)]
                if args.tracking:
                    command += ['--tracking', str(args.tracking)]
                log = (metadata / f'{worker}.log').open('x')
                workers[worker] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                   start_new_session=True)

    def cancel(signum, frame):
        stop.set()
        stop_file.touch()
        with lock:
            for process in workers.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)

    def rows(event):
        parsed = []
        for line in status.read_text().splitlines():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line still being appended
        return [r for r in parsed if r['event'] == event]

    def monitor():
        with (metadata / 'telemetry.jsonl').open('a') as stream:
            while not stop.is_set():
                row = dict(unix_time=time.time(), elapsed_seconds=time.monotonic() - started)
                try:
                    row['gpu'] = nvidia_smi()
                    row['load_average'] = os.getloadavg()
                    row['meminfo'] = Path('/proc/meminfo').read_text().splitlines()[:3]
                    row['pressure'] = {n: Path(f'/proc/pressure/{n}').read_text() for n in ('cpu', 'memory', 'io')
                                       if Path(f'/proc/pressure/{n}').exists()}
                    with lock:
                        row['live_workers'] = sum(p.poll() is None for p in workers.values())
                    done = rows('complete')
                    row['completed_videos'] = len(done)
                    row['source_frames_per_second'] = sum(r['frames'] for r in done) / max(row['elapsed_seconds'], 1)
                except Exception as error:
                    row['monitor_error'] = str(error)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                stop.wait(5)

    def peak_mib(since):
        values = [json.loads(line) for line in (metadata / 'telemetry.jsonl').read_text().splitlines()]
        return max((r['gpu']['used_mib'] for r in values if 'gpu' in r and r['unix_time'] >= since), default=0)

    def wait_for(count, since):
        """Wait until ``count`` videos finish after ``since``; returns their rows."""
        while not stop.is_set():
            finished = [r for r in rows('complete') + rows('failed') if r['unix_time'] >= since]
            alive = any(p.poll() is None for p in workers.values())
            if len(finished) >= count or not alive:
                return finished
            time.sleep(2)
        return []

    threading.Thread(target=monitor, daemon=True).start()
    decision = dict(gpu_count=1, initial_workers=args.initial_workers, pending=len(pending))
    try:
        baseline_start = time.time()
        launch(min(args.initial_workers, len(pending)))
        baseline = wait_for(min(args.initial_workers, len(pending)), baseline_start)
        if stop.is_set():
            raise SystemExit(130)
        ok = [r for r in baseline if r['event'] == 'complete']
        retained = len(workers)
        if ok and len(ok) == len(baseline) and len(pending) > len(baseline):
            window = max(r['unix_time'] for r in ok) - baseline_start
            baseline_rate = sum(r['frames'] for r in ok) / max(window, 1e-6)
            per_worker = max(r['peak_reserved_bytes'] for r in ok) + PER_WORKER_HEADROOM_BYTES
            total = gpu['total_mib'] << 20
            target = min(args.max_workers, int(args.memory_fraction * total // per_worker),
                         len(pending) - len(baseline) + len(workers))
            decision.update(baseline_frames_per_second=baseline_rate, baseline_peak_mib=peak_mib(baseline_start),
                            per_worker_reserved_bytes=per_worker, target_workers=target,
                            baseline_videos=[r['video_id'] for r in ok])
            if target > len(workers):
                trial_start = time.time()
                launch(target - len(workers))
                trial = wait_for(target, trial_start)
                good = [r for r in trial if r['event'] == 'complete']
                trial_window = max((r['unix_time'] for r in good), default=trial_start) - trial_start
                trial_rate = sum(r['frames'] for r in good) / max(trial_window, 1e-6)
                oom = any(r['event'] == 'failed' and r.get('out_of_memory') for r in trial)
                improved = good and not oom and trial_rate > 1.05 * baseline_rate
                decision.update(trial_workers=target, trial_frames_per_second=trial_rate, trial_out_of_memory=oom,
                                trial_peak_mib=peak_mib(trial_start), trial_videos=[r['video_id'] for r in good],
                                comparison_limit='Different clips of the same frame count; one allocation.')
                retained = target if improved else args.initial_workers
                if not improved:
                    for worker in sorted(workers)[args.initial_workers:]:
                        Path(str(stop_file) + f'.{worker}').touch()
        decision['retained_workers'] = retained
        write_json(metadata / 'capacity-decision.json', decision)
        print('MOSE3_CAPACITY', json.dumps(decision), flush=True)
        while any(p.poll() is None for p in workers.values()) and not stop.is_set():
            time.sleep(2)
        # Only out-of-memory failures are retried, once, with the baseline worker count.
        failed = {r['video_id'] for r in rows('failed')} - {r['video_id'] for r in rows('complete')}
        retry = [case for case in failed if all(r.get('out_of_memory') for r in rows('failed') if r['video_id'] == case)]
        if retry and not stop.is_set():
            for case in retry:
                (claims / case).rename(claims / f'{case}.oom-{time.time_ns()}')
            pending[:] = [case for case in order if case in retry]
            for worker in list(workers):
                Path(str(stop_file) + f'.{worker}').unlink(missing_ok=True)
            launch(max(1, min(decision['retained_workers'] // 2, len(retry))))
            decision['out_of_memory_retry'] = dict(videos=retry, workers=max(1, min(decision['retained_workers'] // 2, len(retry))))
            while any(p.poll() is None for p in workers.values()) and not stop.is_set():
                time.sleep(2)
        if stop.is_set():
            raise SystemExit(130)
        missing = [case for case in order if not complete(args.output, case)]
        summary = dict(cases=len(order), complete=len(order) - len(missing), missing=missing,
                       elapsed_seconds=time.monotonic() - started, decision=decision,
                       source_frames=sum(frames.values()))
        write_json(metadata / 'summary.json', summary)
        if missing:
            raise SystemExit(1)
        print('MOSE3_TRACKING_COMPLETE', flush=True)
    finally:
        stop.set()


if __name__ == '__main__':
    main()
