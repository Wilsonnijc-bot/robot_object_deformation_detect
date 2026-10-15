"""End-to-end video lanes on one H200; retain only measured capacity improvements."""
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
from pathlib import Path
from common import ROOT, RESULTS, SOURCE_ROOT, manifest, sha, write, code_sha

sys.path.append(str(SOURCE_ROOT))
from orchestrate import Monitor

ACTIVE = []


def stop(*_):
    for p in ACTIVE:
        if p.is_alive():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    raise KeyboardInterrupt('Owned frame-0 experiment stopped')


def child(tasks, replies, tag, frames):
    os.setsid()
    log = RESULTS / 'metadata/workers' / f'{tag}-{os.getpid()}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        from worker import configure, run_case
        configure()
        replies.put(('ready', os.getpid(), None))
        while True:
            case = tasks.get()
            if case is None:
                return
            result = run_case(case, tag=tag if frames else None, frames=frames)
            replies.put(('complete', case, result))
    except BaseException:
        detail = traceback.format_exc()
        print(detail, flush=True)
        replies.put(('error', os.getpid(), detail))
        raise


def phase(cases, workers, tag, frames, uuid):
    ctx = mp.get_context('spawn')
    tasks, replies = ctx.Queue(), ctx.Queue()
    processes, reports = [], []
    monitor = Monitor(uuid, tag)
    monitor.start()
    started = time.monotonic()
    try:
        for _ in range(workers):
            p = ctx.Process(target=child, args=(tasks, replies, tag, frames))
            p.start()
            ACTIVE.append(p)
            processes.append(p)
        ready = 0
        while ready < workers:
            try:
                kind, key, value = replies.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode is not None for p in processes):
                    raise RuntimeError('Worker startup failure')
                continue
            if kind != 'ready':
                raise RuntimeError(str(value))
            ready += 1
        setup_seconds = time.monotonic() - started
        started = time.monotonic()
        for case in cases:
            tasks.put(case)
        for _ in processes:
            tasks.put(None)
        while len(reports) < len(cases):
            try:
                kind, key, value = replies.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(not p.is_alive() for p in processes):
                    raise RuntimeError('Worker failed before completion')
                continue
            if kind != 'complete':
                raise RuntimeError(str(value))
            reports.append(value)
            print('PHASE_PROGRESS', tag, len(reports), '/', len(cases), key, flush=True)
        seconds = time.monotonic()-started
        for p in processes:
            p.join(30)
            assert p.exitcode == 0
        telemetry = monitor.finish()
        work = sum(r['new_frame_links'] for r in reports)
        result = dict(workers=workers, gpu_count=1, setup_seconds=setup_seconds,
            processing_seconds=seconds, frame_links=work, frame_links_per_second=work/seconds,
            reports=reports, **telemetry)
        write(RESULTS / 'metadata/performance' / (tag + '-summary.json'), result)
        return result
    finally:
        monitor.finish()
        for p in processes:
            if p.is_alive():
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                p.join(10)
                if p.is_alive():
                    os.killpg(p.pid, signal.SIGKILL)
                    p.join()
            ACTIVE.remove(p)


def main():
    import torch
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    assert os.environ.get('SLURM_JOB_ID')
    preflight = json.loads((RESULTS / 'metadata/preflight.json').read_text())
    assert preflight['status'] == 'passed'
    assert preflight['manifest_sha256'] == sha(RESULTS / 'inputs/manifest.json')
    assert preflight['code_sha256'] == code_sha()
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    entries = manifest()
    max_workers = min(12, int(os.environ['SLURM_CPUS_PER_TASK']) // 2)
    assert max_workers >= 5
    pilot = [e['video_id'] for e in entries[:6] + entries[15:21]]
    # Build every pilot normal bank before comparing worker counts. Otherwise
    # the first trial's bank construction would bias the warm later trials.
    phase(pilot, 5, 'reference-warmup', 1, uuid)
    trials = []
    for workers in [5, 8, 12]:
        if workers > max_workers:
            continue
        try:
            result = phase(pilot, workers, f'capacity-{workers}', 8, uuid)
        except RuntimeError as error:
            if 'out of memory' not in str(error).lower() or not trials:
                raise
            write(RESULTS / 'metadata/performance' / f'capacity-{workers}-failure.json',
                  dict(status='capacity_rejected', workers=workers, reason=str(error)))
            break
        trials.append(result)
        if result['peak_gpu_memory_mib'] > props.total_memory / 2**20 * .85:
            break
    safe = [r for r in trials if r['peak_gpu_memory_mib'] < props.total_memory / 2**20 * .85]
    assert safe
    # Fixed clips and frame counts. Require a >2% observed improvement to retain
    # extra workers; retain the smaller lane count for near ties.
    best = safe[0]
    for r in safe[1:]:
        if r['frame_links_per_second'] > best['frame_links_per_second'] * 1.02:
            best = r
    allocation = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
        gpu_count=1, gpu_uuid=uuid, gpu_name=props.name, total_video_workers=len(entries),
        selected_workers=best['workers'], allocated_cpus=int(os.environ['SLURM_CPUS_PER_TASK']),
        reference='per-video per-link frozen frame0', capacity_trials=trials)
    write(RESULTS / 'metadata/allocation.json', allocation)
    print('CONCURRENCY_SELECTED', best['workers'], flush=True)
    performance = phase([e['video_id'] for e in entries], best['workers'], 'full', None, uuid)
    from summarize import summarize
    summary = summarize()
    write(RESULTS / 'metadata/completion.json', dict(status='completed', job_id=os.environ['SLURM_JOB_ID'],
        gpu_uuid=uuid, summary=summary, performance=performance))
    print('EXPERIMENT_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
