"""One allocation, measured video concurrency, owned process groups and durable logs."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import traceback
from common import ROOT, RESULTS, manifest, sha, write, code_sha

ACTIVE = []


def owned_process_usage():
    pending, pids = [os.getpid()], set()
    while pending:
        pid = pending.pop()
        if pid in pids:
            continue
        pids.add(pid)
        try:
            pending.extend(map(int, Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
        except OSError:
            pass
    cpu_ticks = rss_pages = disk_read = disk_write = blocked = 0
    for pid in pids:
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            cpu_ticks += int(fields[11]) + int(fields[12])
            rss_pages += int(fields[21])
            blocked += int(fields[0] == 'D')
            io = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
            disk_read += int(io['read_bytes'])
            disk_write += int(io['write_bytes'])
        except (OSError, ValueError):
            pass
    return dict(owned_process_count=len(pids), owned_cpu_seconds=cpu_ticks / os.sysconf('SC_CLK_TCK'),
                owned_rss_mib=rss_pages * os.sysconf('SC_PAGE_SIZE') / 2**20,
                owned_disk_read_bytes=disk_read, owned_disk_write_bytes=disk_write,
                owned_io_blocked_processes=blocked)


def stop_owned(*_):
    for process in ACTIVE:
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    raise KeyboardInterrupt('Experiment cancelled')


class Monitor:
    def __init__(self, uuid, name):
        self.uuid, self.name = uuid, name
        self.done = threading.Event()
        self.rows = []

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.done.is_set():
            try:
                raw = subprocess.check_output([
                    'nvidia-smi', '-i', self.uuid, '--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                memory, total, utilization = map(float, raw.split(','))
                self.rows.append(dict(time=time.time(), gpu_memory_mib=memory,
                                      gpu_total_mib=total, gpu_utilization_percent=utilization,
                                      load_average=os.getloadavg(),
                                      allocated_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                                      **owned_process_usage()))
            except Exception as error:
                self.rows.append(dict(time=time.time(), monitor_error=str(error)))
            self.done.wait(3)

    def finish(self):
        self.done.set()
        self.thread.join(15)
        write(RESULTS / 'metadata/performance' / (self.name + '.json'), self.rows)
        rows = [r for r in self.rows if 'gpu_memory_mib' in r]
        return dict(peak_gpu_memory_mib=max((r['gpu_memory_mib'] for r in rows), default=0),
                    mean_gpu_utilization_percent=sum(r['gpu_utilization_percent'] for r in rows) / max(1, len(rows)))


def child(tasks, replies, videos, foundationpose, tag, frames):
    os.setsid()
    log = RESULTS / 'metadata/workers' / f'{tag}-{os.getpid()}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        runner = None
        replies.put(('ready', os.getpid(), None))
        while True:
            case = tasks.get()
            if case is None:
                return
            started = time.monotonic()
            print('VIDEO_STARTED', case, os.getpid(), flush=True)
            geometry_seconds = prepare_case(case, videos)
            initialization_seconds = 0.
            if runner is None:
                from worker import Runner
                if (RESULTS / 'metadata/performance/optimization-accepted.json').exists():
                    from optimized_execution import configure
                    configure()
                initialization_started = time.monotonic()
                runner = Runner(videos, foundationpose)
                import faiss
                faiss.omp_set_num_threads(2)
                initialization_seconds = time.monotonic() - initialization_started
            report = runner.run(case, frames=frames, tag=tag if frames else None)
            report.update(geometry_seconds=geometry_seconds,
                          model_initialization_seconds=initialization_seconds,
                          end_to_end_seconds=time.monotonic() - started)
            write(RESULTS / 'cases' / case / 'complete.json', report)
            print('VIDEO_END_TO_END_COMPLETE', case, report['end_to_end_seconds'], flush=True)
            replies.put(('complete', case, report))
    except BaseException:
        detail = traceback.format_exc()
        print(detail, flush=True)
        replies.put(('error', os.getpid(), detail))
        raise


def phase(args, cases, workers, name, frames, gpu_uuid):
    context = mp.get_context('spawn')
    tasks, replies = context.Queue(), context.Queue()
    processes = []
    monitor = Monitor(gpu_uuid, name)
    monitor.start()
    setup_start = time.monotonic()
    reports = []
    try:
        for _ in range(workers):
            process = context.Process(target=child, args=(tasks, replies, args.videos, args.foundationpose, name, frames))
            process.start()
            processes.append(process)
            ACTIVE.append(process)
        ready = 0
        while ready < workers:
            try:
                kind, key, value = replies.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode is not None for p in processes):
                    raise RuntimeError('Worker died during initialization')
                continue
            if kind != 'ready':
                raise RuntimeError(str(value))
            ready += 1
        setup_seconds = time.monotonic() - setup_start
        start = time.monotonic()
        for case in cases:
            tasks.put(case)
        for _ in processes:
            tasks.put(None)
        while len(reports) < len(cases):
            try:
                kind, key, value = replies.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(not p.is_alive() for p in processes):
                    raise RuntimeError('Worker died before completing its assigned videos')
                continue
            if kind != 'complete':
                raise RuntimeError(str(value))
            reports.append(value)
            print('PHASE_PROGRESS', name, len(reports), '/', len(cases), key, flush=True)
        seconds = time.monotonic() - start
        for process in processes:
            process.join(30)
            if process.exitcode != 0:
                raise RuntimeError('Worker failed to exit cleanly')
        frame_links = sum(len(r['timings']) for r in reports)
        result = dict(name=name, workers=workers, gpu_count=1, setup_seconds=setup_seconds,
                      processing_seconds=seconds, frame_links=frame_links,
                      frame_links_per_second=frame_links / seconds, reports=reports,
                      **monitor.finish())
        write(RESULTS / 'metadata/performance' / (name + '-summary.json'), result)
        return result
    finally:
        monitor.finish()
        for process in processes:
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.join(10)
                if process.is_alive():
                    os.killpg(process.pid, signal.SIGKILL)
                    process.join()
            ACTIVE.remove(process)


def prepare_case(case, videos):
    started = time.monotonic()
    dependency = Path('/PHShome/zy992/Wilson/dependency')
    log = RESULTS / 'metadata/workers' / (f"geometry-{os.environ['SLURM_JOB_ID']}-{case}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PATH'] = str(dependency / 'env/geometry/bin') + ':' + env['PATH']
    command = [str(dependency / 'env/geometry/bin/python'), '-u', str(ROOT / 'prepare_geometry.py'),
               '--case', case, '--videos', videos,
               '--previous', '/PHShome/zy992/Wilson/deformationdetection/workspace-link5-pair-selection-20261007-v2/results/link5_pair_selection_ablation_v1',
               '--mega', str(dependency / 'source/infrastructure/vendor/mega_sam')]
    with log.open('w') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Geometry failed for ' + case + '; see ' + str(log))
    return time.monotonic() - started


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--videos', required=True)
    p.add_argument('--foundationpose', required=True)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, stop_owned)
    signal.signal(signal.SIGINT, stop_owned)
    assert os.environ.get('SLURM_JOB_ID')
    preflight = json.loads((RESULTS / 'metadata/preflight.json').read_text())
    assert preflight['status'] == 'passed'
    assert preflight['manifest_sha256'] == sha(RESULTS / 'inputs/manifest.json')
    assert preflight['code_sha256'] == code_sha()
    assert preflight['checkpoint_sha256'] == sha(ROOT / 'models/reg2inv-real3dad.pth.tar')
    optimization_receipt = RESULTS / 'metadata/performance/optimization-accepted.json'
    if optimization_receipt.exists():
        from optimized_execution import execution_sha
        optimization = json.loads(optimization_receipt.read_text())
        assert optimization['status'] == 'accepted'
        assert optimization['code_sha256'] == code_sha()
        assert optimization['execution_sha256'] == execution_sha()
    for relative, expected in preflight['foundationpose_weights'].items():
        assert sha(Path(args.foundationpose) / relative) == expected
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid)
    gpu_bytes = properties.total_memory
    cpu_count = int(os.environ['SLURM_CPUS_PER_TASK'])
    workers = 5  # Latest user instruction: exactly five end-to-end video lanes.
    assert cpu_count >= workers * 2
    cases = [e['video_id'] for e in manifest()]
    write(RESULTS / 'metadata/allocation.json',
          dict(status='running', job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
               gpu_count=1, gpu_uuid=uuid, gpu_name=properties.name,
               selected_workers=workers, scheduling='five end-to-end video workers, user requested',
               allocated_cpus=cpu_count, total_video_workers=len(cases),
               capacity_trials=[], gpu_total_bytes=gpu_bytes))
    print('CONCURRENCY_SELECTED', workers, 'END_TO_END', flush=True)
    performance = phase(args, cases, workers, 'full', None, uuid)
    from summarize import summarize
    summary = summarize()
    write(RESULTS / 'metadata/completion.json', dict(status='completed', summary=summary,
                                                   performance=performance,
                                                   job_id=os.environ['SLURM_JOB_ID']))
    print('EXPERIMENT_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
