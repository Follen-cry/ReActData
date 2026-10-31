"""Host-owned campaign accounting and deadline-bound subprocess execution.

All real experiments in one kickoff must reference the same pre-existing ledger.
This is coordination for trusted code, not a security boundary.
"""
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from contextlib import contextmanager
from storage import read, write


def utc():
    return dt.datetime.now(dt.timezone.utc)


def deadline(c):
    value = dt.datetime.fromisoformat(c['deadline'].replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('campaign deadline must have a timezone')
    return value


@contextmanager
def ledger(path):
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError('existing campaign ledger required')
    with open(str(p) + '.lock', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        c = read(p)
        yield c
        write(p, c)


def register(path, root, trial_id, request_id, estimated_seconds):
    with ledger(path) as c:
        if c['training_budget'] > 7 or c['max_concurrent_gpus'] > 4:
            raise ValueError('campaign exceeds hard limits')
        if len(c['training_attempts']) >= c['training_budget']:
            raise ValueError('campaign training budget exhausted')
        if any(a['status'] in ('preparing', 'running') for a in c['training_attempts']):
            raise ValueError('another campaign attempt is active or requires inspection')
        if estimated_seconds <= 0:
            raise ValueError('positive training estimate required')
        if (deadline(c) - utc()).total_seconds() < estimated_seconds + c['report_reserve_seconds']:
            raise ValueError('insufficient time for trial and final test/report reserve')
        identity = str(Path(root).resolve()) + ':' + request_id
        if any(a['identity'] == identity for a in c['training_attempts']):
            raise ValueError('attempt already registered; inspect campaign and state')
        c['training_attempts'].append(dict(identity=identity, trial_id=trial_id,
            status='preparing', started_at=utc().isoformat(), root=str(Path(root).resolve())))
        c['consumed'] = len(c['training_attempts'])
        c['remaining'] = c['training_budget'] - c['consumed']


def update(path, root, request_id, status):
    with ledger(path) as c:
        identity = str(Path(root).resolve()) + ':' + request_id
        a = next(a for a in c['training_attempts'] if a['identity'] == identity)
        a['status'] = status
        if status in ('failed', 'completed', 'interrupted'):
            a['ended_at'] = utc().isoformat()


def _run_process(path, argv, log, *, reserve_seconds=0, timeout_seconds=None):
    """Launch one owned process group, enforce deadline, restrict GPU visibility."""
    with ledger(path) as c:
        devices = c['gpu_allocation']
        if not devices or len(devices) > min(4, c['max_concurrent_gpus']) or len(set(devices)) != len(devices):
            raise ValueError('invalid allocated GPU list')
        seconds = (deadline(c) - utc()).total_seconds() - reserve_seconds
        if timeout_seconds is not None:
            if timeout_seconds<=0: raise ValueError('positive process timeout required')
            seconds=min(seconds,timeout_seconds)
        if seconds <= 0:
            raise TimeoutError('campaign deadline or reserve reached')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(len(devices)) if os.environ.get('VIDEO_SSL_CONTAINER')=='1' else devices))
    with open(log, 'ab') as output:
        proc = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return proc.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, BaseException):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
                # The leader may exit before its children. Kill any surviving group.
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
            raise


def run_process(path, argv, log, *, reserve_seconds=0, timeout_seconds=None):
    # One GPU process group across every experiment sharing this kickoff ledger.
    with open(str(Path(path).resolve()) + '.gpu.lock', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_process(path, argv, log, reserve_seconds=reserve_seconds,timeout_seconds=timeout_seconds)
