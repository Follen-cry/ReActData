"""GPU slots and job launching (Runner-only).

A job = one train_entry.py process group on one 4-GPU slot. It is started detached (nohup, exit code written to
<run_dir>/exit_code) so an ssh drop does not kill it; the Runner polls exit_code / progress.json. Before a launch the
slot's GPUs must be idle (< 2000 MiB used, no compute processes), so existing jobs of other users are never touched.
"""
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, REPO, TRAIN_SRC, read_json  # noqa: E402

TORCHRUN = os.path.expanduser("~/miniconda3/envs/internvlu/bin/torchrun")
LOCAL = socket.gethostname().split(".")[0]


def sh(node: str, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    if node == LOCAL:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=90", node, cmd],
                          capture_output=True, text=True, timeout=timeout)


def gpus_idle(node: str, gpus: str) -> bool:
    try:
        r = sh(node, f"nvidia-smi -i {gpus} --query-gpu=memory.used --format=csv,noheader,nounits; "
                     f"nvidia-smi -i {gpus} --query-compute-apps=pid --format=csv,noheader", timeout=150)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode != 0:
        return False
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    n = len(gpus.split(","))
    mem, apps = lines[:n], lines[n:]
    return len(mem) == n and all(int(m) < 2000 for m in mem) and not apps


class Slot:
    def __init__(self, name: str, node: str, gpus: str, port: int) -> None:
        self.name, self.node, self.gpus, self.port = name, node, gpus, port
        self.job: Optional[Dict] = None

    def to_json(self):
        return {"name": self.name, "node": self.node, "gpus": self.gpus, "port": self.port}


def launch(slot: Slot, run_json: str) -> Dict:
    run = read_json(run_json)
    d = run["run_dir"]
    os.makedirs(d, exist_ok=True)
    for f in ("exit_code", "pid"):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    env = (f"PYTHONPATH={TRAIN_SRC}:{os.path.dirname(TRAIN_SRC)} CUDA_VISIBLE_DEVICES={slot.gpus} "
           f"TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8")
    for k in ("REACTDATA_REPO", "REACTDATA_SCRATCH"):     # path overrides (runner/common.py) must reach the remote job too
        if os.environ.get(k):
            env += f" {k}={shlex.quote(os.environ[k])}"
    inner = (f"cd {shlex.quote(EXP)} && {env} {TORCHRUN} --nproc_per_node={len(slot.gpus.split(','))} "
             f"--master_port {slot.port} runner/train_entry.py --spec {shlex.quote(run_json)} >> {shlex.quote(d)}/train.log 2>&1; "
             f"echo $? > {shlex.quote(d)}/exit_code")
    cmd = f"nohup setsid bash -c {shlex.quote(inner)} > /dev/null 2>&1 & echo $! > {shlex.quote(d)}/pid"
    r = sh(slot.node, cmd, timeout=150)
    if r.returncode != 0:
        raise RuntimeError(f"launch on {slot.node} failed: {r.stderr[-500:]}")
    job = {"run_json": run_json, "run_dir": d, "slot": slot.to_json(), "start": time.time()}
    slot.job = job
    return job


LIVENESS_EVERY_S = 300


def alive(job: Dict) -> bool:
    """Is the job's process group still alive on its node? (pid file = the setsid wrapper = the process group id)."""
    pidf = os.path.join(job["run_dir"], "pid")
    if not os.path.exists(pidf):
        return True                                    # not started yet / pid not written: assume alive
    pid = open(pidf).read().strip()
    try:
        r = sh(job["slot"]["node"], f"pgrep -g {pid} > /dev/null && echo alive || echo dead", timeout=150)
    except subprocess.TimeoutExpired:
        return True                                    # cannot tell: do not declare a failure
    return "dead" not in r.stdout


def status(job: Dict) -> str:
    """running | done | candidate_failed | infra_failed

    The wrapper writes exit_code when torchrun returns. A wrapper that dies without writing it (seen on torrnode15,
    2026-10-27) must not block the Runner: DONE.json (the last file train_entry writes) means done; otherwise the process
    group is checked every LIVENESS_EVERY_S and a dead job without exit_code is an infrastructure failure."""
    d = job["run_dir"]
    ec = os.path.join(d, "exit_code")
    if not os.path.exists(ec):
        if os.path.exists(os.path.join(d, "DONE.json")):
            return "done"
        now = time.time()
        if now - job.get("_last_alive_check", job["start"]) >= LIVENESS_EVERY_S:
            job["_last_alive_check"] = now
            if not alive(job):
                if os.path.exists(os.path.join(d, "FAILED.json")):
                    return "candidate_failed"
                return "infra_failed"
        return "running"
    code = open(ec).read().strip()
    if code == "0" and os.path.exists(os.path.join(d, "DONE.json")):
        return "done"
    if os.path.exists(os.path.join(d, "FAILED.json")):
        return "candidate_failed"
    return "infra_failed"


def kill(job: Dict) -> None:
    slot = job["slot"]
    pidf = os.path.join(job["run_dir"], "pid")
    if os.path.exists(pidf):
        pid = open(pidf).read().strip()
        sh(slot["node"], f"pkill -TERM -g {pid} || true; sleep 5; pkill -KILL -g {pid} || true", timeout=60)


def progress(job: Dict) -> Dict:
    p = os.path.join(job["run_dir"], "progress.json")
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}
