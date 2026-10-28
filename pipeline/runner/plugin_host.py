"""Runner side of the plugin sandbox: starts plugin_worker.py under bwrap and talks JSON lines to it.

The sandbox mounts only the system libraries, the Python env (read-only), api/ (read-only), the candidate's code
(read-only), the pool view (read-only) and an empty per-run scratch dir; no network, no other file system, no
environment variables. Every call has a wall-time limit (api/plugin_api.STEP_TIME_LIMIT_S, longer for init).
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import time
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import API, PY  # noqa: E402
sys.path.insert(0, API)

ENV_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(PY)))     # .../envs/internvlu


class PluginError(RuntimeError):
    pass


def bwrap_cmd(cand_dir: str, pool_path: str, scratch: str) -> list:
    return ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
            "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", ENV_ROOT, ENV_ROOT,
            "--ro-bind", API, "/api",
            "--ro-bind", os.path.join(HERE, "plugin_worker.py"), "/runner/plugin_worker.py",
            "--ro-bind", cand_dir, "/cand",
            "--ro-bind", pool_path, "/data/pool_view.json",
            "--bind", scratch, "/scratch_plugin", "--chdir", "/scratch_plugin",
            "--setenv", "PYTHONHASHSEED", "0", "--setenv", "HOME", "/scratch_plugin",
            "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "MKL_NUM_THREADS", "1",
            os.path.join(ENV_ROOT, "bin", "python3"), "-I", "/runner/plugin_worker.py"]


class PluginHost:
    def __init__(self, cand_dir: str, pool_path: str, scratch: str, log_path: str, *, clear_scratch: bool = True,
                 time_limit: Optional[float] = None) -> None:
        from plugin_api import STEP_TIME_LIMIT_S
        self.time_limit = time_limit or STEP_TIME_LIMIT_S
        if clear_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch, exist_ok=True)
        self.log = open(log_path, "a")
        self.proc = subprocess.Popen(bwrap_cmd(os.path.abspath(cand_dir), os.path.abspath(pool_path), os.path.abspath(scratch)),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                     env={})       # empty environment for bwrap and the worker
        self.calls = 0
        self.seconds = 0.0

    def call(self, op: str, timeout: Optional[float] = None, **payload) -> Any:
        if self.proc.poll() is not None:
            raise PluginError(f"plugin process exited with {self.proc.returncode}")
        t0 = time.time()
        self.proc.stdin.write(json.dumps({"op": op, **payload}) + "\n")
        self.proc.stdin.flush()
        limit = timeout if timeout is not None else self.time_limit
        r, _, _ = select.select([self.proc.stdout], [], [], limit)
        if not r:
            self.close()
            raise PluginError(f"plugin call {op} exceeded {limit:.0f} s")
        line = self.proc.stdout.readline()
        dt = time.time() - t0
        self.calls += 1
        self.seconds += dt
        if not line:
            raise PluginError(f"plugin process died during {op} (see plugin log)")
        msg = json.loads(line)
        if not msg["ok"]:
            raise PluginError(f"plugin error in {op}:\n{msg['error']}")
        return msg["result"]

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(5)
        finally:
            self.log.close()
