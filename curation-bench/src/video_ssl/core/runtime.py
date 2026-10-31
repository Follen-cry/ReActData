"""Shared storage, integrity, locking and process helpers."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))



def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)



def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def resolve(base, value):
    p = Path(value).expanduser()
    return (base / p).resolve() if not p.is_absolute() else p.resolve()



def keys(value, allowed, where):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Invalid/unknown keys in {where}; allowed: {allowed}")



def load_plan(prompt_path, **kwargs):
    from .layout import load_plan as resolve_layout
    return resolve_layout(prompt_path, **kwargs)



def verify(plan):
    for path, expected in plan["hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")



def process(argv, log, *, cwd=None, prompt=None, timeout=None, env=None, on_start=None):
    """No shell interpolation; on timeout terminate the complete owned process group."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE if prompt is not None else subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                start_new_session=True, env=env)
        try:
            if on_start is not None:
                on_start(proc.pid)
            proc.communicate(input=prompt, timeout=timeout)
        except BaseException:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.wait()
            raise
        if proc.returncode:
            raise RuntimeError(f"Command exited {proc.returncode}; see {log}")



def agent_command(agent):
    cmd = [agent.get("executable", "codex"), "exec", "--dangerously-bypass-approvals-and-sandbox",
           "--skip-git-repo-check", "--json"]
    if agent.get("model"):
        cmd += ["--model", agent["model"]]
    if agent.get("reasoning_effort"):
        cmd += ["-c", "model_reasoning_effort=" + json.dumps(agent["reasoning_effort"])]
    return cmd + ["-"]



def selection_delta(before, after):
    old = [r["source_video_id"] for r in before]
    new = [r["source_video_id"] for r in after]
    positions = {sid: i for i, sid in enumerate(old)}
    old_frames = {r["source_video_id"]: {k: r.get(k) for k in ("input_frames", "target_frame")} for r in before}
    new_frames = {r["source_video_id"]: {k: r.get(k) for k in ("input_frames", "target_frame")} for r in after}
    return {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
            "frame_changes": [{"source": sid, "before": old_frames[sid], "after": new_frames[sid]}
                              for sid in new if sid in old_frames and old_frames[sid] != new_frames[sid]],
            "frame_assignments": new_frames,
            "moved": [{"source": sid, "from": positions[sid], "to": i}
                      for i, sid in enumerate(new) if sid in positions and positions[sid] != i],
            "order": new}



def require_linux():
    if os.name != "posix":
        raise ValueError("Real video SSL execution requires Linux (fcntl / CUDA / process groups)")



def tree_hash(root):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlinks forbidden in submitted code/data')
        if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:
            result[str(p.relative_to(root))] = digest(p)
    return result



@contextmanager
def exclusive(root):
    import os
    lock = Path(root) / 'operation.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink()



def process_is_alive(pid):
    """Read-only PID probe; Windows signal 0 is a console control event."""
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
        api.OpenProcess.restype=wintypes.HANDLE
        api.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
        api.CloseHandle.argtypes=(wintypes.HANDLE,)
        handle=api.OpenProcess(0x1000,False,pid)
        if not handle:
            error=ctypes.get_last_error()
            if error in (87,1168): return False
            raise OSError(error,'Cannot inspect Agent process')
        try:
            status=wintypes.DWORD()
            if not api.GetExitCodeProcess(handle,ctypes.byref(status)):
                raise ctypes.WinError(ctypes.get_last_error())
            return status.value==259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
