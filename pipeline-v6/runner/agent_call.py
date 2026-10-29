"""Builder Agent call: `claude -p --model sonnet` inside bwrap.

The sandbox sees only: system files needed to run and reach the API (/usr, a few /etc files), the claude
executable, a throw-away HOME holding a copy of the OAuth credentials, the round workspace (/work, writable) and the
round inputs (/input, read-only). The scorer, Runner, manifests, data, metadata, test set, the real ~/.claude
(sessions, memory) and every other path are not mounted. Tools: Read/Edit/Write only, restricted to /work and /input.

Session continuity (protocol v2): with ``home`` the agent HOME is a persistent, Runner-only directory of the run
(it keeps the Claude Code session files; the agent's tools cannot read it), and ``resume`` continues an earlier
session (``claude -p --resume <session_id>``). /work is a different host directory per candidate but always mounted
at /work, so the resumed session's project path is the same.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Dict, Optional

CLAUDE = os.path.realpath(os.path.expanduser("~/.local/bin/claude"))
CRED = os.path.expanduser("~/.claude/.credentials.json")
ETC = ["/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/ssl", "/etc/pki", "/etc/ca-certificates",
       "/etc/localtime", "/etc/passwd", "/etc/group", "/etc/ld.so.cache", "/etc/host.conf", "/etc/gai.conf"]
MODEL = "sonnet"


def _bwrap(home: str, work: str, inp: str) -> list:
    cmd = ["/usr/bin/bwrap", "--unshare-all", "--share-net", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for e in ETC:
        if os.path.exists(e):
            cmd += ["--ro-bind", e, e]
    cmd += ["--ro-bind", CLAUDE, "/opt/claude/claude",
            "--bind", home, "/home/agent", "--bind", work, "/work", "--ro-bind", inp, "/input", "--chdir", "/work",
            "--setenv", "HOME", "/home/agent", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
            "--setenv", "DISABLE_AUTOUPDATER", "1", "--setenv", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"]
    return cmd


def call_agent(prompt: str, work: str, inp: str, log_dir: str, *, timeout_s: int = 1800, max_turns: int = 80,
               home: Optional[str] = None, resume: Optional[str] = None) -> Dict:
    """Runs one agent session (or continues ``resume``); returns {"ok", "result", "model_usage", "models", "cost_usd",
    "num_turns", "wall_s", "session_id", "resumed_from", ...}."""
    os.makedirs(log_dir, exist_ok=True)
    persistent = home is not None
    if persistent:
        os.makedirs(home, exist_ok=True)
    else:
        home = tempfile.mkdtemp(prefix="agent_home_", dir=log_dir)
    os.makedirs(os.path.join(home, ".claude"), exist_ok=True)
    shutil.copy2(CRED, os.path.join(home, ".claude", ".credentials.json"))      # fresh credentials every call
    if not os.path.exists(os.path.join(home, ".claude.json")):
        with open(os.path.join(home, ".claude.json"), "w") as f:
            json.dump({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": False}, f)
    if prompt.lstrip().startswith("-"):          # the CLI would parse a leading dash as an option
        prompt = "Message from the Runner:\n" + prompt
    args = ["/opt/claude/claude", "-p", prompt, "--model", MODEL, "--output-format", "json",
            "--tools", "Read,Edit,Write", "--allowedTools", "Read(//work/**)", "Edit(//work/**)", "Write(//work/**)",
            "Read(//input/**)", "--permission-mode", "acceptEdits", "--max-turns", str(max_turns)]
    if resume:
        args += ["--resume", resume]
    cmd = _bwrap(home, os.path.abspath(work), os.path.abspath(inp)) + args
    with open(os.path.join(log_dir, "cmd.json"), "w") as f:
        json.dump({"argv": cmd[:-len(args)] + [a if a != prompt else "<prompt.md>" for a in args], "claude": CLAUDE}, f, indent=1)
    with open(os.path.join(log_dir, "prompt.md"), "w") as f:
        f.write(prompt)
    t0 = time.time()
    cred_mtime = os.path.getmtime(os.path.join(home, ".claude", ".credentials.json"))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env={})
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = -9, e.stdout or "", (e.stderr or "") + "\nTIMEOUT"
        out = out.decode() if isinstance(out, bytes) else out
        err = err.decode() if isinstance(err, bytes) else err
    wall = time.time() - t0
    # keep the user's login valid if the sandbox refreshed the OAuth token
    sb = os.path.join(home, ".claude", ".credentials.json")
    try:
        if os.path.getmtime(sb) > cred_mtime:
            json.load(open(sb))
            if os.path.getmtime(CRED) < os.path.getmtime(sb):
                shutil.copy2(sb, CRED)
    except (OSError, ValueError):
        pass
    with open(os.path.join(log_dir, "stdout.json"), "w") as f:
        f.write(out)
    with open(os.path.join(log_dir, "stderr.txt"), "w") as f:
        f.write(err[-20000:])
    if not persistent:
        shutil.rmtree(home, ignore_errors=True)
    res = {"ok": False, "returncode": rc, "wall_s": wall, "resumed_from": resume}
    try:
        j = json.loads(out)
        mu = j.get("modelUsage") or {}
        res.update({"ok": rc == 0 and not j.get("is_error", False), "result": j.get("result"), "model_usage": mu,
                    "models": sorted(mu), "cost_usd": j.get("total_cost_usd"), "num_turns": j.get("num_turns"),
                    "session_id": j.get("session_id"), "subtype": j.get("subtype")})
    except ValueError:
        res["error"] = (err or out)[-2000:]
    with open(os.path.join(log_dir, "output.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res
