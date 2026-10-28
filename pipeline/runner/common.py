"""Paths, hashing and small I/O helpers shared by every Runner module (reactdata-v5: copied from reactdata-v4)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Iterable, List

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # internvlu-exp/exp/reactdata-v1
REPO = os.path.dirname(os.path.dirname(EXP))                               # internvlu-exp
SSL = os.path.dirname(REPO)                                                # ssl_mllm
SCRATCH = "/scratch/network/ssd/junlin/reactdata-v5"
DATA = os.path.join(SCRATCH, "data")
RUNS = os.path.join(SCRATCH, "runs")
FRAMES = os.path.join(DATA, "frames512")
MANIFESTS = os.path.join(EXP, "manifests")
API = os.path.join(EXP, "api")

RAW_ROOT = "/scratch/network/ssd/junlin/vbvr_next_frame/raw"
RAW_MANIFEST = "/scratch/network/ssd/junlin/raw/vbvr/manifest_all.jsonl"
TP_ROOT = "/scratch/network/ssd/junlin/vbvr_target_pred"
CVLIB_DIR = os.path.join(SSL, "Evaluation", "VBVR-CustomEval", "scorers")      # cvlib.py (shared CV helpers)
SCORER_DIR = os.path.join(EXP, "scorers")                                         # reactdata-v1 task scorers (Runner-only)
WRAP_FILE = os.path.join(SSL, "VBVR-DataGeneration", "target_pred", "vbvr_target_pred_prompt_wrap.py")
TRAIN_SRC = os.path.join(REPO, "src", "internvlu-train")
PY = os.path.expanduser("~/miniconda3/envs/internvlu/bin/python")
TRAIN_TASK = "shape_color_then_scale"
TASKS = ("shape_color_then_scale", "shape_color_then_move")
N_FRAMES = 60
GLOBAL_BATCH = 16
MAX_STEPS = 100


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def sha256_tree(paths: Iterable[str]) -> str:
    """Hash of several files (sorted by path, path names included)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def now() -> float:
    return time.time()


import threading

_WRAP = None
_WRAP_LOCK = threading.Lock()


def wrap_prompt(task: str, raw: str) -> str:
    """The evaluation prompt: VBVR target_pred universal wrapper + the task's cleanup rules (shape_color_then_move
    drops "Show both transformations sequentially in the video."); audited for residual video language."""
    global _WRAP
    with _WRAP_LOCK:                 # row building is multi-threaded: publish the module only once fully loaded
        if _WRAP is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("vbvr_wrap", WRAP_FILE)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _WRAP = mod
    text = _WRAP.wrap_prompt(task, raw)
    task_part = text.split("Task: ", 1)[1]
    hits = _WRAP.find_residual_video_language(task_part)
    if hits:
        raise ValueError(f"{task}: residual video language {hits} in {task_part[:120]!r}")
    return text
