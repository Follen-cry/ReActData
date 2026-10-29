"""Paths, hashing and small I/O helpers shared by every Runner module (reactdata-v6: copied from reactdata-v5)."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.runtime.io import sha256_file, md5_file, sha256_tree, sha256_json, read_jsonl, write_jsonl, write_json, read_json, now

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # this pipeline/ directory
# The pipeline used to live at internvlu-exp/exp/<name>/ and derived REPO from its own location; a standalone checkout
# (e.g. the ReActData repo) has to name the internvlu-exp repo and its scratch root explicitly.
_DERIVED_REPO = os.path.dirname(os.path.dirname(EXP))
REPO = os.environ.get("REACTDATA_REPO") or (
    _DERIVED_REPO if os.path.isdir(os.path.join(_DERIVED_REPO, "src", "internvlu-train"))
    else "/scratch/network/ssd2/junlin/ssl_mllm/internvlu-exp")                  # internvlu-exp
SSL = os.path.dirname(REPO)                                                # ssl_mllm
SCRATCH = os.environ.get("REACTDATA_SCRATCH", "/scratch/network/ssd/junlin/reactdata-v6")
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
TRAIN_TASKS = ("shape_color_then_scale", "shape_color_then_move", "2d_geometric_transformation")
TASKS = TRAIN_TASKS
N_FRAMES = 60
GLOBAL_BATCH = 16
MAX_STEPS = 8


def implementation_files() -> List[str]:
    """All trusted implementations, including modules extracted out of runner/."""
    from experiment.runtime.provenance import implementation_files as collect
    return collect(EXP)


def implementation_hash() -> str:
    # Include relative paths (unlike the historical basename-only tree hash).
    from experiment.runtime.provenance import implementation_hash as digest
    return digest(EXP)


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
