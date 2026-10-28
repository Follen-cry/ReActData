"""Recipe -> rendered text -> internvlu_train Row (Runner-only; the only place training rows are made).

F0 and a final-frame target are the clip's PNG endpoints (512x512); context frames and earlier targets (v4) are the clip's
colour-corrected mp4 frames (512x512), all from <DATA>/frames512 (runner/build_data.py). cond_image is always F0.
"""
from __future__ import annotations

import functools
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import API, DATA, FRAMES, MANIFESTS, read_jsonl, wrap_prompt  # noqa: E402
sys.path.insert(0, API)
from recipe import FPS, N_FRAMES  # noqa: E402
from text_slots import check_templates, render  # noqa: E402


class Materializer:
    def __init__(self, candidate_dir: str) -> None:
        self.templates = check_templates(yaml.safe_load(open(os.path.join(candidate_dir, "text_templates.yaml"))))
        self.template_ids = set(self.templates)
        cy = os.path.join(candidate_dir, "candidate.yaml")
        self.candidate_yaml = yaml.safe_load(open(cy)) if os.path.exists(cy) else {}
        pool = read_jsonl(os.path.join(MANIFESTS, "train_pool.jsonl"))
        probe = read_jsonl(os.path.join(MANIFESTS, "inner_probe.jsonl"))
        self.clips = {r["clip_id"]: r for r in pool + probe}
        self.clip_ids = {r["clip_id"] for r in pool}               # trainable clips (inner probe excluded)
        self.probe_ids = sorted(r["clip_id"] for r in probe)
        self.probe_eval_rows = read_jsonl(os.path.join(MANIFESTS, "inner_probe_eval.jsonl"))
        self.pool_view_path = os.path.join(DATA, "pool_view.json")

    @staticmethod
    def frame(clip_id: str, t: int) -> str:
        if t == 0:
            return os.path.join(FRAMES, clip_id, "f0.png")
        return os.path.join(FRAMES, clip_id, f"mp4_{t:02d}.png")

    @functools.lru_cache(maxsize=None)
    def _wrapped(self, task: str, desc: str) -> str:
        return wrap_prompt(task, desc)

    def render(self, r: dict) -> str:
        c = self.clips[r["clip_id"]]
        return render(self.templates[r["template_id"]], description=c["prompt"], wrapped_description=self._wrapped(c["task"], c["prompt"]),
                      context_frames=r["context_frames"], n_frames=N_FRAMES, fps=FPS, target_frame=r.get("target_frame", N_FRAMES - 1))

    def row(self, r: dict, rid: str):
        from internvlu_train.data import Row, ensure_gen_token     # lazy: render() is also used by the CPU checks
        cid = r["clip_id"]
        f0 = self.frame(cid, 0)
        imgs = [f0] + [self.frame(cid, t) for t in r["context_frames"]]
        conv = [{"from": "human", "value": self.render(r)}, {"from": "gpt", "value": "<img>"}]
        tgt = r.get("target_frame", N_FRAMES - 1)
        target = os.path.join(FRAMES, cid, "final.png") if tgt == N_FRAMES - 1 else self.frame(cid, tgt)   # PNG final / corrected mp4
        return Row(id=f"{cid}|{rid}", conversations=ensure_gen_token(conv), images=imgs, target_image=target, cond_image=f0, raw=dict(r))
