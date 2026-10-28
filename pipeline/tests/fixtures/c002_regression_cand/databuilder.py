"""Regression candidate (reactdata-v0 c002 failure): context frames with a template lacking {context_list}; checks must reject it."""
"""Test candidate (not a search candidate): exercises context frames, two templates, feedback and probes."""
import random


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.rng_seed = seed
        self.use_ctx = True

    def next_batch(self, step):
        rng = random.Random(f"{self.rng_seed}:{step}")
        out = []
        for j in range(16):
            cid = rng.choice(self.ids)
            if self.use_ctx and j % 2 == 0:
                out.append({"clip_id": cid, "context_frames": sorted(rng.sample(range(10, 49), 1)), "template_id": "eval_format"})
            else:
                out.append({"clip_id": cid, "context_frames": [], "template_id": "eval_format"})
        return out

    def observe(self, step, summary):
        if "ctx_gap" in summary:
            self.use_ctx = summary["ctx_gap"] < 0.05

    def state_dict(self):
        return {"use_ctx": self.use_ctx}

    def load_state_dict(self, state):
        self.use_ctx = state["use_ctx"]
