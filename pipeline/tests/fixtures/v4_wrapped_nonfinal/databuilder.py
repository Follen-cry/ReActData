"""Regression candidate (v4): non-final targets with the evaluation-wrapper template; checks must reject it."""
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
            if j % 2 == 0:
                ctx = []
                out.append({"clip_id": cid, "context_frames": ctx, "template_id": "eval_format", "target_frame": 28})
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
