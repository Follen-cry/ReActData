"""c008: c006/c007-style process rows (8 plain final rows + 8 process rows per step, target in the F phase), where 4
of the 8 process rows additionally show an earlier F-present frame (34..42) as ViT-only context and target a later
frame (>= ctx+4, up to 56). Final rows never have context, so the test-time mapping stays exactly as in the baseline.
"""
import random


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.seed = seed
        self.n_proc = int(config.get("n_proc", 8))
        self.n_ctx = int(config.get("n_ctx", 4))
        self.lo = int(config.get("late_lo", 36))
        self.hi = int(config.get("late_hi", 52))
        self.ctx_lo = int(config.get("ctx_lo", 34))
        self.ctx_hi = int(config.get("ctx_hi", 42))
        self.gap = int(config.get("gap", 4))
        self.pos = {"f": 0, "s": 0}

    def _perm(self, kind, k):
        ids = list(self.ids)
        random.Random(f"{self.seed}:{kind}:{k}").shuffle(ids)
        return ids

    def _take(self, kind, n):
        out = []
        for _ in range(n):
            k, i = divmod(self.pos[kind], len(self.ids))
            out.append(self._perm(kind, k)[i])
            self.pos[kind] += 1
        return out

    def next_batch(self, step):
        rng = random.Random(f"{self.seed}:proc:{step}")
        rows = [{"clip_id": c, "context_frames": [], "template_id": "eval_format"}
                for c in self._take("f", 16 - self.n_proc)]
        for j, c in enumerate(self._take("s", self.n_proc)):
            if j < self.n_ctx:
                cf = rng.randint(self.ctx_lo, self.ctx_hi)
                tf = rng.randint(cf + self.gap, max(cf + self.gap, self.hi + 4))
                rows.append({"clip_id": c, "context_frames": [cf], "template_id": "late_ctx", "target_frame": min(tf, 56)})
            else:
                rows.append({"clip_id": c, "context_frames": [], "template_id": "late",
                             "target_frame": rng.randint(self.lo, self.hi)})
        random.Random(f"{self.seed}:mix:{step}").shuffle(rows)
        return rows

    def observe(self, step, summary):
        pass

    def state_dict(self):
        return {"pos": dict(self.pos)}

    def load_state_dict(self, state):
        self.pos = dict(state["pos"])
