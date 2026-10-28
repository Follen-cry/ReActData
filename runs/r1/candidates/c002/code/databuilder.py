"""c002: baseline (all rows target the final frame) where, for steps < ctx_until, n_ctx of the 16 rows additionally
show an early process frame (E recoloured, F not yet present) as ViT-only context. All targets are final.
After ctx_until all rows are plain baseline rows.
"""
import random


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.seed = seed
        self.ctx_until = int(config.get("ctx_until", 60))
        self.ctx_frame = int(config.get("ctx_frame", 27))
        self.n_ctx = int(config.get("n_ctx", 8))
        self.pos = 0

    def _perm(self, k):
        ids = list(self.ids)
        random.Random(f"{self.seed}:perm:{k}").shuffle(ids)
        return ids

    def next_batch(self, step):
        n_ctx = self.n_ctx if step < self.ctx_until else 0
        rows = []
        for j in range(16):
            k, i = divmod(self.pos, len(self.ids))
            cid = self._perm(k)[i]
            self.pos += 1
            if j < n_ctx:
                rows.append({"clip_id": cid, "context_frames": [self.ctx_frame], "template_id": "ctx_final"})
            else:
                rows.append({"clip_id": cid, "context_frames": [], "template_id": "eval_format"})
        random.Random(f"{self.seed}:mix:{step}").shuffle(rows)
        return rows

    def observe(self, step, summary):
        pass

    def state_dict(self):
        return {"pos": self.pos}

    def load_state_dict(self, state):
        self.pos = state["pos"]
