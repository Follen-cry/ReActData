"""c002: move-heavy mixture, task blocks with move last. Fractions 1/4 scale, 1/8 geo, 5/8 move of the n = steps x 16 rows.

Derived from c001 (random-within-task draw). The quota is scaled to the actual number of rows (c001's fixed 56/56/16
quota summed to 128 and was silently truncated when the run had fewer rows). Order: scale rows, then geo, then move;
the first steps are mostly scale, the last steps are all move.
"""
import random

ROWS_PER_STEP = 16
FRAC = {"shape_color_then_scale": 0.25, "2d_geometric_transformation": 0.125, "shape_color_then_move": 0.625}
BLOCK_ORDER = ["shape_color_then_scale", "2d_geometric_transformation", "shape_color_then_move"]


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        rng = random.Random(f"{seed}:subset")
        n = total_steps * ROWS_PER_STEP
        tasks = sorted({c["task"] for c in pool["clips"]})
        quota = {t: int(round(FRAC.get(t, 0.0) * n)) for t in tasks}
        quota[BLOCK_ORDER[-1]] += n - sum(quota.values())
        order = []
        for t in BLOCK_ORDER:
            ids = sorted(c["clip_id"] for c in pool["clips"] if c["task"] == t)
            order += rng.sample(ids, quota[t])
        self.order = order
        self.pos = 0

    def next_batch(self, step):
        out = [{"clip_id": c, "context_frames": [], "template_id": "eval_format"} for c in self.order[self.pos:self.pos + ROWS_PER_STEP]]
        self.pos += ROWS_PER_STEP
        return out

    def observe(self, step, summary):
        pass

    def state_dict(self):
        return {"pos": self.pos}

    def load_state_dict(self, state):
        self.pos = state["pos"]
