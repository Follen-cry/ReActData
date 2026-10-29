"""c001: task-mixture shift. Quota 56 scale / 56 move / 16 geo, clips drawn uniformly within task, mixed order.

Derived from the fixed baseline (c000): same random-within-task draw and seeded shuffle, only the quota changes.
"""
import random

ROWS_PER_STEP = 16
QUOTA = {"shape_color_then_scale": 56, "shape_color_then_move": 56, "2d_geometric_transformation": 16}


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        rng = random.Random(f"{seed}:subset")
        tasks = sorted({c["task"] for c in pool["clips"]})
        n = total_steps * ROWS_PER_STEP
        quota = dict(config.get("quota", QUOTA)) if isinstance(config, dict) else dict(QUOTA)
        if sum(quota.get(t, 0) for t in tasks) != n:
            quota = dict(QUOTA)
        order = []
        for t in tasks:
            ids = sorted(c["clip_id"] for c in pool["clips"] if c["task"] == t)
            order += rng.sample(ids, quota[t])
        rng.shuffle(order)
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
