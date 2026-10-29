"""c001: shift the budget away from geometric transformation toward the two analogy tasks.

Quota is configurable (default scale 56 / move 56 / geo 16); clips inside a task are drawn uniformly without
replacement; all steps mix the tasks in one seeded shuffled order. Started from the c000 baseline.
"""
import random

ROWS_PER_STEP = 16
DEFAULT_QUOTA = {"shape_color_then_scale": 56, "shape_color_then_move": 56, "2d_geometric_transformation": 16}


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        rng = random.Random(f"{seed}:subset")
        quota = dict(config.get("quota") or DEFAULT_QUOTA)
        n = total_steps * ROWS_PER_STEP
        order = []
        frac = config.get("top_frac", 1.0)
        for t in sorted(quota):
            f = frac.get(t, 1.0) if isinstance(frac, dict) else frac
            cl = sorted((c for c in pool["clips"] if c["task"] == t), key=lambda c: (-c["change_px"], c["clip_id"]))
            ids = sorted(c["clip_id"] for c in cl[:max(quota[t], int(round(len(cl) * f)))])
            order += rng.sample(ids, min(quota[t], len(ids)))
        rng.shuffle(order)
        self.order = order[:n]
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
