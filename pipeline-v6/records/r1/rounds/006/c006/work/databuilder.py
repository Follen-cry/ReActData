"""c006: edit-size selection on the two analogy tasks only. Balanced quota 43/43/42. scale and move: uniform random
from the 50% of clips with the largest change_px; geo: uniform random from the whole pool (any change_px selection
on geo lowered shape_fidelity in c002-c005). One seeded mixed order.
"""
import random

ROWS_PER_STEP = 16
DEFAULT_QUOTA = {"shape_color_then_scale": 43, "shape_color_then_move": 43, "2d_geometric_transformation": 42}
DEFAULT_FRAC = {"shape_color_then_scale": 0.5, "shape_color_then_move": 0.5, "2d_geometric_transformation": 1.0}
SMALL_FIRST = ("2d_geometric_transformation",)


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        rng = random.Random(f"{seed}:subset")
        quota = dict(config.get("quota") or DEFAULT_QUOTA)
        n = total_steps * ROWS_PER_STEP
        order = []
        frac = config.get("top_frac") or DEFAULT_FRAC
        for t in sorted(quota):
            f = frac.get(t, 1.0) if isinstance(frac, dict) else frac
            sign = 1 if t in SMALL_FIRST else -1
            cl = sorted((c for c in pool["clips"] if c["task"] == t), key=lambda c: (sign * c["change_px"], c["clip_id"]))
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
