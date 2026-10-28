"""c004: near-final target augmentation. Every step: 10 plain final rows (eval_format) + 6 rows whose target is a
late frame of the settled final state (frames 50..57, F already at its final size), template 'late'. Which clip/frame
is drawn from seeded streams; no context frames; same schedule for all 100 steps.
"""
import random


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.seed = seed
        self.n_late = int(config.get("n_late", 6))
        self.lo = int(config.get("late_lo", 50))
        self.hi = int(config.get("late_hi", 57))
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
        rng = random.Random(f"{self.seed}:late:{step}")
        rows = [{"clip_id": c, "context_frames": [], "template_id": "eval_format"} for c in self._take("f", 16 - self.n_late)]
        rows += [{"clip_id": c, "context_frames": [], "template_id": "late", "target_frame": rng.randint(self.lo, self.hi)}
                 for c in self._take("s", self.n_late)]
        random.Random(f"{self.seed}:mix:{step}").shuffle(rows)
        return rows

    def observe(self, step, summary):
        pass

    def state_dict(self):
        return {"pos": dict(self.pos)}

    def load_state_dict(self, state):
        self.pos = dict(state["pos"])
