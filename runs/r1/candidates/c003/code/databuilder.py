"""c003: short early sub-goal warm-up. Steps < sub_until: 8 final rows + 8 rows targeting the E-state frame
(E recoloured, F absent). Afterwards 16 plain final-frame baseline rows (60 steps of pure test-time format).
"""
import random


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.seed = seed
        self.sub_until = int(config.get("sub_until", 40))
        self.sub_frame = int(config.get("sub_frame", 26))
        self.n_sub = int(config.get("n_sub", 8))
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
        n_sub = self.n_sub if step < self.sub_until else 0
        rows = [{"clip_id": c, "context_frames": [], "template_id": "eval_format"} for c in self._take("f", 16 - n_sub)]
        rows += [{"clip_id": c, "context_frames": [], "template_id": "subgoal", "target_frame": self.sub_frame}
                 for c in self._take("s", n_sub)]
        random.Random(f"{self.seed}:mix:{step}").shuffle(rows)
        return rows

    def observe(self, step, summary):
        pass

    def state_dict(self):
        return {"pos": dict(self.pos)}

    def load_state_dict(self, state):
        self.pos = dict(state["pos"])
