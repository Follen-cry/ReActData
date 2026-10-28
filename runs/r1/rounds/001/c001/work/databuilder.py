"""c001: baseline + sub-goal rows (E recoloured, F not yet present) for steps < sub_until, then pure final rows.

Per step in the sub-goal phase: 8 rows target the final frame (eval_format), 8 rows target an early frame
(default 26: E has its new colour, F not yet) with the 'subgoal' template. After sub_until all 16 rows are final.
"""
import random


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.seed = seed
        self.sub_until = int(config.get("sub_until", 70))
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
