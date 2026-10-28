"""Fixed baseline databuilder: F0 + original evaluation-format description -> final frame.

Each row is one clip drawn uniformly without replacement from the training pool (a new seeded permutation per
pass), no context frames, template "eval_format" (= exactly the dev/test text). No feedback, no schedule.
"""
import random


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.seed = seed
        self.epoch, self.pos = 0, 0
        self._perm = None

    def _order(self):
        if self._perm is None or self._perm[0] != self.epoch:
            ids = list(self.ids)
            random.Random(f"{self.seed}:perm:{self.epoch}").shuffle(ids)
            self._perm = (self.epoch, ids)
        return self._perm[1]

    def next_batch(self, step):
        out = []
        for _ in range(16):
            order = self._order()
            out.append({"clip_id": order[self.pos], "context_frames": [], "template_id": "eval_format"})
            self.pos += 1
            if self.pos >= len(order):
                self.epoch, self.pos = self.epoch + 1, 0
        return out

    def observe(self, step, summary):
        pass

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, state):
        self.epoch, self.pos = state["epoch"], state["pos"]
