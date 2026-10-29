"""c001: sub-goal curriculum. Every step: 10 final-frame rows (eval text, no context) + 6 rows whose target is the
E-done / F-not-yet frame (SUBGOAL_T) of a clip, for the first 70% of the steps; the last 30% are final-only
(the model ends on the test-time mapping). Clips are drawn from a seeded permutation per pass; sub-goal clips are
offset by half a pass from the final clips so a clip is seen with both targets."""
import random

SUBGOAL_T = 26
N_SUB = 6


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.seed = seed
        self.total = total_steps
        self.sub_frac = config.get("sub_frac", 0.7)
        self.epoch, self.pos = 0, 0
        self._perm = None

    def _order(self):
        if self._perm is None or self._perm[0] != self.epoch:
            ids = list(self.ids)
            random.Random(f"{self.seed}:perm:{self.epoch}").shuffle(ids)
            self._perm = (self.epoch, ids)
        return self._perm[1]

    def _next_id(self):
        order = self._order()
        cid = order[self.pos]
        self.pos += 1
        if self.pos >= len(order):
            self.epoch, self.pos = self.epoch + 1, 0
        return cid

    def next_batch(self, step):
        n_sub = 8 if step < self.sub_frac * self.total else 0
        out = []
        for j in range(16):
            cid = self._next_id()
            if j < n_sub:
                # chain of sub-goals: E done (t=26) and F half-way rescaled (t=42)
                t = SUBGOAL_T if j % 2 == 0 else 42
                out.append({"clip_id": cid, "context_frames": [], "template_id": "subgoal", "target_frame": t})
            else:
                out.append({"clip_id": cid, "context_frames": [], "template_id": "eval_format"})
        return out

    def observe(self, step, summary):
        pass

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, state):
        self.epoch, self.pos = state["epoch"], state["pos"]
