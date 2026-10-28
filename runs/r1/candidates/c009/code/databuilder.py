"""c009: c006 process rows (8 plain final rows + 8 rows per step targeting an F-rescaling frame 36..48, template
'late', no context) until step proc_until, then 16 plain final rows so the run ends on the pure test-time format.
"""
import random


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.ids = [c["clip_id"] for c in pool["clips"]]
        self.seed = seed
        self.n_proc = int(config.get("n_proc", 8))
        self.proc_until = int(config.get("proc_until", 75))
        self.lo = int(config.get("late_lo", 36))
        self.hi = int(config.get("late_hi", 48))
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
        n = self.n_proc if step < self.proc_until else 0
        rng = random.Random(f"{self.seed}:proc:{step}")
        rows = [{"clip_id": c, "context_frames": [], "template_id": "eval_format"} for c in self._take("f", 16 - n)]
        rows += [{"clip_id": c, "context_frames": [], "template_id": "late", "target_frame": rng.randint(self.lo, self.hi)}
                 for c in self._take("s", n)]
        random.Random(f"{self.seed}:mix:{step}").shuffle(rows)
        return rows

    def observe(self, step, summary):
        pass

    def state_dict(self):
        return {"pos": dict(self.pos)}

    def load_state_dict(self, state):
        self.pos = dict(state["pos"])
