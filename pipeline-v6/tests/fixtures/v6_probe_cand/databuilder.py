"""Fixed baseline databuilder (reactdata-v6): a task-balanced random subset, one epoch.

The run's 128 rows are 43 / 43 / 42 clips of the three tasks (which task gets 42 rotates with the seed), each clip drawn
uniformly without replacement within its task, then shuffled into one seeded order (so every step mixes tasks at
random). Each clip is used once; no context frames; template "eval_format" (= exactly the dev/test text); target = final.
No feedback, no schedule.
"""
import random

ROWS_PER_STEP = 16


class Builder:
    def __init__(self, config, pool, seed, total_steps):
        rng = random.Random(f"{seed}:subset")
        tasks = sorted({c["task"] for c in pool["clips"]})
        n = total_steps * ROWS_PER_STEP
        quota = {t: n // len(tasks) for t in tasks}
        for i in range(n - sum(quota.values())):
            quota[tasks[(seed + i) % len(tasks)]] += 1
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
