"""Fixed baseline analysis: keeps a running mean of the training loss for the logs; requests no probes."""


class Analysis:
    def __init__(self, config, seed, total_steps, inner_probe_ids):
        self.n, self.mean = 0, 0.0

    def update(self, step, rows, step_info, probe_results):
        for r in rows:
            self.n += 1
            self.mean += (r["loss"] - self.mean) / self.n
        return {"mean_loss": self.mean}

    def request_probes(self, step, budget):
        return []

    def state_dict(self):
        return {"n": self.n, "mean": self.mean}

    def load_state_dict(self, state):
        self.n, self.mean = state["n"], state["mean"]
