"""Logs mean loss per target frame (final vs sub-goal) per step and the data mix; one generate probe on the last step."""


class Analysis:
    def __init__(self, config, seed, total_steps, inner_probe_ids):
        self.total = total_steps
        self.probe_ids = list(inner_probe_ids)[:16]
        self.stats = {}

    def update(self, step, rows, step_info, probe_results):
        by = {}
        for r in rows:
            by.setdefault(r["target_frame"], []).append(r["loss"])
        for t, v in by.items():
            s = self.stats.setdefault(str(t), [0, 0.0])
            s[0] += len(v)
            s[1] += sum(v)
        out = {"step_loss_by_target": {str(t): sum(v) / len(v) for t, v in by.items()},
               "n_rows_by_target": {str(t): len(v) for t, v in by.items()},
               "running_mean_by_target": {t: s[1] / s[0] for t, s in self.stats.items()},
               "grad_norm": step_info.get("grad_norm")}
        if probe_results:
            out["probe_results"] = probe_results
        return out

    def request_probes(self, step, budget):
        if step == self.total - 1 and budget.get("generate_probes_left", 0) > 0 and self.probe_ids:
            return [{"kind": "generate", "clip_ids": self.probe_ids}]
        return []

    def state_dict(self):
        return {"stats": self.stats}

    def load_state_dict(self, state):
        self.stats = state["stats"]
