"""Logs mean training loss per task per step, and requests loss probes on all inner-probe clips every other step
plus one generate probe after the last step."""


def _task(cid):
    return cid.split("_")[0]


class Analysis:
    def __init__(self, config, seed, total_steps, inner_probe_ids):
        self.total = total_steps
        self.probe_ids = sorted(inner_probe_ids)
        self.n, self.mean = 0, 0.0

    def update(self, step, rows, step_info, probe_results):
        by = {}
        for r in rows:
            by.setdefault(_task(r["clip_id"]), []).append(r["loss"])
            self.n += 1
            self.mean += (r["loss"] - self.mean) / self.n
        out = {"mean_loss": self.mean,
               "task_loss": {t: round(sum(v) / len(v), 5) for t, v in by.items()},
               "task_rows": {t: len(v) for t, v in by.items()},
               "grad_norm": step_info.get("grad_norm")}
        for p in probe_results or []:
            if p.get("kind") == "loss":
                agg = {}
                for it in p["items"]:
                    agg.setdefault(_task(it["clip_id"]), []).append(it["loss"])
                out["probe_loss_at_%d" % p.get("step", -1)] = {t: round(sum(v) / len(v), 5) for t, v in agg.items()}
            elif p.get("kind") == "generate":
                agg = {}
                for it in p["items"]:
                    agg.setdefault(it["task"], []).append(it["score"])
                out["probe_gen_score"] = {t: round(sum(v) / len(v), 4) for t, v in agg.items()}
        return out

    def request_probes(self, step, budget):
        reqs = []
        if step % 2 == 0 and budget.get("loss_probes_left", 0) > 0:
            reqs.append({"kind": "loss", "items": [{"clip_id": c, "context_frames": [], "template_id": "eval_format"}
                                                    for c in self.probe_ids]})
        if step == self.total - 1 and budget.get("generate_probes_left", 0) > 0:
            reqs.append({"kind": "generate", "clip_ids": self.probe_ids})
        return reqs

    def state_dict(self):
        return {"n": self.n, "mean": self.mean}

    def load_state_dict(self, state):
        self.n, self.mean = state["n"], state["mean"]
