"""Logs per-task training loss each step; requests a loss probe on all inner-probe clips after every step and one generate probe."""


def _task(cid):
    return cid.split("_")[0]


class Analysis:
    def __init__(self, config, seed, total_steps, inner_probe_ids):
        self.ids = list(inner_probe_ids)
        self.total = total_steps
        self.n, self.mean = 0, 0.0

    def update(self, step, rows, step_info, probe_results):
        acc = {}
        for r in rows:
            self.n += 1
            self.mean += (r["loss"] - self.mean) / self.n
            a = acc.setdefault(_task(r["clip_id"]), [0, 0.0])
            a[0] += 1
            a[1] += r["loss"]
        out = {"mean_loss": self.mean, "grad_norm": step_info.get("grad_norm")}
        for t, (c, s) in acc.items():
            out["rows_" + t] = c
            out["loss_" + t] = s / c
        for p in probe_results or []:
            if p.get("kind") == "loss" and "items" in p:
                pa = {}
                for it in p["items"]:
                    b = pa.setdefault(_task(it["clip_id"]), [0, 0.0])
                    b[0] += 1
                    b[1] += it["loss"]
                out["probe_loss_after_step_%d" % p.get("step", -1)] = {t: v[1] / v[0] for t, v in pa.items()}
            elif p.get("kind") == "generate" and "items" in p:
                ga = {}
                for it in p["items"]:
                    b = ga.setdefault(_task(it["clip_id"]), [0, 0.0])
                    b[0] += 1
                    b[1] += it["score"]
                out["probe_score_step_%d" % p.get("step", -1)] = {t: v[1] / v[0] for t, v in ga.items()}
        return out

    def request_probes(self, step, budget):
        req = []
        items = [{"clip_id": c, "context_frames": [], "template_id": "eval_format"} for c in self.ids]
        if budget.get("loss_probes_left", 0) > 0 and step >= budget.get("loss_probe_next_allowed_step", 0):
            req.append({"kind": "loss", "items": items})
        if step >= 4 and budget.get("generate_probes_left", 0) > 0 and step >= budget.get("generate_probe_next_allowed_step", 4):
            req.append({"kind": "generate", "clip_ids": list(self.ids)})
        return req

    def state_dict(self):
        return {"n": self.n, "mean": self.mean}

    def load_state_dict(self, state):
        self.n, self.mean = state["n"], state["mean"]
