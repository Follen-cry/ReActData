"""Windowed loss by (template, context) every 10 steps; loss probe every 20 steps on 8 held-out clips for the plain
and the context template (final target); generate probes (plain F0 -> final) at steps 49 and 99."""


class Analysis:
    def __init__(self, config, seed, total_steps, inner_probe_ids):
        self.ctx_frame = int(config.get("ctx_frame", 27))
        self.ids = list(inner_probe_ids)[:8]
        self.win = {}
        self.probe_log = []

    def update(self, step, rows, step_info, probe_results):
        for r in rows:
            w = self.win.setdefault(r["template_id"], [0, 0.0])
            w[0] += 1
            w[1] += r["loss"]
        for p in probe_results or []:
            if p.get("kind") == "loss" and "items" in p:
                by = {}
                for it in p["items"]:
                    by.setdefault(it["template_id"], []).append(it["loss"])
                self.probe_log.append({"step": p["step"], "loss": {k: sum(v) / len(v) for k, v in by.items()}})
            elif p.get("kind") == "generate" and "items" in p:
                sc = [it["score"] for it in p["items"]]
                n = max(1, len(sc))
                keys = sorted({k for it in p["items"] for k in (it.get("sub") or {})})
                entry = {"step": p["step"], "gen_mean": sum(sc) / n}
                for k in keys:
                    entry[k] = sum(float((it.get("sub") or {}).get(k) or 0.0) for it in p["items"]) / n
                self.probe_log.append(entry)
        out = {}
        if step % 10 == 9 or step == total_steps_guard(step):
            out = {"step": step, "loss_by_template": {k: v[1] / v[0] for k, v in self.win.items()},
                   "probes": self.probe_log}
            self.win = {}
        return out

    def request_probes(self, step, budget):
        req = []
        if step % 20 == 19 and budget.get("loss_probes_left", 0) > 0:
            items = []
            for c in self.ids:
                items.append({"clip_id": c, "context_frames": [], "template_id": "eval_format"})
                items.append({"clip_id": c, "context_frames": [self.ctx_frame], "template_id": "ctx_final"})
            req.append({"kind": "loss", "items": items})
        if step in (49, 99) and budget.get("generate_probes_left", 0) > 0:
            req.append({"kind": "generate", "clip_ids": self.ids})
        return req

    def state_dict(self):
        return {"win": self.win, "probe_log": self.probe_log}

    def load_state_dict(self, state):
        self.win, self.probe_log = state["win"], state["probe_log"]


def total_steps_guard(step):
    return -1
