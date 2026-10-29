"""v6 test analysis: per-task mean loss each step; a loss probe on all inner-probe clips every step, one generate probe at step 4."""


class Analysis:
    def __init__(self, config, seed, total_steps, inner_probe_ids):
        self.ids = list(inner_probe_ids)
        self.last = {}

    def update(self, step, rows, step_info, probe_results):
        by = {}
        for r in rows:
            by.setdefault(r["clip_id"].split("_", 1)[0], []).append(r["loss"])
        probe = {}
        for p in probe_results:
            if p.get("kind") == "loss":
                for it in p["items"]:
                    probe.setdefault(it["clip_id"].split("_", 1)[0], []).append(it["loss"])
            elif p.get("kind") == "generate":
                probe["generate_mean"] = [sum(x["score"] for x in p["items"]) / max(1, len(p["items"]))]
        self.last = {"train_loss": {k: sum(v) / len(v) for k, v in by.items()},
                     "probe": {k: sum(v) / len(v) for k, v in probe.items()}}
        return self.last

    def request_probes(self, step, budget):
        out = []
        if step >= budget["loss_probe_next_allowed_step"] and budget["loss_probes_left"] > 0:
            out.append({"kind": "loss", "items": [{"clip_id": c, "context_frames": [], "template_id": "eval_format"} for c in self.ids]})
        if step == 4 and budget["generate_probes_left"] > 0:
            out.append({"kind": "generate", "clip_ids": self.ids[::4]})
        return out

    def state_dict(self):
        return {"last": self.last}

    def load_state_dict(self, state):
        self.last = state["last"]
