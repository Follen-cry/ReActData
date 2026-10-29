class Analysis:
    def __init__(self, config, seed, total_steps, inner_probe_ids):
        self.ids = list(inner_probe_ids)
        self.last = {}

    def update(self, step, rows, step_info, probe_results):
        for p in probe_results:
            if p.get("kind") == "loss" and "items" in p:
                a = [x["loss"] for x in p["items"] if not x["context_frames"]]
                b = [x["loss"] for x in p["items"] if x["context_frames"]]
                if a and b:
                    self.last["ctx_gap"] = sum(a) / len(a) - sum(b) / len(b)
            if p.get("kind") == "generate" and "items" in p:
                self.last["probe_score"] = sum(x["score"] for x in p["items"]) / max(1, len(p["items"]))
        return dict(self.last)

    def request_probes(self, step, budget):
        if step == 1:
            items = []
            for c in self.ids[:2]:
                items.append({"clip_id": c, "context_frames": [], "template_id": "eval_format"})
                items.append({"clip_id": c, "context_frames": [12], "template_id": "subgoal", "target_frame": 28})
            return [{"kind": "loss", "items": items}]
        if step == 2:
            return [{"kind": "generate", "clip_ids": self.ids[:4]}, {"kind": "loss", "items": []}]
        return []

    def state_dict(self):
        return {"last": self.last}

    def load_state_dict(self, state):
        self.last = state["last"]
