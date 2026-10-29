"""Task-specific probe scheduling, independent of torch and Trainer lifecycle."""
import plugin_api as PA
from recipe import RecipeError, check_recipe
from text_slots import TemplateError

class ProbeController:
    def __init__(self, probe_ids, template_ids, render):
        self.probe_ids, self.template_ids, self.render = probe_ids, template_ids, render
        self.pending = []
        self.sched = {"last_loss": -10 ** 9, "last_gen": -10 ** 9, "n_loss": 0, "n_gen": 0}

    def budget(self):
        return {"loss_probe_next_allowed_step": self.sched["last_loss"] + PA.LOSS_PROBE_MIN_GAP,
                "generate_probe_next_allowed_step": self.sched["last_gen"] + PA.GENERATE_PROBE_MIN_GAP,
                "loss_probes_left": PA.MAX_LOSS_PROBES - self.sched["n_loss"],
                "generate_probes_left": PA.MAX_GENERATE_PROBES - self.sched["n_gen"]}

    def accept(self, requests, step):
        for request in requests:
            key = "loss" if request["kind"] == "loss" else "gen"
            self.sched[f"last_{key}"] = step
            self.sched[f"n_{key}"] += 1

    def state_dict(self):
        return {"pending": self.pending, "sched": self.sched}

    def load_state_dict(self, state):
        self.pending, self.sched = state["pending"], state["sched"]

    def check(self, reqs, step):
        """rank 0: keep the legal requests, report the others back to the plugin."""
        ok, rejected = [], []
        seen = set()
        for q in reqs[:4]:
            kind = q.get("kind") if isinstance(q, dict) else None
            why = None
            if kind not in ("loss", "generate"):
                why = "unknown kind"
            elif kind in seen:
                why = "one request per kind per step"
            elif kind == "loss":
                if step - self.sched["last_loss"] < PA.LOSS_PROBE_MIN_GAP or self.sched["n_loss"] >= PA.MAX_LOSS_PROBES:
                    why = "loss probe budget (gap / count)"
                else:
                    try:
                        items = q["items"]
                        if not isinstance(items, list) or not 1 <= len(items) <= PA.PROBE_MAX_ITEMS:
                            raise RecipeError(f"1..{PA.PROBE_MAX_ITEMS} items")
                        items = [check_recipe(it, set(self.probe_ids), self.template_ids) for it in items]
                        for it in items:
                            self.render(it)                  # template / context compatibility
                        q = {"kind": "loss", "items": items}
                    except (RecipeError, TemplateError, KeyError, TypeError) as e:
                        why = f"bad items: {e}"[:300]
            else:
                if step - self.sched["last_gen"] < PA.GENERATE_PROBE_MIN_GAP or self.sched["n_gen"] >= PA.MAX_GENERATE_PROBES:
                    why = "generate probe budget (gap / count)"
                else:
                    ids = q.get("clip_ids")
                    if (not isinstance(ids, list) or not 1 <= len(ids) <= PA.PROBE_MAX_ITEMS or len(set(ids)) != len(ids)
                            or any(i not in self.probe_ids for i in ids)):
                        why = "clip_ids must be distinct inner-probe ids (1..32)"
                    else:
                        q = {"kind": "generate", "clip_ids": list(ids)}
            if why:
                rejected.append({"kind": kind, "status": "rejected", "reason": why, "step": step})
            else:
                seen.add(kind)
                ok.append(q)
        return ok, rejected
