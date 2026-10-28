"""Candidate checks (Runner-only, CPU): files, config keys, templates, static scan, 200-step sandboxed simulation,
determinism, save/load round trip, class tests (diagnostic labels, §6.5).

    python runner/checks.py <candidate_dir> [--out checks.json]
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import API, DATA, MANIFESTS, read_jsonl, write_json  # noqa: E402
sys.path.insert(0, API)
import plugin_api as PA  # noqa: E402
from recipe import RecipeError, check_batch, check_recipe  # noqa: E402
from text_slots import TemplateError, check_templates  # noqa: E402

from runner.plugin_host import PluginError, PluginHost  # noqa: E402
from runner.materialize import Materializer  # noqa: E402

REQUIRED = ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml", "proposal.md")
OPTIONAL = ("memory.md",)
MAX_BYTES = 100_000
CANDIDATE_KEYS = {"declared_class", "builder", "analysis"}
CLASSES = ("fixed", "static", "schedule", "feedback")
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint",
                "help", "memoryview", "exit", "quit"}
SIM_STEPS = 200


def static_scan(path: str) -> List[str]:
    from experiment.improvement.checks import static_scan as scan
    return scan(path, PA.ALLOWED_IMPORTS)

def synth_rows(step: int, recipes: List[Dict], key: str) -> List[Dict]:
    rng = random.Random(f"{key}:{step}")
    return [{"j": j, **r, "loss": 0.02 + 0.04 * rng.random(), "sigma": rng.random(),
             "seq_len": 420 + 280 * len(r["context_frames"]), "vit_images": 1 + len(r["context_frames"]), "cfg_drop": "none"}
            for j, r in enumerate(recipes)]


def synth_probe(q: Dict, step: int, key: str) -> Dict:
    rng = random.Random(f"{key}:probe:{step}:{q['kind']}")
    if q["kind"] == "loss":
        return {"kind": "loss", "step": step, "items": [{**it, "loss": 0.02 + 0.04 * rng.random(), "sigma": rng.random()} for it in q["items"]]}
    return {"kind": "generate", "step": step, "items": [{"clip_id": c, "score": rng.random(), "valid": True,
            "sub": {"rotation_angle": rng.random(), "position_alignment": rng.random(), "shape_fidelity": rng.random()}}
            for c in q["clip_ids"]]}


class Sim:
    """Drives a candidate for SIM_STEPS steps with synthetic signals, applying the real recipe and probe limits."""

    def __init__(self, cand: str, scratch: str, log: str, seed: int, signal_key: str, templates: set, cy: Dict, *, paths=None, host_factory=None, prompt_wrapper=None):
        manifests = paths.manifests if paths else MANIFESTS
        data = paths.data if paths else DATA
        self.mat = Materializer(cand, paths=paths, prompt_wrapper=prompt_wrapper)          # every recipe's text is rendered exactly as in training (reactdata-v0 gap)
        self.pool_ids = {r["clip_id"] for r in read_jsonl(os.path.join(manifests, "train_pool.jsonl"))}
        self.probe_ids = sorted(r["clip_id"] for r in read_jsonl(os.path.join(manifests, "inner_probe.jsonl")))
        self.templates, self.key = templates, signal_key
        self.host = (host_factory or PluginHost)(cand, os.path.join(data, "pool_view.json"), scratch, log,
                                                **({"api": paths.api} if paths else {}))
        tl = sorted(templates)
        self.host.call("init", timeout=120, builder_config={**(cy.get("builder") or {}), "templates": tl},
                       analysis_config={**(cy.get("analysis") or {}), "templates": tl}, seed=seed,
                       total_steps=SIM_STEPS, inner_probe_ids=self.probe_ids)
        self.sched = {"last_loss": -10 ** 9, "last_gen": -10 ** 9, "n_loss": 0, "n_gen": 0}
        self.pending: List[Dict] = []
        self.trace: List[List[Dict]] = []
        self.n_probe = Counter()

    def step(self, k: int) -> None:
        recipes = check_batch(self.host.call("next_batch", step=k), self.pool_ids, self.templates)
        for r in recipes:
            self.mat.render(r)
        self.trace.append(recipes)
        budget = {"loss_probe_next_allowed_step": self.sched["last_loss"] + PA.LOSS_PROBE_MIN_GAP,
                  "generate_probe_next_allowed_step": self.sched["last_gen"] + PA.GENERATE_PROBE_MIN_GAP,
                  "loss_probes_left": PA.MAX_LOSS_PROBES - self.sched["n_loss"],
                  "generate_probes_left": PA.MAX_GENERATE_PROBES - self.sched["n_gen"]}
        info = {"step": k, "lr": 1e-5, "grad_norm": 0.05, "gen_weight": 0.5, "sec_per_step": 5.0, "total_steps": SIM_STEPS}
        res = self.host.call("after_step", step=k, rows=synth_rows(k, recipes, self.key), step_info=info,
                             probe_results=self.pending, budget=budget)
        self.pending = []
        for q in (res["requests"] or [])[:4]:
            kind = q.get("kind") if isinstance(q, dict) else None
            if kind == "loss" and k - self.sched["last_loss"] >= PA.LOSS_PROBE_MIN_GAP and self.sched["n_loss"] < PA.MAX_LOSS_PROBES:
                items = q.get("items")
                if not isinstance(items, list) or not 1 <= len(items) <= PA.PROBE_MAX_ITEMS:
                    raise RecipeError("loss probe: 1..32 items")
                q = {"kind": "loss", "items": [check_recipe(it, set(self.probe_ids), self.templates) for it in items]}
                for it in q["items"]:
                    self.mat.render(it)
                self.sched.update(last_loss=k, n_loss=self.sched["n_loss"] + 1)
            elif kind == "generate" and k - self.sched["last_gen"] >= PA.GENERATE_PROBE_MIN_GAP and self.sched["n_gen"] < PA.MAX_GENERATE_PROBES:
                ids = q.get("clip_ids")
                if not isinstance(ids, list) or not ids or any(i not in self.probe_ids for i in ids):
                    raise RecipeError("generate probe: clip_ids must be inner-probe ids")
                q = {"kind": "generate", "clip_ids": ids[:PA.PROBE_MAX_ITEMS]}
                self.sched.update(last_gen=k, n_gen=self.sched["n_gen"] + 1)
            else:
                self.pending.append({"kind": kind, "status": "rejected", "reason": "budget or kind", "step": k})
                continue
            self.n_probe[q["kind"]] += 1
            self.pending.append(synth_probe(q, k, self.key))

    def state(self):
        return {"plugin": self.host.call("state"), "pending": self.pending, "sched": dict(self.sched)}

    def load(self, st):
        self.host.call("load", state=st["plugin"])
        self.pending, self.sched = st["pending"], dict(st["sched"])

    def close(self):
        self.host.close()


def trace_hash(trace) -> str:
    return hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()


def features(batches) -> Dict[str, Counter]:
    f = {"n_context": Counter(), "context_bin": Counter(), "template": Counter(), "target_bin": Counter()}
    for b in batches:
        for r in b:
            f["n_context"][len(r["context_frames"])] += 1
            f["target_bin"][r.get("target_frame", 59) // 10] += 1
            for t in r["context_frames"]:
                f["context_bin"][t // 10] += 1
            f["template"][r["template_id"]] += 1
    return f


def tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def run_checks(cand: str, work: str, *, paths=None, host_factory=None, prompt_wrapper=None) -> Dict:
    sim_options = dict(paths=paths, host_factory=host_factory, prompt_wrapper=prompt_wrapper)
    out = {"candidate": os.path.abspath(cand), "errors": [], "warnings": []}
    E = out["errors"]
    from experiment.improvement.checks import file_errors
    E.extend(file_errors(cand, REQUIRED, OPTIONAL, MAX_BYTES))
    if E:
        return out
    try:
        cy = yaml.safe_load(open(os.path.join(cand, "candidate.yaml"))) or {}
        if not isinstance(cy, dict) or set(cy) - CANDIDATE_KEYS:
            E.append(f"candidate.yaml: allowed keys {sorted(CANDIDATE_KEYS)}")
        elif cy.get("declared_class") not in CLASSES:
            E.append(f"candidate.yaml: declared_class must be one of {CLASSES}")
        elif not isinstance(cy.get("builder", {}) or {}, dict) or not isinstance(cy.get("analysis", {}) or {}, dict):
            E.append("candidate.yaml: builder / analysis must be mappings")
        elif len(json.dumps(cy)) > 20000:
            E.append("candidate.yaml too large")
    except yaml.YAMLError as e:
        E.append(f"candidate.yaml: {e}")
        cy = {}
    try:
        templates = set(check_templates(yaml.safe_load(open(os.path.join(cand, "text_templates.yaml")))))
    except (TemplateError, yaml.YAMLError) as e:
        E.append(f"text_templates.yaml: {e}")
        templates = set()
    for f in ("databuilder.py", "analysis.py"):
        E.extend(static_scan(os.path.join(cand, f)))
    if E:
        return out
    out["declared_class"] = cy.get("declared_class")
    # simulations: A (reference), B (same seed: determinism), R (resume at 100), P (perturbed signals)
    import shutil
    for _ in range(10):                    # NFS: a just-closed plugin worker can leave .nfs* files for a moment
        shutil.rmtree(work, ignore_errors=True)
        if not os.path.exists(work) or not os.listdir(work):
            break
        time.sleep(3)
    os.makedirs(work, exist_ok=True)
    if os.listdir(work):
        raise RuntimeError(f"cannot clear check work dir {work}")
    log = os.path.join(work, "plugin.log")
    sims = {}
    try:
        for name, key in (("A", "sig0"), ("B", "sig0"), ("P", "sig1")):
            s = Sim(cand, os.path.join(work, f"scr_{name}"), log, 1000, key, templates, cy, **sim_options)
            for k in range(SIM_STEPS):
                s.step(k)
            sims[name] = (s.trace, dict(s.n_probe))
            s.close()
        s = Sim(cand, os.path.join(work, "scr_R1"), log, 1000, "sig0", templates, cy, **sim_options)
        for k in range(100):
            s.step(k)
        st = json.loads(json.dumps(s.state()))
        first = s.trace
        s.close()
        s = Sim(cand, os.path.join(work, "scr_R2"), log, 1000, "sig0", templates, cy, **sim_options)
        s.load(st)
        for k in range(100, SIM_STEPS):
            s.step(k)
        sims["R"] = (first + s.trace, dict(s.n_probe))
        s.close()
    except (PluginError, RecipeError, TemplateError, KeyError, TypeError) as e:
        E.append(f"simulation: {type(e).__name__}: {e}"[:3000])
        out["plugin_log_tail"] = open(log).read()[-3000:] if os.path.exists(log) else ""
        return out
    A, B, P, R = (sims[k][0] for k in "ABPR")
    if trace_hash(A) != trace_hash(B):
        E.append("not deterministic: two runs with the same seed and signals gave different recipes")
    if trace_hash(A) != trace_hash(R):
        E.append("save/load round trip at step 100 changed the recipes of steps 100..199")
    fa = features(A[:50]), features(A[150:])
    step_tv = {k: round(tv(fa[0][k], fa[1][k]), 3) for k in fa[0]}
    sens = sum(json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True) for a, b in zip(A, P))
    feedback = sens > 0
    schedule = max(step_tv.values()) > 0.2
    tested = "feedback" if feedback else ("schedule" if schedule else "static")
    declared = cy.get("declared_class")
    order = {"fixed": 0, "static": 1, "schedule": 2, "feedback": 3}
    label = max(declared, tested, key=lambda c: order[c]) if declared != "fixed" else ("fixed" if tested == "static" else tested)
    allf = features(A)
    out.update({
        "tested_behaviour": {"steps_with_different_recipes_under_perturbed_signals": sens,
                             "early_vs_late_tv": step_tv, "tested_class": tested},
        "class_label": label, "class_consistent": order.get(declared, 0) >= order[tested] if declared != "fixed" else tested == "static",
        "class_note": "diagnostic only: limited perturbation tests cannot prove program properties",
        "sim_probe_requests": sims["A"][1],
        "sim_stats": {"rows": sum(len(b) for b in A), "unique_clips": len({r["clip_id"] for b in A for r in b}),
                      "vit_images": sum(1 + len(r["context_frames"]) for b in A for r in b),
                      "n_context": dict(allf["n_context"]), "templates": dict(allf["template"]),
                      "context_frame_bins_of_10": {str(k): v for k, v in sorted(allf["context_bin"].items())},
                      "target_frame_bins_of_10": {str(k): v for k, v in sorted(allf["target_bin"].items())}},
    })
    if not out["class_consistent"]:
        out["warnings"].append(f"declared class {declared} but behaved like {tested} under the tests; label {label}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run_checks(a.candidate, a.work or os.path.join(a.candidate + "_checkwork"))
    if a.out:
        write_json(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "plugin_log_tail"}, indent=1))
    sys.exit(1 if res["errors"] else 0)
