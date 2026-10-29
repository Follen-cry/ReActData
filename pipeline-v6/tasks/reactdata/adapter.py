"""Reactdata capabilities: jobs, measurements, audit, pilot and presentation.

Experiment stages and statistical design live in experiment/, not in this adapter.
"""
import os
import time
from functools import partial
from dataclasses import asdict
from experiment.runtime.io import read_json, read_jsonl, write_json, sha256_file, sha256_tree
from experiment.workflow.statistics import Statistics
from experiment.evidence.access import EvaluationAccessPolicy
from experiment.runtime.artifacts import freeze_candidate
from tasks.reactdata.contract import ReactDataContract, CAND_FILES
from tasks.reactdata.prompts import PromptBuilder
from tasks.reactdata.evidence import ReactDataEvidence

BACKUP_EXT = (".json", ".md", ".diff", ".yaml", ".py", ".txt")

class ReactDataTask:
    def __init__(self, protocol, paths, smoke=False, hooks=None):
        self.protocol, self.paths, self.smoke, self.hooks = protocol, paths, smoke, hooks or {}
        self.baseline_dir = os.path.join(paths.exp, "seeds", "fixed_f0_final")
        self.statistics = Statistics(lambda item_id: item_id.rsplit("_", 1)[0])
        self.prompts = PromptBuilder()
        self.access = EvaluationAccessPolicy()

    def bind(self, store, protocol_sha):
        from runner.checks import run_checks
        from runner.report import write_report
        self.contract = ReactDataContract(store.root, self.protocol, protocol_sha, self.hooks.get("run_checks", partial(run_checks, paths=self.paths)),
            store=store, exp=self.paths.exp, data=self.paths.data, manifests=self.paths.manifests)
        self.evidence_source = ReactDataEvidence(self.hooks.get("write_report", write_report), os.path.join(self.paths.manifests, "dev.jsonl"), paths=self.paths)

    def evals(self, store, split, seeds):
        if split == "test":
            self.access.require_test(store.snapshot().data, self.contract.verify_frozen)
        if split == "test" and self.smoke:
            return [{"name": "test", "file": os.path.join(self.paths.manifests, "dev.jsonl"), "seeds": list(seeds), "smoke_standin": True}]
        return [{"name": split, "file": os.path.join(self.paths.manifests, split + ".jsonl"), "seeds": list(seeds)}]

    def authorize_job(self, store, spec):
        if any(ev.get("name") == "test" for ev in spec.get("evals", [])):
            self.access.require_test(store.snapshot().data, self.contract.verify_frozen)

    def build_spec(self, store, request, code, settings):
        if request.candidate_id is None:
            name = "base_dev" if request.split == "dev" else "base_test"
        else:
            name = ("formal_" if request.formal else "") + f"{request.candidate_id}-s{request.seed}-g4"
        directory = os.path.join(store.root, "jobs", name)
        return store.job_spec(name, {"run_id": name, "run_dir": directory, "candidate_dir": code,
            "run_seed": request.seed, "steps": request.steps, "evals": self.evals(store, request.split, request.generation_seeds),
            "max_eval_rows": settings.max_rows, "protocol_sha256": sha256_file(self.paths.protocol), "gpus": 4, "resources": asdict(self.paths),
            **self.schedule_fields()})

    def schedule_fields(self):
        """training.overrides (config keys, e.g. optim.lr) and training.warmup_steps of the protocol (v6)."""
        tr = self.protocol["training"]
        ov = [f"checkpoint.every_steps={tr['checkpoint_every']}"] if tr.get("checkpoint_every") else []
        out = {"overrides": ov + list(tr.get("overrides", []))}
        if tr.get("warmup_steps") is not None:
            out["warmup_steps"] = int(tr["warmup_steps"])
        return out

    item_means = staticmethod(Statistics.item_means)

    @staticmethod
    def cost(spec):
        path = os.path.join(read_json(spec)["run_dir"], "cost.json")
        return read_json(path) if os.path.exists(path) else {}

    def cost_summary(self, spec):
        cost = self.cost(spec)
        return {k: cost.get(k) for k in ("rows", "unique_clips", "repeated_rows", "max_clip_uses", "vit_images", "vit_tokens", "seq_tokens",
                                        "train_gpu_s", "probe_gpu_s", "eval_gpu_s", "plugin_wall_s")}

    def audit(self, directory):
        from runner.audit import audit_run
        return self.hooks.get("audit_run", partial(audit_run, paths=self.paths))(directory)

    @staticmethod
    def audit_summary(audit):
        return {k: audit.get(k) for k in ("mi_bits", "p_perm", "flag", "templates")}

    @staticmethod
    def estimate_job_hours(cost, settings, design):
        scale = len(settings.S) / len(settings.S + settings.S_null) if design != "v3" else 1
        return (cost["load_wall_s"] + cost["train_wall_s"] + sum(v["gen_wall_s"] for v in cost["evals"].values()) * scale
                + (120 if design == "v3" else 180)) / 3600

    def freeze(self, store, selected):
        return freeze_candidate(store, self.contract, selected)

    def provenance(self):
        scorer_version = self.hooks.get("scorer_version")
        if scorer_version is None:
            from runner.score import scorer_version
        return {"api_sha256": sha256_tree([os.path.join(self.paths.api, f) for f in os.listdir(self.paths.api) if f.endswith((".py", ".md"))]),
                "builder_agent_md_sha256": sha256_file(os.path.join(self.paths.exp, self.protocol["search"]["agent"].get("instructions", "BUILDER_AGENT.md"))),
                "data_build": read_json(os.path.join(self.paths.data, "BUILD_DONE.json")), "scorer": scorer_version()}

    def report(self, view):
        from runner.final_report import build, build_v3
        (build_v3 if view.design == "v3" else build)(view, paths=self.paths)

    def pilot(self, store, execute, settings) -> None:
        store.commit("phase_start", updates={"phase": "P0", "budget_start": store.state.get("budget_start", time.time())})
        spec = os.path.join(store.root, "jobs", "p0_pilot", "run.json")
        if not os.path.exists(spec):
            d = os.path.dirname(spec)
            os.makedirs(d, exist_ok=True)
            write_json(spec, {"run_id": "p0_pilot", "run_dir": d, "candidate_dir": self.baseline_dir,
                              "resources": asdict(self.paths), "run_seed": 1, "steps": 4, "evals": self.evals(store, "dev", [0]), "max_eval_rows": 8,
                              **self.schedule_fields()})
        st = execute(spec)
        if st != "done":
            raise SystemExit("P0 pilot failed: blocking (see failures in state.json)")
        cost = self.cost(spec)
        fs = read_json(os.path.join(os.path.dirname(spec), "first_step.json"))
        ev = list(cost["evals"].values())[0]
        sec_step = cost["train_wall_s"] / 4
        sec_row = ev["gen_wall_s"] / 8 * 4               # per row per GPU (8 rows on 4 GPUs)
        n_dev = 4 if self.smoke else len(read_jsonl(os.path.join(self.paths.manifests, "dev.jsonl")))
        est = {"sec_per_step": sec_step, "gen_sec_per_row_gpu": sec_row,
               "candidate_job_h": (cost["load_wall_s"] + sec_step * settings.steps + 4 * 15 + len(settings.S) * sec_row * n_dev / 4 + 120) / 3600}
        store.commit("pilot", updates={"estimates": est, "first_step_check": fs}, estimates=est, first_step=fs)
        if not fs.get("ok"):
            raise SystemExit("first-step acceptance failed: blocking")
