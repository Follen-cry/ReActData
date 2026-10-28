"""A second task: learn a numeric offset, using text programs and opaque query IDs.

Only a test fixture; it uses the SAME engine, search, candidates, store, evidence,
agent service, resources and executor contracts as reactdata.
"""
from pathlib import Path
import shutil
import time
from experiment.runtime.io import read_json, write_json, sha256_tree
from experiment.evidence.access import EvaluationAccessPolicy
from experiment.improvement.agents import AgentService, FakeAgentBackend
from experiment.improvement.candidates import CandidateService
from experiment.workflow.engine import ExperimentEngine
from experiment.evidence import EvidenceService, VisibilityPolicy
from experiment.runtime.execution import FakeExecutor
from experiment.improvement.policy import DefaultSearchPolicy
from experiment.workflow.selection import NoiseAwareSelection
from experiment.runtime.resources import ResourceManager
from experiment.improvement.search import SearchController


class TextContract:
    files = ("program.txt", "proposal.md", "memory.md")

    def __init__(self, store):
        self.store = store

    def program_sha(self, directory):
        return sha256_tree([str(Path(directory) / "program.txt")])

    def check_candidate(self, work, scratch):
        int((Path(work) / "program.txt").read_text())
        return {"errors": [], "warnings": [], "class_label": "static", "declared_class": "static",
                "tested_behaviour": {}, "sim_stats": {}}

    def valid_dev_ids(self):
        return {"query:a-17", "urn:query/beta"}

    def proposal_errors(self, work):
        text = (Path(work) / "proposal.md").read_text()
        return [] if all(i in text for i in self.valid_dev_ids()) else ["cite both query IDs"]

    def instructions(self):
        return "Learn an integer offset for a numeric predictor."

    def pool_stats(self):
        return {"numbers": [1, 3, 5]}

    frozen_hash = program_sha

    def verify_frozen(self, record):
        return self.frozen_hash(record["dir"]) == record["code_sha256"]

    def candidate_record(self, cid, directory, **values):
        self.store.candidate_record(cid, {"cand_id": cid, "program_sha": self.program_sha(directory), **values})


class TextPrompts:
    def build(self, context, cid, rnd, idx, start, siblings, instructions):
        from experiment.contracts import PromptBundle
        return PromptBundle(["numeric evidence"], f"FULL {instructions} {cid}", f"CONTINUE {cid}")

    def repair(self, prompt, checks, resumed):
        return "REPAIR cite the numeric queries"


class TextEvidence:
    @staticmethod
    def search_options(snapshot, best):
        return {}

    def collect_evidence(self, request):
        from experiment.evidence.authorized import AuthorizedEvidence, OutputFile
        dest = Path(request["round_dir"]) / "input"
        measurements = []
        for entry in request["entries"]:
            rd = entry.get("run_dir")
            measurements.append({"id": entry["id"], "status": entry["status"],
                                 "metrics": read_json(str(Path(rd) / "measurements.json")) if rd and request["visibility"].allows("results") else None})
        return AuthorizedEvidence(str(dest), (OutputFile("results.json", "json", measurements),
            OutputFile("pool.json", "json", request["pool_stats"] if request["visibility"].allows("train_examples") else {})))

    def render_evidence(self, authorized):
        from experiment.evidence.authorized import render_authorized
        return render_authorized(authorized)

    def build_evidence(self, request):
        return self.render_evidence(self.collect_evidence(request))


class NumericAdapter:
    """No baseline/confirm/select/final handlers: all come from shared design operations."""
    def __init__(self):
        from experiment.workflow.statistics import Statistics
        self.statistics = Statistics(lambda item_id: "numeric")
        self.prompts = TextPrompts()
        self.evidence_source = TextEvidence()

    def bind(self, store, protocol_sha):
        self.contract = TextContract(store)
        self.baseline_dir = str(Path(store.root) / "seed")
        baseline = Path(self.baseline_dir)
        baseline.mkdir(exist_ok=True)
        for filename, value in (("program.txt", "0"), ("proposal.md", "baseline"), ("memory.md", "notes")):
            if not (baseline / filename).exists():
                (baseline / filename).write_text(value)

    def build_spec(self, store, request, code, settings):
        name = request.key if not request.formal else "formal_" + request.key
        return store.job_spec(name, {"run_id": name, "run_dir": str(Path(store.root) / "jobs" / name),
            "candidate_dir": code, "seed": request.seed, "steps": request.steps,
            "payload": {"split": request.split, "seeds": list(request.generation_seeds),
                        "queries": sorted(self.contract.valid_dev_ids())}})

    def measure(self, run):
        value = int((Path(run["candidate_dir"]) / "program.txt").read_text()) / 10 if run["candidate_dir"] else -0.2
        noise = (run["seed"] % 5) * 0.001
        payload = run["payload"]
        measurements = {str(seed): {i: 0.2 + value + noise for i in payload["queries"]} for seed in payload["seeds"]}
        write_json(str(Path(run["run_dir"]) / "measurements.json"), {payload["split"]: measurements})
        return "done"

    def item_means(self, directory, split, seeds):
        values = read_json(str(Path(directory) / "measurements.json"))[split]
        return {i: sum(values[str(seed)][i] for seed in seeds) / len(seeds) for i in values[str(seeds[0])]}

    def cost(self, spec):
        return {"load_wall_s": 1, "train_wall_s": 1, "evals": {"fake": {"gen_wall_s": 1}}}

    def cost_summary(self, spec):
        return {"train_gpu_s": 0}

    def estimate_job_hours(self, cost, settings, design):
        return 0.01

    def audit(self, directory):
        return {"flag": False}

    def audit_summary(self, audit):
        return audit

    def freeze(self, store, selected):
        from experiment.runtime.artifacts import freeze_candidate
        return freeze_candidate(store, self.contract, selected)

    def authorize_job(self, store, spec):
        if spec["payload"]["split"] == "test":
            EvaluationAccessPolicy().require_test(store.snapshot().data, self.contract.verify_frozen)

    def pilot(self, store, execute, settings):
        store.commit("pilot", updates={"phase": "P0", "budget_start": time.time(), "estimates": {"candidate_job_h": 0.01}})

    def provenance(self):
        return {"task": "synthetic"}

    def report(self, view):
        write_json(str(Path(view.root) / "summary.json"), {"selection": view.state["selection"],
                   "p3": view.state.get("p3_stats", view.state["p3"])})


class SyntheticTask:
    """Test assembly helper, not an alternate implementation of the experimental stages."""
    def __init__(self, store):
        import yaml
        from experiment.assembly import RunPaths, build_experiment
        from dataclasses import asdict
        protocol_path = Path(store.root) / "protocol.yaml"
        if not protocol_path.exists():
            original = Path(__file__).resolve().parents[2] / "protocol.yaml"
            protocol = yaml.safe_load(original.read_text())
            protocol["search"].update(max_agent_candidates=3, rounds=3)
            protocol["confirm"]["top_k"] = 1
            protocol["confirm"]["seeds"] = [1001, 1002]
            protocol["final"]["formal_seeds"] = [2000]
            protocol["budget"].pop("disk_gb", None)
            protocol_path.write_text(yaml.safe_dump(protocol))
        self.store, self.state = store, store.state
        self.task = NumericAdapter()
        self.backend = FakeAgentBackend(self.respond)
        self.executor = FakeExecutor(self.task.measure)
        paths = RunPaths(store.root, store.root, store.root, store.root, str(Path(store.root).parent), str(protocol_path))
        self.app = build_experiment(Path(store.root).name, paths=paths, store=store, task=self.task,
                                    backend=self.backend, executor=self.executor)
        self.policy, self.design = self.app.policy, self.app.design
        self.contract, self.evidence = self.task.contract, self.app.evidence

    def respond(self, prompt, work, inp, log, **options):
        assert (Path(inp) / "results.json").is_file()
        path = Path(work)
        cid = path.parent.name
        if prompt.startswith("REPAIR") or cid != "c002":
            (path / "proposal.md").write_text("query:a-17 urn:query/beta")
        else:
            (path / "proposal.md").write_text("missing evidence")
        (path / "program.txt").write_text("1" if cid == "c001" else "2")
        return {"ok": True, "session_id": "text-session", "wall_s": 0, "models": ["fake"]}

    def handlers(self):
        return self.app.handlers()

    def run(self):
        self.app.run()
