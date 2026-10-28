"""Run the ORIGINAL flow assertions with temporary data and trusted CPU plugins.

This harness does not test sandbox isolation. Never use TrustedHost for untrusted code.
The production PluginHost and the sandbox test are deliberately unchanged.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration import test_v3_flow as flow
import checks
import common
from runner import materialize, report


class TrustedHost:
    def __init__(self, cand, pool_path, *args, **kwargs):
        self.cand, self.pool = Path(cand), common.read_json(pool_path)

    def call(self, op, **kw):
        if op == "init":
            def load(name, cls, *args):
                spec = importlib.util.spec_from_file_location(name, self.cand / (name + ".py"))
                mod = importlib.util.module_from_spec(spec)
                # A real worker has a read-only candidate mount and cannot create pycache.
                old = sys.dont_write_bytecode
                try:
                    sys.dont_write_bytecode = True
                    spec.loader.exec_module(mod)
                finally:
                    sys.dont_write_bytecode = old
                return getattr(mod, cls)(*args)
            self.b = load("databuilder", "Builder", kw["builder_config"], self.pool, kw["seed"], kw["total_steps"])
            self.a = load("analysis", "Analysis", kw["analysis_config"], kw["seed"], kw["total_steps"], kw["inner_probe_ids"])
        elif op == "next_batch":
            return json.loads(json.dumps(self.b.next_batch(kw["step"])))
        elif op == "after_step":
            s = self.a.update(kw["step"], kw["rows"], kw["step_info"], kw["probe_results"])
            self.b.observe(kw["step"], s)
            return {"summary": s, "requests": self.a.request_probes(kw["step"], kw["budget"])}
        elif op == "state":
            return {"builder": self.b.state_dict(), "analysis": self.a.state_dict()}
        elif op == "load":
            self.b.load_state_dict(kw["state"]["builder"])
            self.a.load_state_dict(kw["state"]["analysis"])

    def close(self):
        pass


def run():
    # Everything the upstream test deletes or writes lives under this directory.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        exp, data, runs = (root / n for n in ("exp", "data", "runs"))
        exp.mkdir()
        for d in ("api", "manifests", "seeds"):
            shutil.copytree(Path(flow.EXP) / d, exp / d)
        for f in ("protocol.yaml", "BUILDER_AGENT_v5.md"):
            shutil.copy2(Path(flow.EXP) / f, exp / f)
        data.mkdir()
        common.write_json(str(data / "BUILD_DONE.json"), {"counts": {"fake": True}})
        clips = common.read_jsonl(str(exp / "manifests/train_pool.jsonl"))
        common.write_json(str(data / "pool_view.json"), {
            "task": "shape_color_then_scale", "n_frames": 60, "fps": 16,
            "clips": [{"clip_id": c["clip_id"], "fg_area": [0] * 60, "frame_change": [0] * 59} for c in clips]})
        frames = data / "frames512"
        for c in clips[:3]:
            d = frames / c["clip_id"]
            d.mkdir(parents=True)
            for n in ["f0.png", "final.png"] + [f"mp4_{t:02d}.png" for t in report.EXAMPLE_FRAMES if t]:
                (d / n).write_bytes(b"fake-image")
        for split in ("dev", "test"):
            p = exp / "manifests" / (split + ".jsonl")
            rows = common.read_jsonl(str(p))
            image = str(data / "fake.png")
            Path(image).write_bytes(b"fake-image")
            for row in rows:
                row["image"], row["target_image"] = [image], image
            common.write_jsonl(str(p), rows)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for mod in (flow, flow.SM, checks, common, materialize, report):
                for k, v in {"EXP": exp, "DATA": data, "RUNS": runs, "FRAMES": frames,
                             "MANIFESTS": exp / "manifests", "API": exp / "api"}.items():
                    if hasattr(mod, k):
                        stack.enter_context(patch.object(mod, k, str(v)))
            stack.enter_context(patch.object(checks, "PluginHost", TrustedHost))
            stack.enter_context(patch.object(materialize, "wrap_prompt", lambda task, text: text))
            # The scorer's external CV library is absent; only provenance reads it here.
            import types
            stack.enter_context(patch.dict(sys.modules, {"runner.score": types.SimpleNamespace(scorer_version=lambda: {"fake": True})}))
            # Upstream main mutates these compatibility hooks; restore them afterwards.
            for obj, attr in ((flow.SM.Runner, "execute"), (flow.SM, "call_agent"), (flow.SM, "audit_run")):
                stack.enter_context(patch.object(obj, attr, getattr(obj, attr)))
            original_run = flow.SM.Runner.run
            def checked_run(runner):
                original_run(runner)
                # Rebuild actual reactdata inputs, including images and summaries, from records.
                if hasattr(runner, "evidence"):
                    from common import sha256_file
                    request = Path(runner.root) / "rounds/002/c002/evidence_request.json"
                    inp = request.parent / "input"
                    snapshot = lambda: {str(p.relative_to(inp)): sha256_file(str(p)) for p in inp.rglob("*") if p.is_file()}
                    before = snapshot()
                    # Hidden evaluator artifacts are retained in the source, never copied to /input.
                    job = Path(runner.state["candidates"]["c000"]["run_dir"])
                    write_secret = job / "eval/test/scorer_source.py"
                    write_secret.parent.mkdir(parents=True)
                    write_secret.write_text("FORBIDDEN_EVIDENCE")
                    (job / "generator_metadata.json").write_text("FORBIDDEN_EVIDENCE")
                    runner.evidence.rebuild(str(request))
                    assert before == snapshot()
                    assert all(b"FORBIDDEN_EVIDENCE" not in p.read_bytes() for p in inp.rglob("*") if p.is_file())
                    events = (Path(runner.root) / "events.jsonl").read_bytes()
                    original_run(runner)
                    assert events == (Path(runner.root) / "events.jsonl").read_bytes()
            stack.enter_context(patch.object(flow.SM.Runner, "run", checked_run))
            flow.CALLS["n"] = 0
            flow.INPUTS.clear()
            flow.REPAIRS.clear()
            flow.main()


if __name__ == "__main__":
    run()
