"""CPU architecture boundaries; no real Agent, cluster or GPU is contacted."""
from pathlib import Path
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

PIPELINE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PIPELINE), str(PIPELINE / "runner"), str(Path(__file__).resolve().parents[1])]
from common import read_json, write_json, sha256_tree
from experiment.evidence.access import EvaluationAccessPolicy
from experiment.improvement.agents import AgentService, FakeAgentBackend
from experiment.workflow.engine import ExperimentEngine
from experiment.evidence import EvidenceService, VisibilityPolicy
from experiment.runtime.execution import SlotExecutor
from experiment.improvement.policy import DefaultSearchPolicy
from experiment.workflow.selection import NoiseAwareSelection
from experiment.improvement.policy import LegacySearchPolicy
from experiment.runtime.resources import ResourceManager
from experiment.runtime.store import RunStore
from support.synthetic_task import SyntheticTask, TextEvidence
from support.fakes import Slot


def store_at(path):
    return RunStore(str(path), {"phase": "INIT", "candidates": {}, "failures": [], "rejects_in_a_row": 0})


def test_second_task_full_flow_and_recovery(tmp_path):
    store = store_at(tmp_path)
    task = SyntheticTask(store)
    task.run()
    c = store.state["candidates"]
    assert store.state["phase"] == "DONE"
    assert c["c002"]["parent"] == "c001"
    assert c["c003"]["status"] == "duplicate"
    assert store.state["selection"]["selected"] == "c002"
    assert {"query:a-17", "urn:query/beta"} == task.contract.valid_dev_ids()
    assert [x["prompt"].split()[0] for x in task.backend.calls] == ["FULL", "CONTINUE", "REPAIR", "CONTINUE"]
    before = list(task.executor.executed)
    # Re-open from JSON; the next Engine does not need any previous Python objects or session state.
    recovered = SyntheticTask(RunStore(store.root))
    recovered.run()
    assert recovered.executor.executed == []
    assert task.executor.executed == before
    index = read_json(str(tmp_path / "research_index.json"))
    row = next(r for r in index["candidates"] if r["candidate_id"] == "c002")
    assert row["parent_id"] == "c001" and row["protocol_decisions"]["selected"]
    assert Path(row["agent_authored"]["memory"]).is_file()
    assert "proved" not in json.dumps(index)


def test_recovery_after_interrupted_confirmation_does_not_repeat_search(tmp_path):
    store = store_at(tmp_path)
    task = SyntheticTask(store)
    handlers = task.handlers()
    def fail():
        raise RuntimeError("simulated interruption")
    handlers["confirm"] = fail
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ExperimentEngine(store, task.design, handlers).run()
    assert store.state["done_upto"] == "P2"
    recovered = SyntheticTask(RunStore(store.root))
    recovered.run()
    assert set(recovered.executor.executed) == {"c002-s1001", "c002-s1002", "base_test", "formal_c000-s2000", "formal_c002-s2000"}
    assert recovered.backend.calls == []


def test_evidence_rebuild_without_session_or_memory(tmp_path):
    task = SyntheticTask(store_at(tmp_path))
    task.run()
    request = tmp_path / "rounds/002/c002/evidence_request.json"
    inp = request.parent / "input"
    before = {p.name: p.read_bytes() for p in inp.iterdir()}
    # New service, no agent/session, no previous input files.
    import shutil
    shutil.rmtree(inp)
    fresh = EvidenceService(TextEvidence(), VisibilityPolicy())
    fresh.rebuild(str(request))
    assert before == {p.name: p.read_bytes() for p in inp.iterdir()}


def test_visibility_projects_before_renderer_and_never_allows_test(tmp_path):
    captured = []
    renderer = SimpleNamespace(build_evidence=lambda r: captured.append(r))
    evidence = EvidenceService(renderer, VisibilityPolicy())
    evidence.build(str(tmp_path), [{"id": "p", "role": "agent", "status": "new", "metadata": "SECRET", "test": "SECRET",
                                  "scorer_source": "SECRET", "credentials": "SECRET"}], {}, [])
    assert "SECRET" not in json.dumps(captured[0]["entries"])
    assert "SECRET" not in (tmp_path / "evidence_request.json").read_text()
    with pytest.raises(PermissionError):
        evidence.build(str(tmp_path), [{"id": "p", "split": "test"}], {}, [])
    jd = tmp_path / "formal"
    jd.mkdir()
    write_json(str(jd / "run.json"), {"evals": [{"name": "test"}]})
    with pytest.raises(PermissionError):
        evidence.build(str(tmp_path), [{"id": "p", "run_dir": str(jd)}], {}, [])
    with pytest.raises(PermissionError):
        VisibilityPolicy().project({"entries": []}, phase="P3")
    assert len(captured) == 1


def test_actual_renderer_uses_disclosure_capabilities(tmp_path, monkeypatch):
    from runner import report
    from tasks.reactdata.evidence import ReactDataEvidence
    # If a denied source is touched this raises, even if copying would fail later.
    def forbidden(*args, **kwargs):
        raise AssertionError("denied source was read")
    monkeypatch.setattr(report.shutil, "copy2", forbidden)
    monkeypatch.setattr(report, "link", forbidden)
    evidence = EvidenceService(ReactDataEvidence(report.write_report), VisibilityPolicy(sections=[]))
    inp = evidence.build(str(tmp_path), [{"id": "p", "role": "agent", "status": "new", "candidate_dir": "/hidden"}],
                         {"summary": {}, "example_clips": ["hidden"]}, [])
    assert not list(Path(inp).rglob("*.py"))
    assert not list(Path(inp).rglob("*.png"))


def test_agent_resume_fallback_and_repair_options(tmp_path):
    store = store_at(tmp_path)
    p = {"search": {"agent_retry": 1, "agent": {"timeout_s": 7, "max_turns": 9, "session": "resume_chain"}}}
    replies = iter([{"ok": True, "session_id": "S1", "wall_s": 0},
                    {"ok": False, "error": "resume missing", "wall_s": 0},
                    {"ok": True, "session_id": "S2", "wall_s": 0}, {"ok": True, "wall_s": 0}])
    backend = FakeAgentBackend(lambda *a, **k: next(replies))
    agent = AgentService(store, p, backend)
    agent.propose("p1", "work", "inp", str(tmp_path / "p1"), "FULL", "SHORT")
    agent.propose("p2", "work", "inp", str(tmp_path / "p2"), "FULL", "SHORT")
    agent.repair("work", "inp", str(tmp_path / "p2"), "REPAIR")
    assert [c["prompt"] for c in backend.calls] == ["FULL", "SHORT", "FULL", "REPAIR"]
    assert [c["options"]["resume"] for c in backend.calls] == [None, "S1", None, "S2"]
    assert all(c["options"]["timeout_s"] == 7 and c["options"]["max_turns"] == 9 for c in backend.calls)
    assert store.state["failures"][0]["kind"] == "agent_resume_failed"


@pytest.mark.parametrize("D,lo,selected", [(0.1, 0.0, "c000"), (0.05, 0.001, "c1"), (0.049, 0.1, "c000")])
def test_selection_thresholds(D, lo, selected):
    assert NoiseAwareSelection.select({"c1": {"D": D, "ci_lo": lo}}, 0.05)["selected"] == selected


def test_policy_limits_and_legacy_schedule():
    policy = DefaultSearchPolicy()
    state = {"rejects_in_a_row": 3}
    search = {"max_agent_candidates": 10, "stop_after_consecutive_rejects": 3}
    assert policy.stop_reason(state, search, 1, True) == "consecutive rejects"
    state["rejects_in_a_row"] = 0
    assert policy.stop_reason(state, search, 1, False) == "budget"
    assert policy.stop_reason(state, search, 1, False, "checked") is None
    assert policy.stop_reason(state, search, 11, True) == "max candidates"
    assert list(policy.candidates(1, 2, 4, 9)) == [(5, [("c010", 0), ("c011", 1)])]
    assert "CONFIRM" not in [s.name for s in LegacySearchPolicy().stages()]


def test_test_execution_needs_freeze_but_disclosure_is_still_denied():
    policy = EvaluationAccessPolicy()
    for state in ({"phase": "P2"}, {"phase": "P3"}):
        with pytest.raises(RuntimeError):
            policy.require_test(state, lambda r: True)
    state = {"phase": "P3", "frozen": {"code_sha256": "hash"}}
    policy.require_test(state, lambda r: True)
    with pytest.raises(RuntimeError):
        policy.require_test(state, lambda r: False)
    with pytest.raises(PermissionError):
        VisibilityPolicy().project({"entries": [{"split": "test"}]})


def test_reactdata_opaque_proposal_ids_and_smoke_standin(tmp_path, monkeypatch):
    from tasks.reactdata.contract import ReactDataContract
    from tasks.reactdata import ReactDataTask
    contract = ReactDataContract(str(tmp_path), {"search": {"proposal": {"sections": ["Evidence"], "min_item_refs": 2}}}, "hash", checks=None, exp=str(tmp_path), data=str(tmp_path), manifests=str(tmp_path))
    monkeypatch.setattr(contract, "valid_dev_ids", lambda: {"query:a-17", "urn:query/beta"})
    proposal = tmp_path / "proposal.md"
    proposal.write_text("## Evidence\nquery:a-17 urn:query/beta")
    assert contract.proposal_errors(str(tmp_path)) == []
    proposal.write_text("## Evidence\nquery:a-170 urn:query/beta")
    assert contract.proposal_errors(str(tmp_path))
    obj = object.__new__(ReactDataTask)
    obj.smoke, obj.access, obj.contract = True, EvaluationAccessPolicy(), contract
    store = store_at(tmp_path / "run")
    store.state["phase"] = "P2"
    obj.paths = SimpleNamespace(manifests=str(tmp_path))
    with pytest.raises(RuntimeError):
        obj.evals(store, "test", [0])
    monkeypatch.setattr(contract, "verify_frozen", lambda frozen: True)
    store.state.update(phase="P3", frozen={"dir": "fake"})
    result = obj.evals(store, "test", [0])[0]
    assert result["smoke_standin"] and Path(result["file"]).name == "dev.jsonl"


def executor_fixture(tmp_path, statuses):
    store = store_at(tmp_path)
    launches = []
    statuses = iter(statuses)
    transport = SimpleNamespace(Slot=Slot, gpus_idle=lambda *a: True,
        launch=lambda slot, spec: (launches.append(spec) or {"start": time.time()}), status=lambda j: next(statuses))
    protocol = {"budget": {"slots": [{"name": "A", "node": "fake", "gpus": "0", "port": 1}], "max_concurrent_jobs": 1},
                "search": {"train_infra_retry": 1}}
    resource = SimpleNamespace(_disk_safe=lambda **k: None)
    ex = SlotExecutor(store, protocol, resource, transport)
    spec = store.job_spec("job", {"run_id": "job", "run_dir": str(tmp_path / "jobs/job")})
    (tmp_path / "jobs/job/train.log").write_text("fake log")
    return store, ex, spec, launches


def test_completed_and_active_jobs_do_not_launch_again(tmp_path):
    store, ex, spec, launches = executor_fixture(tmp_path, ["done", "done"])
    job = read_json(spec)
    write_json(str(Path(job["run_dir"]) / "executor.json"), {**job, "run_json": spec,
               "slot": ex.slots[0].to_json(), "start": time.time()})
    assert ex.execute(spec) == "done" and launches == []
    write_json(str(Path(job["run_dir"]) / "DONE.json"), {})
    assert ex.execute(spec) == "done" and launches == []


def test_infrastructure_retry_retains_limit(tmp_path):
    store, ex, spec, launches = executor_fixture(tmp_path, ["infra_failed"] * 4)
    assert ex.execute(spec) == "infra_failed"
    assert len(launches) == 2
    ex.transport.status = lambda job: "infra_failed"
    assert ex.execute(spec) == "infra_failed"
    assert len(launches) == 2  # recovery never resets attempts to zero


def test_legacy_job_reattachment_and_ambiguous_launch(tmp_path):
    store, ex, spec, launches = executor_fixture(tmp_path, ["done", "done"])
    (tmp_path / "jobs/job/pid").write_text("123")
    store.event("job_launched", job="job", slot="A", node="fake", attempt=0)
    assert ex.execute(spec) == "done" and not launches
    write_json(str(tmp_path / "jobs/job/executor.json"), {"launch_pending": True})
    (tmp_path / "jobs/job/pid").unlink()
    with pytest.raises(RuntimeError, match="ambiguous relaunch"):
        ex.execute(spec)


def test_program_hash_ignores_config_only_but_keeps_code(tmp_path):
    from tasks.reactdata.contract import ReactDataContract
    c = ReactDataContract(str(tmp_path), {"search": {"config_only_is_duplicate": True}}, "hash", checks=None, exp=str(tmp_path), data=str(tmp_path), manifests=str(tmp_path))
    for name in ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml"):
        (tmp_path / name).write_text(name)
    before = c.program_sha(str(tmp_path))
    (tmp_path / "candidate.yaml").write_text("changed salt")
    assert c.program_sha(str(tmp_path)) == before
    (tmp_path / "analysis.py").write_text("changed program")
    assert c.program_sha(str(tmp_path)) != before


def test_explicit_task_registry(tmp_path, monkeypatch):
    from experiment import registry
    store = store_at(tmp_path)
    monkeypatch.setitem(registry.TASKS, "text", SyntheticTask)
    assert isinstance(registry.task("text", store=store), SyntheticTask)
    with pytest.raises(ValueError):
        registry.task("__import__('os')")


def test_slot_executor_cannot_bypass_test_gate_with_a_saved_spec(tmp_path):
    store, ex, spec, launches = executor_fixture(tmp_path, [])
    payload = read_json(spec)
    payload["evals"] = [{"name": "test", "file": "hidden"}]
    write_json(spec, payload)
    with pytest.raises(RuntimeError, match="frozen-program authorization"):
        ex.execute(spec)
    assert launches == []


def test_frozen_program_cannot_be_rehashed_after_tampering(tmp_path):
    from tasks.reactdata import ReactDataTask
    from tasks.reactdata.contract import ReactDataContract
    source = tmp_path / "code"
    source.mkdir()
    for f in ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml"):
        (source / f).write_text(f)
    task = object.__new__(ReactDataTask)
    store = store_at(tmp_path)
    store.state["candidates"] = {"c0": {"code_dir": str(source)}}
    task.contract = ReactDataContract(str(tmp_path), {}, "hash", checks=None, exp=str(tmp_path), data=str(tmp_path), manifests=str(tmp_path))
    frozen = task.freeze(store, "c0")
    assert task.freeze(store, "c0") == frozen
    (Path(frozen) / "databuilder.py").write_text("tampered")
    with pytest.raises(RuntimeError, match="frozen program changed"):
        task.freeze(store, "c0")


def test_rebuild_cannot_widen_original_disclosure_policy(tmp_path):
    seen = []
    task = SimpleNamespace(build_evidence=lambda r: seen.append(r["visibility"].sections))
    EvidenceService(task, VisibilityPolicy(sections=["results"])).build(str(tmp_path), [], {}, [])
    EvidenceService(task, VisibilityPolicy()).rebuild(str(tmp_path / "evidence_request.json"))
    assert seen == [frozenset({"results"}), frozenset({"results"})]


def test_provenance_covers_extracted_modules_and_old_hashes_stay_readable(tmp_path):
    import common
    paths = [str(Path(p).relative_to(PIPELINE)).replace(os.sep, "/") for p in common.implementation_files()]
    assert {"experiment/workflow/engine.py", "experiment/improvement/candidates.py", "tasks/reactdata/adapter.py", "runner/train_entry.py"} <= set(paths)
    store = store_at(tmp_path)
    store.state["run_json"] = {"runner_sha256": "original"}
    store.record_implementation({"experiment/workflow/engine.py": "first"})
    store.record_implementation({"experiment/workflow/engine.py": "second"})
    assert len(read_json(str(tmp_path / "implementations.json"))) == 2
    assert store.state["run_json"]["runner_sha256"] == "original"


def test_sandbox_command_boundaries_are_unchanged():
    from plugin_host import bwrap_cmd
    from agent_call import _bwrap
    plugin = bwrap_cmd("/candidate", "/pool", "/scratch")
    assert "--unshare-all" in plugin and "--share-net" not in plugin
    agent = _bwrap("/agent_home", "/work", "/input")
    assert "--unshare-all" in agent and "--share-net" in agent
    assert any(agent[i:i + 3] == ["--ro-bind", "/input", "/input"] for i in range(len(agent) - 2))
    assert not any("manifests" in arg or "scorers" in arg for arg in agent)


def test_reactdata_renderer_rejects_mislabeled_test_and_source_resources(tmp_path):
    from tasks.reactdata.evidence import ReactDataEvidence
    called = []
    renderer = ReactDataEvidence(lambda **kw: called.append(kw), str(tmp_path / "dev.jsonl"))
    request = {"visibility": VisibilityPolicy(), "entries": [], "instructions": "runner/score.py"}
    with pytest.raises(PermissionError):
        renderer.build_evidence(request)
    request["instructions"] = "BUILDER_AGENT_v5.md"
    job = tmp_path / "job"
    job.mkdir()
    write_json(str(job / "run.json"), {"evals": [{"name": "dev", "file": str(tmp_path / "test.jsonl")}]})
    request["entries"] = [{"run_dir": str(job)}]
    with pytest.raises(PermissionError):
        renderer.build_evidence(request)
    assert called == []


def test_continuation_preserves_noise_and_replicate_reference_without_mutating_source(tmp_path, monkeypatch):
    from experiment.runtime.continuation import inherit_run
    src, dest = tmp_path / "source", tmp_path / "next"
    (src / "jobs").mkdir(parents=True)
    source = {"phase": "DONE", "smoke": False, "candidates": {"c000": {"replicates": {"1000": "job-a"}}},
              "noise": {"sd_train": 0.02, "delta": 0.02}, "baseline_mean_items": {"opaque:id": 0.3},
              "baseline_sd_items": {"opaque:id": 0.01}, "run_json": {"protocol_sha256": "old", "runner_sha256": "old-code"}}
    write_json(str(src / "state.json"), source)
    before = (src / "state.json").read_bytes()
    store = store_at(dest)
    inherit_run(store, str(tmp_path), False, "source")
    assert store.state["noise"] == source["noise"]
    assert store.state["baseline_mean_items"] == source["baseline_mean_items"]
    assert store.state["done_upto"] == "P1"
    assert (src / "state.json").read_bytes() == before
