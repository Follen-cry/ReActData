"""Second-refactor regression: shared science, durable effects and narrow data boundaries."""
from pathlib import Path
import sys
import math
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest

PIPELINE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PIPELINE), str(PIPELINE / "runner"), str(PIPELINE / "api"), str(Path(__file__).resolve().parents[1])]
from common import read_json, write_json
from experiment.runtime.store import RunStore
from experiment.workflow.design import NoiseAwareDesign
from experiment.workflow import operations
from experiment.improvement.agents import AgentService, FakeAgentBackend
from experiment.runtime.journal import OutcomeUnknown
from experiment.evidence import EvidenceService, VisibilityPolicy
from support.synthetic_task import SyntheticTask, NumericAdapter


def new_store(path):
    return RunStore(str(path), {"phase": "INIT", "candidates": {}, "failures": [], "rejects_in_a_row": 0})


def test_numeric_task_uses_shared_scientific_stages_and_real_confirmation(tmp_path):
    task = SyntheticTask(new_store(tmp_path))
    assert isinstance(task.app.design, NoiseAwareDesign)
    assert not any(hasattr(NumericAdapter, name) for name in ("p1", "confirm", "select_v3", "p3_v3", "handlers"))
    handlers = task.handlers()
    assert handlers["baseline"].func is operations.run_baseline
    assert handlers["confirm"].func is operations.run_confirmation
    assert handlers["final"].func is operations.run_final
    task.run()
    result = task.state["confirm"]["c002"]
    assert result["R"] == 2 and result["m_comparisons"] == 1
    assert result["SE_train"] > 0 and "SE_item" in result
    assert task.state["selection"]["selected"] == "c002"


def test_snapshot_is_immutable_and_parallel_commits_keep_all_updates(tmp_path):
    store = new_store(tmp_path)
    snapshot = store.snapshot()
    with pytest.raises(TypeError):
        snapshot.data["candidates"]["p"] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: store.commit("checked", candidates={str(n): {"status": "checked"}}), range(8)))
    assert len(read_json(str(tmp_path / "state.json"))["candidates"]) == 8
    assert not snapshot.data["candidates"]


def test_agent_receipt_recovers_output_after_return_before_bookkeeping(tmp_path):
    store = new_store(tmp_path)
    result = {"ok": True, "wall_s": 0, "session_id": "durable", "cost_usd": 0.75}
    def respond(prompt, work, inp, directory, **options):
        write_json(str(Path(directory) / "output.json"), result)
        raise KeyboardInterrupt("crash after CLI result")
    backend = FakeAgentBackend(respond)
    protocol = {"search": {"agent_retry": 1, "agent": {"timeout_s": 1, "max_turns": 1, "session": "resume_chain"}}}
    with pytest.raises(KeyboardInterrupt):
        AgentService(store, protocol, backend).propose("p", "work", "inp", str(tmp_path / "round"), "FULL", "SHORT")
    recovered = AgentService(store, protocol, backend)
    assert recovered.propose("p", "work", "inp", str(tmp_path / "round"), "FULL", "SHORT")[0] == result
    assert len(backend.calls) == 1 and store.state["agent_session"] == "durable"


def test_unknown_external_outcome_never_repeats_agent(tmp_path):
    store = new_store(tmp_path)
    directory = tmp_path / "round/agent_0"
    write_json(str(directory / "operation.json"), {"status": "started"})
    backend = FakeAgentBackend(lambda *a, **k: pytest.fail("must not repeat an ambiguous paid call"))
    protocol = {"search": {"agent_retry": 1, "agent": {"timeout_s": 1, "max_turns": 1, "session": "resume_chain"}}}
    with pytest.raises(OutcomeUnknown):
        AgentService(store, protocol, backend).propose("p", "work", "inp", str(tmp_path / "round"), "FULL", "SHORT")
    assert backend.calls == []


def test_candidate_recovers_after_checks_interruption_without_agent_or_cost_duplication(tmp_path, monkeypatch):
    store = new_store(tmp_path)
    task = SyntheticTask(store)
    task.handlers()["pilot"]()
    task.handlers()["baseline"]()
    def crash(*args):
        raise KeyboardInterrupt("CPU checker interrupted")
    monkeypatch.setattr(task.contract, "check_candidate", crash)
    with pytest.raises(KeyboardInterrupt):
        task.app.candidates.make_candidate("c001", 1, 0, [])
    assert len(task.backend.calls) == 1
    recovered = SyntheticTask(RunStore(store.root))
    assert recovered.app.candidates.make_candidate("c001", 1, 0, [])
    assert recovered.backend.calls == []
    assert len(recovered.state["candidates"]["c001"]["agent_calls"]) == 1


def test_authorized_renderer_receives_values_and_explicit_files_only(tmp_path):
    from runner import report
    from tasks.reactdata.evidence import ReactDataEvidence
    request = {"round_dir": str(tmp_path), "entries": [{"id": "p", "role": "agent", "status": "trained",
               "comparison": {"secret": "HIDDEN"}, "cost": {"secret": "HIDDEN"}, "audit": {"secret": "HIDDEN"}}],
               "pool_stats": {"summary": {"private": "HIDDEN"}, "example_clips": []}, "notes": ["HIDDEN"],
               "visibility": VisibilityPolicy(sections=[])}
    source = ReactDataEvidence(report.write_report)
    authorized = source.collect_evidence(request)
    assert not hasattr(authorized, "run_dir") and not hasattr(authorized, "entries")
    source.render_evidence(authorized)
    assert all("HIDDEN" not in p.read_text() for p in (tmp_path / "input").rglob("*") if p.is_file())


def test_statistical_kernel_accepts_explicit_strata_and_keeps_noise_formula():
    from experiment.workflow.statistics import Statistics
    metrics = Statistics(lambda item_id: "a" if item_id.startswith("opaque:") else "b")
    a = {"opaque:one": 0.2, "q/one": 0.6, "q/two": 0.6}
    b = {i: 0.0 for i in a}
    assert metrics.agg(a) == pytest.approx(0.4)  # equal strata weights, not 3-item average
    baseline = [{i: 0.2 for i in a}, {i: 0.3 for i in a}]
    candidate = [{i: v + 0.1 for i, v in r.items()} for r in baseline]
    result = metrics.confirm_compare(candidate, baseline, 0.02, m=3)
    assert result["D"] == pytest.approx(0.1)
    assert result["SE_train"] == pytest.approx(math.sqrt(2) * 0.02 / math.sqrt(2))
    assert result["ci_lo"] == pytest.approx(result["D"] - result["z"] * result["SE"])


def test_slot_lease_releases_after_failure_without_releasing_another_owner():
    from experiment.runtime.slots import SlotPool
    from support.fakes import Slot
    slot = Slot("A", "fake", "0", 1)
    pool = SlotPool([slot], 1, threading.RLock(), lambda *a: True)
    with pytest.raises(RuntimeError):
        with pool.lease():
            raise RuntimeError("launch failed")
    assert slot.job is None
    with pool.lease():
        assert slot.job is not None
    slot.job = {"owned": True}
    with pytest.raises(RuntimeError, match="already reserved"):
        with pool.lease(slot.to_json()):
            pass
    assert slot.job == {"owned": True}


def test_training_journal_resume_truncates_only_uncommitted_steps(tmp_path):
    from training_journal import TrainingJournal
    journal = TrainingJournal(str(tmp_path))
    journal.open("signals.jsonl", 0)
    for step in range(4):
        journal.write("signals.jsonl", {"step": step, "summary": step})
    journal.close()
    recovered = TrainingJournal(str(tmp_path))
    recovered.open("signals.jsonl", 2)
    recovered.write("signals.jsonl", {"step": 2, "summary": "replayed"})
    recovered.close()
    rows = [json.loads(line) for line in (tmp_path / "signals.jsonl").read_text().splitlines()]
    assert rows == [{"step": 0, "summary": 0}, {"step": 1, "summary": 1}, {"step": 2, "summary": "replayed"}]


def test_probe_state_uses_original_budget_limits_and_roundtrips():
    from tasks.reactdata.probes import ProbeController
    import plugin_api as PA
    controller = ProbeController(["p"], {"t"}, lambda recipe: "text")
    request = {"kind": "generate", "clip_ids": ["p"]}
    ok, rejected = controller.check([request], 0)
    assert ok == [request] and rejected == []
    controller.accept(ok, 0)
    controller.pending = [{"kind": "generate", "items": []}]
    restored = ProbeController(["p"], {"t"}, lambda recipe: "text")
    restored.load_state_dict(json.loads(json.dumps(controller.state_dict())))
    assert restored.state_dict() == controller.state_dict()
    assert restored.budget()["generate_probes_left"] == PA.MAX_GENERATE_PROBES - 1
    assert restored.check([request], PA.GENERATE_PROBE_MIN_GAP - 1)[1]
    # v6: MAX_GENERATE_PROBES = 1, so after one generate probe the count limit (not the gap) rejects the next one
    expected = [request] if PA.MAX_GENERATE_PROBES > 1 else []
    assert restored.check([request], PA.GENERATE_PROBE_MIN_GAP)[0] == expected


def test_recover_confirmation_after_jobs_complete_before_stage_commit(tmp_path):
    store = new_store(tmp_path)
    task = SyntheticTask(store)
    execute = task.app.execute
    def crash(spec):
        result = execute(spec)
        if read_json(spec)["run_id"] == "c002-s1001":
            raise KeyboardInterrupt("after confirmation job DONE")
        return result
    task.app.execute = crash
    with pytest.raises(KeyboardInterrupt):
        task.run()
    recovered = SyntheticTask(RunStore(store.root))
    recovered.run()
    assert set(recovered.executor.executed) == {"base_test", "formal_c000-s2000", "formal_c002-s2000"}
    assert recovered.backend.calls == []


def test_recover_after_freeze_without_replacing_hash_or_repeating_search(tmp_path, monkeypatch):
    store = new_store(tmp_path)
    task = SyntheticTask(store)
    freeze = task.task.freeze
    def crash(*args):
        freeze(*args)
        raise KeyboardInterrupt("after freeze committed")
    monkeypatch.setattr(task.task, "freeze", crash)
    with pytest.raises(KeyboardInterrupt):
        task.run()
    digest = store.state["frozen"]["code_sha256"]
    recovered = SyntheticTask(RunStore(store.root))
    recovered.run()
    assert recovered.state["frozen"]["code_sha256"] == digest
    assert recovered.backend.calls == []
    assert set(recovered.executor.executed) == {"base_test", "formal_c000-s2000", "formal_c002-s2000"}


def test_legacy_design_uses_same_task_without_confirmation(tmp_path):
    import yaml
    from experiment.workflow.design import LegacyDesign
    store = new_store(tmp_path)
    SyntheticTask(store)  # create an isolated protocol copy, not any experiment results
    path = tmp_path / "protocol.yaml"
    protocol = yaml.safe_load(path.read_text())
    protocol["design"] = "v1"
    protocol["training"].update(search_run_seed=1000, formal_run_seed=2000)
    protocol["evaluation"]["S_null"] = [1]
    path.write_text(yaml.safe_dump(protocol))
    # No persisted state/job exists yet; drop only the initialization provenance in this test's fresh store.
    store.state.pop("run_json")
    task = SyntheticTask(store)
    assert isinstance(task.app.design, LegacyDesign)
    task.run()
    assert "confirm" not in task.state
    assert task.state["selection"]["selected"] == "c002"
    assert set(task.state["p3"]) == {"baseline_formal", "base_test", "selected_formal"}


def test_validation_receipt_cannot_hide_new_unexpected_files(tmp_path):
    from experiment.improvement.submissions import validate_submission
    work = tmp_path / "work"
    work.mkdir()
    (work / "program.txt").write_text("1")
    contract = SimpleNamespace(files=("program.txt",), proposal_errors=lambda work: [],
        check_candidate=lambda work, scratch: {"errors": [] if set(Path(work).iterdir()) == {Path(work) / "program.txt"} else ["unexpected file"]})
    assert validate_submission(contract, str(work), str(tmp_path))["errors"] == []
    (work / "unexpected.py").write_text("import os")
    assert validate_submission(contract, str(work), str(tmp_path))["errors"] == ["unexpected file"]
