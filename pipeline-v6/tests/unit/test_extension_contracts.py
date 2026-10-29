"""Extension points must preserve authorization and per-instance resources."""
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

PIPELINE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PIPELINE), str(PIPELINE / "tests")]

from experiment.assembly import build_experiment
from experiment.contracts import RunPaths, PromptBundle, JobRequest
from experiment.runtime.io import read_json, read_jsonl, write_json, write_jsonl
from experiment.runtime.store import RunStore
from experiment.evidence import EvidenceService, VisibilityPolicy
from support.synthetic_task import SyntheticTask


@pytest.mark.parametrize("override", [False, True])
def test_assembly_guards_custom_executor_and_override_even_for_completed_jobs(tmp_path, override):
    store = RunStore(str(tmp_path), {"phase": "INIT", "candidates": {}, "failures": [], "rejects_in_a_row": 0})
    setup = SyntheticTask(store)
    calls = []
    backend_calls = []
    backend = SimpleNamespace(execute=lambda spec: backend_calls.append(spec) or "done")
    paths = RunPaths(str(tmp_path), "unused", "unused", "unused", str(tmp_path.parent), str(tmp_path / "protocol.yaml"))
    app = build_experiment(tmp_path.name, paths=paths, task=setup.task, store=store,
                           backend=setup.backend, executor=backend,
                           execute_override=(lambda spec: calls.append(spec) or "done") if override else None)
    spec = store.job_spec("formal", {"run_id": "formal", "run_dir": str(tmp_path / "jobs/formal"),
                                    "payload": {"split": "test"}})
    write_json(str(tmp_path / "jobs/formal/DONE.json"), {})
    for execute in (app.execute, app.executor.execute):
        with pytest.raises(RuntimeError):
            execute(spec)
    assert calls == backend_calls == []
    store.update_state(candidates={"c000": {"code_dir": setup.task.baseline_dir}})
    frozen = setup.task.freeze(store, "c000")
    store.begin_phase("P3")
    assert app.execute(spec) == "done"
    assert len(calls if override else backend_calls) == 1
    (Path(frozen) / "program.txt").write_text("tampered")
    with pytest.raises(RuntimeError):
        app.execute(spec)
    assert len(calls if override else backend_calls) == 1


def task_paths(root, label):
    exp = root / label
    exp.mkdir()
    shutil.copytree(PIPELINE / "manifests", exp / "manifests")
    shutil.copytree(PIPELINE / "api", exp / "api", ignore=shutil.ignore_patterns("__pycache__"))
    (exp / "BUILDER_AGENT_v5.md").write_text(label)
    paths = RunPaths(str(exp), str(exp / "api"), str(exp / "data"), str(exp / "manifests"),
                     str(exp / "runs"), str(exp / "protocol.yaml"))
    return paths


def test_two_tasks_keep_evidence_checks_materialization_and_audit_resources_separate(tmp_path):
    from tasks.reactdata import ReactDataTask
    from runner.materialize import Materializer
    from runner.checks import Sim
    from runner.audit import audit_run

    paths = [task_paths(tmp_path, label) for label in ("alpha", "beta")]
    tasks, materializers, hosts = [], [], []

    class Host:
        def __init__(self, candidate, pool, scratch, log, **kwargs):
            hosts.append((pool, kwargs["api"]))
        def call(self, *args, **kwargs):
            return None
        def close(self):
            pass

    for location in paths:
        label = Path(location.exp).name
        task = ReactDataTask({"search": {"agent": {"instructions": "BUILDER_AGENT_v5.md"}}}, location)
        task.bind(RunStore(str(Path(location.runs) / "test"), {"phase": "P2", "candidates": {}}), "hash")
        tasks.append(task)
        mat = Materializer(str(PIPELINE / "seeds/fixed_f0_final"), paths=location, prompt_wrapper=lambda t, s: s)
        materializers.append(mat)
        sim = Sim(str(PIPELINE / "seeds/fixed_f0_final"), "unused", "unused", 1, "sig", mat.template_ids,
                  mat.candidate_yaml, paths=location, host_factory=Host, prompt_wrapper=lambda t, s: s)
        assert sim.mat.pool_view_path == mat.pool_view_path
        sim.close()
        # Real audit reads each task's private training metadata through its own paths.
        clip = next(iter(mat.clip_ids))
        write_json(str(Path(location.data) / "pool_view.json"), {"clips": [{"clip_id": clip, "fg_area": [1]}]})
        write_json(str(Path(location.data) / "meta/train" / (clip + ".json")),
                   {"parameters": {"scale_from": 1, "scale_to": 2}})
        run = Path(location.runs) / "audit"
        write_jsonl(str(run / "rows_realized.jsonl"), [{"clip_id": clip, "template_id": label}])
        assert audit_run(str(run), paths=location)["templates"] == {label: 1}

    # Alternate use after both tasks are constructed: no last-created-task global wins.
    for task, mat, location in zip(tasks, materializers, paths):
        service = EvidenceService(task.evidence_source, VisibilityPolicy(["api", "instructions"]))
        inp = Path(service.build(str(Path(location.runs) / "evidence"), [], {}, [], instructions="BUILDER_AGENT_v5.md"))
        assert (inp / "BUILDER_AGENT.md").read_text() == Path(location.exp).name
        assert (inp / "api/recipe.py").read_bytes() == (PIPELINE / "api/recipe.py").read_bytes()
        assert mat.frame("clip", 0) == str(Path(location.data) / "frames512/clip/f0.png")
        assert task.contract.manifests == location.manifests
    assert hosts == [(str(Path(p.data) / "pool_view.json"), p.api) for p in paths]


def test_candidate_prompt_contract_has_named_fields():
    from tasks.reactdata.prompts import PromptBuilder
    from experiment.improvement.prompts import PromptContext
    context = PromptContext({"training": {"screen_seed": 1}}, {}, 1, 4, [0], 1, [2], {})
    result = PromptBuilder().build(context, "c001", 1, 0, "c000", [], "instructions")
    assert isinstance(result, PromptBundle)
    assert result.full.startswith("instructions")
    assert "c001" in result.continuation and result.notes


def test_explicit_api_mount_keeps_same_sandbox_destinations(tmp_path):
    from runner.plugin_host import bwrap_cmd
    original = bwrap_cmd("/candidate", "/pool", "/scratch")
    custom = bwrap_cmd("/candidate", "/pool", "/scratch", api=str(tmp_path / "api"))
    differences = [i for i, (a, b) in enumerate(zip(original, custom)) if a != b]
    assert len(original) == len(custom) and len(differences) == 1
    index = differences[0]
    assert custom[index - 1] == "--ro-bind" and custom[index + 1] == "/api"


def test_new_job_specs_carry_resources_without_rewriting_legacy_specs(tmp_path):
    from dataclasses import asdict
    from tasks.reactdata import ReactDataTask
    paths = task_paths(tmp_path, "resources")
    Path(paths.protocol).write_text("frozen test protocol")
    store = RunStore(str(Path(paths.runs) / "run"), {"phase": "P1", "candidates": {}})
    task = ReactDataTask({"training": {}}, paths)
    request = JobRequest("base", None, 0, 0, "dev", (0,))
    spec = task.build_spec(store, request, None, SimpleNamespace(max_rows=4))
    assert read_json(spec)["resources"] == asdict(paths)
    old = read_json(spec)
    old.pop("resources")
    write_json(spec, old)
    before = Path(spec).read_bytes()
    assert task.build_spec(store, request, None, SimpleNamespace(max_rows=4)) == spec
    assert Path(spec).read_bytes() == before


def test_reactdata_evidence_plan_serializes_snapshot_without_changing_display_rules(tmp_path):
    from tasks.reactdata.evidence import ReactDataEvidence
    store = RunStore(str(tmp_path), {"phase": "P2", "candidates": {
        f"c{i:03}": {"status": "trained"} for i in range(7)},
        "baseline_mean_items": {"q": 0.2}, "baseline_sd_items": {"q": 0.01}})
    options = ReactDataEvidence.search_options(store.snapshot().data, "c001")
    assert options["gallery_ids"] == {"c000", "c001", "c003", "c004", "c005", "c006"}
    from support.synthetic_task import TextEvidence
    service = EvidenceService(TextEvidence(), VisibilityPolicy())
    service.build(str(tmp_path / "round"), [], {}, [], **options)
    saved = read_json(str(tmp_path / "round/evidence_request.json"))["request"]
    assert saved["baseline_items"] == [{"q": 0.2}, {"q": 0.01}]
    assert saved["gallery_ids"] == sorted(options["gallery_ids"])
