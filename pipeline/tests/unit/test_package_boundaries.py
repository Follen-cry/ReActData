"""Directory changes must preserve imports, source inventory and sandbox resources."""
import ast
from pathlib import Path
import subprocess
import sys

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))


def test_reusable_packages_do_not_import_task_or_runner_implementations():
    for package in ("workflow", "improvement", "evidence", "runtime"):
        for path in (PIPELINE / "experiment" / package).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                         [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    assert name.split(".")[0] not in {
                        "runner", "tasks", "common", "stats", "agent_call", "assembly"
                    }, (path, name)


def test_public_imports_work_without_runner_path_or_gpu_dependencies():
    # Fresh interpreter: earlier tests must not hide a reliance on sys.path mutations.
    script = f"""
import sys
sys.path[:] = {sys.path!r}
sys.path[:] = [p for p in sys.path if p not in { [str(PIPELINE / 'runner'), str(PIPELINE / 'api')]!r}]
from experiment.assembly import RunPaths, build_experiment
from experiment.evidence import EvidenceService, AuthorizedEvidence
from experiment.workflow.design import NoiseAwareDesign
from tasks.reactdata import ReactDataTask
assert 'runner.common' not in sys.modules and 'common' not in sys.modules
assert 'torch' not in sys.modules and 'plugin_api' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_source_inventory_covers_nested_packages_and_excludes_fixtures(tmp_path):
    from experiment.runtime.provenance import implementation_manifest, implementation_hash
    from experiment.runtime.io import sha256_json
    manifest = implementation_manifest(str(PIPELINE))
    expected = {str(p.relative_to(PIPELINE)).replace("\\", "/")
                for name in ("runner", "experiment", "tasks")
                for p in (PIPELINE / name).rglob("*.py")}
    assert set(manifest) == expected
    assert {"experiment/workflow/engine.py", "tasks/reactdata/gpu_probes.py",
            "experiment/runtime/provenance.py"} <= expected
    assert not any(name.startswith("tests/") for name in manifest)
    assert implementation_hash(str(PIPELINE)) == sha256_json(manifest)
    # A new nested source must enter the hash without extending a hard-coded file list.
    nested = tmp_path / "tasks/another/adapter.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("VALUE = 1\n")
    first = implementation_hash(str(tmp_path))
    nested.write_text("VALUE = 2\n")
    assert first != implementation_hash(str(tmp_path))


def test_sandbox_mount_sources_survive_layout_change():
    sys.path.insert(0, str(PIPELINE / "runner"))
    from plugin_host import bwrap_cmd
    command = bwrap_cmd("/candidate", "/pool", "/scratch")
    mounts = {command[i + 2]: command[i + 1]
              for i, arg in enumerate(command) if arg == "--ro-bind"}
    assert Path(mounts["/runner/plugin_worker.py"]).is_file()
    assert Path(mounts["/api"]).resolve() == (PIPELINE / "api").resolve()
    assert not any("experiment" in target or "tasks" in target for target in mounts)
    assert (PIPELINE / "tests/fixtures/evil_cand/databuilder.py").is_file()
