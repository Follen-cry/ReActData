"""Composition root. Configuration is interpreted once; no task module globals change."""
from experiment.contracts import RunPaths
import os
import sys
import yaml
from experiment.runtime.io import read_json, sha256_file, sha256_json, write_json
from experiment.runtime.provenance import implementation_manifest
from experiment.runtime.store import RunStore
from experiment.workflow.design import make_design
from experiment.application import ExperimentApplication
from experiment.improvement.agents import AgentService
from experiment.evidence import EvidenceService, VisibilityPolicy
from experiment.evidence.execution import AuthorizedExecutor
from experiment.improvement.candidates import CandidateService
from experiment.runtime.resources import ResourceManager
from experiment.runtime.continuation import inherit_run
from experiment import registry


def default_paths(protocol):
    common = runner_common()
    return RunPaths(common.EXP, common.API, common.DATA, common.MANIFESTS, common.RUNS,
                    os.path.abspath(protocol or os.path.join(common.EXP, "protocol.yaml")))


def runner_common():
    # Honor the historical patched module at the compatibility boundary, otherwise use package imports.
    if "common" in sys.modules:
        return sys.modules["common"]
    from runner import common
    return common


def build_experiment(run_id, smoke=False, replay_src=None, protocol=None, continue_from=None, *, paths=None,
                     hooks=None, backend=None, executor=None, search_policy=None, task=None, store=None, execute_override=None):
    hooks = hooks or {}
    paths = paths or default_paths(protocol)
    with open(paths.protocol) as f:
        config = yaml.safe_load(f)
    components = config.get("components", {})
    root = os.path.join(paths.runs, run_id)
    state_path = os.path.join(root, "state.json")
    state = read_json(state_path) if os.path.isfile(state_path) else {
        "run_id": run_id, "smoke": smoke, "phase": "INIT", "candidates": {}, "rejects_in_a_row": 0, "failures": []}
    store = store or RunStore(root, state, os.path.join(paths.exp, "records", run_id))
    digest = sha256_file(paths.protocol)
    if "run_json" in store.state and store.state["run_json"]["protocol_sha256"] != digest:
        raise SystemExit("the protocol changed since this run started: start a new run id (new protocol version)")
    design = make_design(config, smoke)
    policy = search_policy or registry.search_policy(components.get("search_policy", "default"), config.get("design", "v1"))
    task = task or registry.task(components.get("task", "reactdata"), protocol=config, paths=paths, smoke=smoke, hooks=hooks)
    task.bind(store, digest)
    selection = registry.selection_policy(components.get("selection_policy"), config.get("design", "v1"), task.statistics)
    resources = ResourceManager(store, config, policy)
    if executor is None:
        from runner import jobs
        executor = registry.executor(components.get("executor", "slots"), store=store, protocol=config, resources=resources,
                                     transport=jobs, authorize=lambda spec: task.authorize_job(store, spec))
    if backend is None:
        from runner.agent_call import call_agent
        backend = registry.agent_backend(components.get("agent_backend", "claude"), hooks.get("call_agent", call_agent))
    agent = AgentService(store, config, backend)
    evidence = EvidenceService(task.evidence_source, VisibilityPolicy(components.get("visibility", {}).get("sections")))
    s = design.settings
    candidates = CandidateService(store, config, task.contract, policy, evidence, agent, seeds=list(s.S), steps=s.steps,
        per_round=s.per_round, top_k=s.top_k, confirm_seeds=list(s.confirm_seeds), replay_src=replay_src,
        runs_root=paths.runs, prompt_builder=task.prompts)
    # Hash the executing source tree, even when resources live in a temporary/data tree.
    implementation = implementation_manifest(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "run_json" not in store.state:
        record = {"protocol": os.path.relpath(paths.protocol, paths.exp), "protocol_sha256": digest,
                  "runner_sha256": sha256_json(implementation), **task.provenance()}
        store.state["run_json"] = record
        write_json(os.path.join(root, "run.json"), record)
    store.record_implementation(implementation)
    if continue_from and "inherited_from" not in store.state:
        inherit_run(store, paths.runs, smoke, continue_from)
    elif continue_from and store.state["inherited_from"]["run"] != continue_from:
        raise SystemExit(f"this run already continues {store.state['inherited_from']['run']}")
    authorize = lambda spec: task.authorize_job(store, spec)
    guarded_executor = AuthorizedExecutor(executor.execute, authorize)
    execute = AuthorizedExecutor(execute_override, authorize).execute if execute_override else guarded_executor.execute
    return ExperimentApplication(run_id, smoke, config, store, task, design, policy, selection, guarded_executor,
                                 resources, candidates, evidence, execute)
