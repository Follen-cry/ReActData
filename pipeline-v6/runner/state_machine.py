"""Compatible CLI and import facade for the modular experiment runner."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import API, DATA, EXP, MANIFESTS, RUNS, read_json, read_jsonl, sha256_file, sha256_json, sha256_tree, write_json
from agent_call import call_agent
from audit import audit_run
from checks import run_checks
from report import write_report
import jobs
import stats
from tasks import reactdata

_default_audit = audit_run
_default_checks = run_checks

PROTO = os.path.join(EXP, "protocol.yaml")
CAND_FILES = reactdata.CAND_FILES
BACKUP_EXT = reactdata.BACKUP_EXT

# Run the experiment
# python -u runner/state_machine.py --run-id r2 --protocol protocol.yaml

class Runner:
    def __init__(self, run_id, smoke, replay_src=None, protocol=PROTO, continue_from=None, **components):
        from experiment.assembly import RunPaths, build_experiment
        paths = RunPaths(EXP, API, DATA, MANIFESTS, RUNS, os.path.abspath(protocol))
        hooks = {"call_agent": lambda *a, **k: call_agent(*a, **k),
                 "audit_run": lambda *a, **k: audit_run(*a, **({"paths": paths} if audit_run is _default_audit else {}), **k),
                 "run_checks": lambda *a, **k: run_checks(*a, **({"paths": paths} if run_checks is _default_checks else {}), **k),
                 "write_report": lambda *a, **k: write_report(*a, **k)}
        self._app = build_experiment(run_id, smoke, replay_src, protocol, continue_from, paths=paths, hooks=hooks,
                                      execute_override=lambda spec: self.execute(spec), **components)

    @property
    def state(self):
        return self._app.store.state

    @property
    def root(self):
        return self._app.store.root

    @property
    def evidence(self):
        return self._app.evidence

    @property
    def P(self):
        return self._app.protocol

    @property
    def run_id(self):
        return self._app.run_id

    @property
    def smoke(self):
        return self._app.smoke

    @property
    def design(self):
        return self._app.design.name

    @property
    def S(self):
        return list(self._app.design.settings.S)

    @property
    def S_null(self):
        return list(self._app.design.settings.S_null)

    @property
    def top_k(self):
        return self._app.design.settings.top_k

    @property
    def confirm_seeds(self):
        return list(self._app.design.settings.confirm_seeds)

    @property
    def formal_seeds(self):
        return list(self._app.design.settings.formal_seeds)

    def execute(self, spec):
        return self._app.executor.execute(spec)

    def event(self, kind, **values):
        return self._app.store.event(kind, **values)

    def best_so_far(self):
        return self._app.policy.best_so_far(self._app.store.snapshot().data)

    def program_sha(self, directory):
        return self._app.task.contract.program_sha(directory)

    def run(self):
        self._app.run()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replay-decisions", default=None, help="source run id: reuse its candidate code, no LLM calls")
    ap.add_argument("--protocol", default=PROTO, help="protocol file (default protocol.yaml)")
    ap.add_argument("--continue-from", default=None, help="finished source run id to continue (candidates, baselines, agent session)")
    a = ap.parse_args()
    Runner(a.run_id, a.smoke, a.replay_decisions, a.protocol, a.continue_from).run()

if __name__ == "__main__":
    main()
