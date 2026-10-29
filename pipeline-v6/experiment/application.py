"""Small application assembly result. Scientific stages are shared functions."""
from dataclasses import dataclass
from functools import partial
from experiment.workflow.engine import ExperimentEngine
from experiment.improvement.search import SearchController
from experiment.workflow import operations
from experiment.contracts import TaskAdapter, JobExecutor, ReportContext


@dataclass
class ExperimentApplication:
    run_id: str
    smoke: bool
    protocol: dict
    store: object
    task: TaskAdapter
    design: object
    policy: object
    selection: object
    executor: JobExecutor
    resources: object
    candidates: object
    evidence: object
    execute: object

    def handlers(self):
        store, task, design, execute = self.store, self.task, self.design, self.execute
        screen = partial(operations.run_screening, store, design, task, self.resources, self.protocol, execute)
        return {
            "pilot": partial(task.pilot, store, execute, design.settings),
            "baseline": partial(operations.run_baseline, store, design, task, execute),
            "search": SearchController(store, self.policy, self.resources, self.candidates, screen,
                                       self.protocol, design.settings.rounds, design.settings.per_round).run,
            "confirm": partial(operations.run_confirmation, store, design, task, self.resources, self.policy, execute),
            "select": partial(operations.run_selection, store, design, task, self.selection),
            "final": partial(operations.run_final, store, design, task, self.resources, execute),
            "report": self.report,
        }

    def report(self):
        import copy
        self.store.begin_phase("REPORT")
        self.resources._disk_safe()
        s = self.design.settings
        view = ReportContext(state=copy.deepcopy(self.store.state), root=self.store.root, P=self.protocol,
            run_id=self.run_id, smoke=self.smoke, design=self.design.name, S=list(s.S), S_null=list(s.S_null),
            top_k=s.top_k, confirm_seeds=list(s.confirm_seeds), formal_seeds=list(s.formal_seeds))
        self.task.report(view)
        self.store.commit("done", updates={"phase": "DONE"}, elapsed_h=round(self.resources.elapsed_h(), 2))

    def run(self):
        ExperimentEngine(self.store, self.design, self.handlers()).run()
