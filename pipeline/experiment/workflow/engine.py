"""Explicit, resumable stage dispatch. No task fields or execution backend here."""
from experiment.contracts import StagePlan


class ExperimentEngine:
    def __init__(self, store, design: StagePlan, handlers):
        self.store, self.design, self.handlers = store, design, handlers

    def run(self):
        stages = self.design.stages()
        order = ["INIT"] + [stage.name for stage in stages] + ["DONE"]
        if len(set(order)) != len(order):
            raise ValueError("stage plan contains duplicate phase names")
        for stage in stages:
            if stage.operation not in self.handlers:
                raise ValueError(f"missing task operation {stage.operation}")
        done = self.store.state.get("done_upto", "INIT")
        # Old runs could record the intermediate freeze marker inside the P3 operation.
        if done == "FROZEN":
            done = "SELECT"
        if done not in order:
            raise ValueError(f"unknown recovery marker {done}")
        for stage in stages:
            if order.index(done) >= order.index(stage.name):
                continue
            self.handlers[stage.operation]()
            self.store.complete_phase(stage.name)
