"""Pure experiment design: job plans and statistical decisions, independent of task."""
from dataclasses import dataclass
from experiment.contracts import JobRequest, Stage
from experiment.workflow.selection import NoiseAwareSelection


@dataclass(frozen=True)
class DesignSettings:
    steps: int
    max_rows: int | None
    S: tuple
    S_null: tuple
    base_seeds: tuple
    confirm_seeds: tuple
    formal_seeds: tuple
    top_k: int
    screen_seed: int
    rounds: int
    per_round: int


def settings(protocol, smoke):
    v3 = protocol.get("design", "v1") == "v3"
    take = lambda xs: tuple(xs[:1] if smoke else xs)
    return DesignSettings(4 if smoke else protocol["training"]["max_steps"], 4 if smoke else None,
        take(protocol["evaluation"]["S"]), take(protocol["evaluation"].get("S_null", [])),
        tuple(protocol["noise"]["baseline_seeds"][:2] if smoke else protocol["noise"]["baseline_seeds"]) if v3 else
            (protocol["training"]["search_run_seed"],),
        take(protocol["confirm"]["seeds"]) if v3 else (),
        take(protocol["final"]["formal_seeds"]) if v3 else (protocol["training"]["formal_run_seed"],),
        (1 if smoke else protocol["confirm"]["top_k"]) if v3 else 0,
        protocol["training"].get("screen_seed", protocol["training"].get("search_run_seed")),
        2 if smoke else protocol["search"]["rounds"], 1 if smoke else protocol["search"]["candidates_per_round"])


class NoiseAwareDesign:
    name = "v3"

    def __init__(self, protocol, smoke=False):
        self.settings = settings(protocol, smoke)

    def stages(self):
        return stage_plan(self.name)

    def baseline_plan(self):
        s = self.settings
        return tuple(JobRequest(f"c000-s{seed}", "c000", seed, s.steps, "dev", s.S) for seed in s.base_seeds) + (
            JobRequest("base", None, 0, 0, "dev", s.S),)

    def confirmation_plan(self, candidates):
        s = self.settings
        return tuple(JobRequest(f"{cid}-s{seed}", cid, seed, s.steps, "dev", s.S)
                     for cid in candidates for seed in s.confirm_seeds)

    def final_plan(self, selected):
        s = self.settings
        arms = ("c000", selected) if selected != "c000" else ("c000",)
        return (JobRequest("base_test", None, 0, 0, "test", self.test_seeds),) + tuple(
            JobRequest(f"{cid}-s{seed}", cid, seed, s.steps, "test", self.test_seeds, True)
            for cid in arms for seed in s.formal_seeds)

    def select(self, confirmation, delta):
        return NoiseAwareSelection.select(confirmation, delta)


class LegacyDesign(NoiseAwareDesign):
    name = "legacy"

    def stages(self):
        return stage_plan(self.name)

    def baseline_plan(self):
        s = self.settings
        return (JobRequest("c000", "c000", s.base_seeds[0], s.steps, "dev", s.S + s.S_null),
                JobRequest("base", None, 0, 0, "dev", s.S))

    def final_plan(self, selected):
        s = self.settings
        jobs = [JobRequest("baseline_formal", "c000", s.formal_seeds[0], s.steps, "test", s.S, True),
                JobRequest("base_test", None, 0, 0, "test", s.S)]
        if selected != "c000":
            jobs.append(JobRequest("selected_formal", selected, s.formal_seeds[0], s.steps, "test", s.S, True))
        return tuple(jobs)


def make_design(protocol, smoke=False):
    design = NoiseAwareDesign(protocol, smoke) if protocol.get("design", "v1") == "v3" else LegacyDesign(protocol, smoke)
    design.test_seeds = tuple(protocol.get("final", {}).get("test_S", design.settings.S))
    return design


def stage_plan(name):
    stages = (("P0", "pilot"), ("P1", "baseline"), ("P2", "search"), ("CONFIRM", "confirm"),
              ("SELECT", "select"), ("P3", "final"), ("REPORT", "report"))
    return tuple(Stage(phase, operation) for phase, operation in stages if name == "v3" or phase != "CONFIRM")
