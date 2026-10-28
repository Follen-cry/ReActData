"""Small public contracts. Task payloads remain opaque to the engine."""
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, TypedDict


@dataclass(frozen=True)
class RunPaths:
    """Resource locations; runs is the parent of run IDs, protocol is a file."""
    exp: str
    api: str
    data: str
    manifests: str
    runs: str
    protocol: str


class _CheckRequired(TypedDict):
    errors: list[str]
    warnings: list[str]


class CheckResult(_CheckRequired, total=False):
    declared_class: str
    class_label: str
    tested_behaviour: dict
    sim_stats: dict
    plugin_log_tail: str


class _AgentRequired(TypedDict):
    ok: bool


class AgentResult(_AgentRequired, total=False):
    error: str
    session_id: str | None
    wall_s: float
    cost_usd: float | None
    models: list[str]
    num_turns: int


@dataclass(frozen=True)
class PromptBundle:
    notes: list[str]
    full: str
    continuation: str


class PromptBuilder(Protocol):
    def build(self, context, cid, rnd, idx, start, siblings, instructions) -> PromptBundle: ...
    def repair(self, prompt: str, checks: CheckResult, resumed: bool) -> str: ...


class TaskStatistics(Protocol):
    def task_of(self, item_id: str) -> str: ...
    def agg(self, items: Mapping[str, float]) -> float: ...
    def paired(self, a, b) -> dict: ...
    def training_noise(self, replicates) -> dict: ...
    def null_delta(self, a, b) -> dict: ...
    def confirm_compare(self, candidates, baseline, sd_train, m=1) -> dict: ...
    def select(self, candidates, baseline, delta) -> dict: ...


@dataclass(frozen=True)
class ReportContext:
    state: dict
    root: str
    P: dict
    run_id: str
    smoke: bool
    design: str
    S: list[int]
    S_null: list[int]
    top_k: int
    confirm_seeds: list[int]
    formal_seeds: list[int]


@dataclass(frozen=True)
class Stage:
    name: str
    operation: str


@dataclass(frozen=True)
class JobRequest:
    key: str
    candidate_id: str | None
    seed: int
    steps: int
    split: str
    generation_seeds: tuple[int, ...]
    formal: bool = False


@dataclass(frozen=True)
class ExperimentSnapshot:
    """A detached read view; mutating it cannot modify the live run."""
    data: Mapping[str, Any]


class StagePlan(Protocol):
    def stages(self) -> tuple[Stage, ...]: ...


class TaskAdapter(Protocol):
    """Capabilities consumed by assembly and the shared scientific workflow."""
    contract: "CandidateContract"
    statistics: TaskStatistics
    baseline_dir: str
    prompts: PromptBuilder
    evidence_source: "EvidenceRenderer"
    def bind(self, store, protocol_sha: str) -> None: ...
    def build_spec(self, store, request: JobRequest, code: str | None, settings) -> str: ...
    def item_means(self, directory: str, split: str, seeds) -> dict: ...
    def audit(self, directory: str) -> dict: ...
    def audit_summary(self, audit: dict) -> dict: ...
    def cost(self, spec: str) -> dict: ...
    def cost_summary(self, spec: str) -> dict: ...
    def estimate_job_hours(self, cost: dict, settings, design: str) -> float: ...
    def freeze(self, store, selected: str) -> str: ...
    def authorize_job(self, store, spec: dict) -> None: ...
    def pilot(self, store, execute: Callable, settings) -> None: ...
    def provenance(self) -> dict: ...
    def report(self, view: ReportContext) -> None: ...


class EvidenceRenderer(Protocol):
    def search_options(self, snapshot: Mapping, best: str) -> dict: ...
    def collect_evidence(self, request: dict): ...
    def render_evidence(self, authorized) -> str: ...


class CandidateContract(Protocol):
    files: tuple[str, ...]
    def valid_dev_ids(self) -> set[str]: ...
    def program_sha(self, directory: str) -> str: ...
    def check_candidate(self, work: str, scratch: str) -> CheckResult: ...
    def verify_frozen(self, record: Mapping) -> bool: ...
    def frozen_hash(self, directory: str) -> str: ...
    def proposal_errors(self, work: str) -> list[str]: ...
    def candidate_record(self, cid: str, directory: str, **fields: Any) -> dict: ...
    def instructions(self) -> str: ...
    def pool_stats(self) -> dict: ...


class AgentBackend(Protocol):
    def call(self, prompt: str, work: str, inp: str, log_dir: str, **options: Any) -> AgentResult: ...


class JobExecutor(Protocol):
    def execute(self, spec: str) -> str: ...
