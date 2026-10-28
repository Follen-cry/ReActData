"""Small explicit factories. No eval, candidate imports or plugin discovery."""


def resolve(registry, name, **options):
    try:
        factory = registry[name]
    except KeyError:
        raise ValueError(f"unknown implementation {name!r}; choose one of {sorted(registry)}") from None
    return factory(**options)


def search_policy(name, design):
    return resolve(SEARCH_POLICIES, name, design=design)


def agent_backend(name, call):
    return resolve(AGENT_BACKENDS, name, call=call)


def executor(name, **options):
    return resolve(EXECUTORS, name, **options)


def selection_policy(name, design, statistics):
    return resolve(SELECTION_POLICIES, name or ("noise-aware" if design == "v3" else "legacy"), statistics=statistics)


def _reactdata(**options):
    from tasks.reactdata import ReactDataTask
    return ReactDataTask(**options)


TASKS = {"reactdata": _reactdata}

from experiment.improvement.agents import ClaudeBackend
from experiment.runtime.execution import SlotExecutor
from experiment.improvement.policy import DefaultSearchPolicy, LegacySearchPolicy
from experiment.workflow.selection import NoiseAwareSelection, LegacySelection

SEARCH_POLICIES = {"default": lambda design: DefaultSearchPolicy(design), "legacy": lambda design: LegacySearchPolicy()}
SELECTION_POLICIES = {"noise-aware": lambda statistics: NoiseAwareSelection(), "legacy": LegacySelection}
AGENT_BACKENDS = {"claude": ClaudeBackend}
EXECUTORS = {"slots": SlotExecutor}


def task(name, **options):
    return resolve(TASKS, name, **options)
