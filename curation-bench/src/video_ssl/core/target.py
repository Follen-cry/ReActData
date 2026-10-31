"""Queue records adapted from curation-bench 24eea15 tools/suite.py."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Target:
    task_id: str
    iteration: int
    state: str = 'pending'
    run_dir: Path | None = None
    task_path: Path | None = None
    eval_result: dict | None = None

    @property
    def key(self):
        return f'{self.task_id}_iter{self.iteration}'

    def to_dict(self):
        return dict(task_id=self.task_id, iteration=self.iteration, state=self.state,
                    key=self.key, run_dir=str(self.run_dir) if self.run_dir else None,
                    task_path=str(self.task_path) if self.task_path else None,
                    eval_result=self.eval_result)


def build_target_queue(task_ids, iterations):
    return [Target(task_id, iteration) for task_id in task_ids
            for iteration in range(1, iterations + 1)]
