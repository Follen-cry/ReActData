"""The application authorization boundary, independent of execution backend."""
from experiment.runtime.io import read_json


class AuthorizedExecutor:
    def __init__(self, execute, authorize):
        self._execute, self._authorize = execute, authorize

    def execute(self, spec: str) -> str:
        # Check even completed jobs and overrides before delegating any operation.
        self._authorize(read_json(spec))
        return self._execute(spec)
