"""Small operation receipts; ambiguous external calls are never repeated automatically."""
import os
from experiment.runtime.io import read_json, write_json


class OutcomeUnknown(RuntimeError):
    pass


class OperationJournal:
    def __init__(self):
        self.reused = False

    def call(self, directory, action, *, recover=None, retry_safe=False):
        self.reused = False
        path = os.path.join(directory, "operation.json")
        if os.path.isfile(path):
            receipt = read_json(path)
            if receipt["status"] == "completed":
                self.reused = True
                return receipt["result"]
            recovered = recover() if recover else None
            if recovered is not None:
                self.reused = True
                write_json(path, {"status": "completed", "result": recovered})
                return recovered
            if not retry_safe:
                raise OutcomeUnknown(f"external operation outcome is unknown: {directory}; inspect its logs before retrying")
        elif recover:
            recovered = recover()
            if recovered is not None:
                self.reused = True
                write_json(path, {"status": "completed", "result": recovered})
                return recovered
        write_json(path, {"status": "started"})
        result = action()
        write_json(path, {"status": "completed", "result": result})
        return result
