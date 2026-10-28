"""Execution permission is separate from Agent evidence disclosure."""


class EvaluationAccessPolicy:
    def require_test(self, state, verify_frozen):
        if state.get("phase") not in ("P3", "REPORT", "DONE"):
            raise RuntimeError("test set requested before FROZEN")
        frozen = state.get("frozen")
        if not frozen or not verify_frozen(frozen):
            raise RuntimeError("test requires the recorded frozen program with its original hash")
