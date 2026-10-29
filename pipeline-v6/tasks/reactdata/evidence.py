"""Analogy-grid evidence renderer capability; no access to the experiment runtime."""
import os
import re
from experiment.runtime.io import read_json


class ReactDataEvidence:
    def __init__(self, render, dev_manifest=None, *, paths=None, collect=None):
        self.paths, self.collect = paths, collect
        self.render, self.dev_manifest = render, dev_manifest

    @staticmethod
    def search_options(snapshot, best):
        """Reactdata presentation rules; candidate lifecycle need not know about images."""
        trained = sorted(cid for cid, c in snapshot["candidates"].items() if c["status"] == "trained")
        return {"gallery_ids": {"c000", best} | set(trained[-4:]),
                "baseline_items": tuple(dict(snapshot[key]) if key in snapshot else None
                                        for key in ("baseline_mean_items", "baseline_sd_items"))}

    def _validate(self, request):
        visibility = request["visibility"]
        if visibility.allows("instructions"):
            name = request.get("instructions", "BUILDER_AGENT.md")
            if not re.fullmatch(r"BUILDER_AGENT(?:_[A-Za-z0-9-]+)?\.md", name):
                raise PermissionError("only task instruction resources may enter the input bundle")
        if self.dev_manifest:
            for entry in request["entries"]:
                if entry.get("run_dir"):
                    spec = read_json(os.path.join(entry["run_dir"], "run.json"))
                    for ev in spec.get("evals", []):
                        if os.path.realpath(ev["file"]) != os.path.realpath(self.dev_manifest):
                            raise PermissionError("reactdata evidence must come from the frozen dev manifest")

    def collect_evidence(self, request):
        self._validate(request)
        from runner.report import collect_report
        return (self.collect or collect_report)(**request, paths=self.paths)

    def render_evidence(self, authorized):
        from experiment.evidence.authorized import render_authorized
        return render_authorized(authorized)

    def build_evidence(self, request):
        """Legacy combined entry; normal assembly uses the two explicit operations."""
        self._validate(request)
        return self.render(**request, **({"paths": self.paths} if self.paths else {}))
