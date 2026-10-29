"""Disclosure gates and replayable input requests; full records stay in RunStore."""
import os
from experiment.runtime.io import read_json, write_json


class VisibilityPolicy:
    # Data capabilities, not a denylist of filenames copied after the fact.
    sections = frozenset({"api", "instructions", "train_examples", "results", "dev_items", "analysis_traces", "failures", "candidates"})
    entry_fields = frozenset({"id", "role", "status", "candidate_dir", "run_dir", "seeds", "comparison", "cost",
                              "checks_summary", "audit", "error"})

    def __init__(self, sections=None):
        if sections is not None:
            unknown = set(sections) - self.sections
            if unknown:
                raise ValueError(f"unsupported disclosure capabilities: {sorted(unknown)}")
            self.sections = frozenset(sections)

    def allows(self, section):
        return section in self.sections

    def project(self, request, role="builder", phase="P2"):
        if role != "builder" or phase != "P2":
            raise PermissionError("search evidence is only disclosed to the builder during P2")
        out = dict(request)
        entries = []
        for entry in request["entries"]:
            if entry.get("split", "dev") != "dev":
                raise PermissionError("test evidence is never disclosed to the search agent")
            run_dir = entry.get("run_dir")
            if run_dir:
                spec = os.path.join(run_dir, "run.json")
                if os.path.isfile(spec):
                    evs = read_json(spec).get("evals", [])
                    if any(e["name"] != "dev" for e in evs):
                        raise PermissionError("only dev jobs may supply search evidence")
            entries.append({k: v for k, v in entry.items() if k in self.entry_fields})
        out["entries"] = entries
        return out


class EvidenceService:
    def __init__(self, task, visibility):
        self.task, self.visibility = task, visibility

    def search_options(self, snapshot, best):
        return self.task.search_options(snapshot, best)

    def build(self, round_dir, entries, pool_stats, notes, **options):
        request = {"round_dir": round_dir, "entries": entries, "pool_stats": pool_stats, "notes": notes, **options}
        # JSON-normalize sets/tuples before persisting so rebuild follows the same path.
        if request.get("gallery_ids") is not None:
            request["gallery_ids"] = sorted(request["gallery_ids"])
        request = self.visibility.project(request)
        write_json(os.path.join(round_dir, "evidence_request.json"), {"version": 1, "role": "builder", "phase": "P2",
                   "sections": sorted(self.visibility.sections), "request": request})
        return self._render(request)

    def _render(self, request, visibility=None):
        request = {**request, "visibility": visibility or self.visibility}
        if hasattr(self.task, "collect_evidence"):
            authorized = self.task.collect_evidence(request)
            return self.task.render_evidence(authorized)
        # Compatibility for existing renderer plugins; new adapters expose the two operations.
        return self.task.build_evidence(request)

    def rebuild(self, request_file):
        saved = read_json(request_file)
        if saved["version"] != 1:
            raise ValueError("unsupported evidence request version")
        effective = VisibilityPolicy(self.visibility.sections & set(saved.get("sections", VisibilityPolicy.sections)))
        return self._render(effective.project(saved["request"], saved["role"], saved["phase"]), effective)
