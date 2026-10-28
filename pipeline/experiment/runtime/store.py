"""Filesystem storage, legacy JSON formats, backups and research artifact index."""
import json
import os
import shutil
import threading
import time
import copy
from types import MappingProxyType
from experiment.runtime.io import read_json, write_json
from experiment.runtime.io import sha256_json

BACKUP_EXT = (".json", ".md", ".diff", ".yaml", ".py", ".txt")

class RunStore:
    def __init__(self, root, state=None, backup_root=None):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.state_path = os.path.join(self.root, "state.json")
        self.state = state if state is not None else read_json(self.state_path)
        self.backup_root = backup_root
        self.lock = threading.RLock()

    def snapshot(self):
        from experiment.contracts import ExperimentSnapshot
        def freeze(value):
            if isinstance(value, dict):
                return MappingProxyType({k: freeze(v) for k, v in value.items()})
            if isinstance(value, list):
                return tuple(freeze(v) for v in value)
            return value
        with self.lock:
            return ExperimentSnapshot(freeze(copy.deepcopy(self.state)))

    def update_state(self, *, updates=None, candidates=None, failures=(), persist=True):
        with self.lock:
            self.state.update(updates or {})
            for cid, values in (candidates or {}).items():
                self.state.setdefault("candidates", {}).setdefault(cid, {}).update(values)
            self.state.setdefault("failures", []).extend(failures)
            if persist:
                self.save()

    def commit(self, kind, *, updates=None, candidates=None, failures=(), event_data=None, **event):
        """Serialize a state transition and its event under the same writer lock."""
        with self.lock:
            self.update_state(updates=updates, candidates=candidates, failures=failures, persist=False)
            self.event(kind, **(event_data or {}), **event)

    def begin_phase(self, phase):
        self.commit("phase_start", updates={"phase": phase})

    def complete_phase(self, phase):
        with self.lock:
            self.state["done_upto"] = phase
            self.save()
            self.backup()

    def save(self) -> None:
        with self.lock:
            write_json(self.state_path, self.state)
            self.write_research_index()

    def job_spec(self, name, payload):
        path = os.path.join(self.root, "jobs", name, "run.json")
        if not os.path.exists(path):
            write_json(path, payload)
        return path

    def candidate_record(self, cid, record):
        write_json(os.path.join(self.root, "candidates", cid, "candidate.json"), record)

    def record_implementation(self, files):
        """Keep original run.json provenance, and record code used on later resumes."""
        path = os.path.join(self.root, "implementations.json")
        history = read_json(path) if os.path.isfile(path) else []
        digest = sha256_json(files)
        if not history or history[-1]["sha256"] != digest:
            history.append({"time": time.time(), "sha256": digest, "files": files})
            write_json(path, history)


    def event(self, kind: str, **kw) -> None:
        rec = {"time": time.time(), "phase": self.state["phase"], "kind": kind, **kw}
        with self.lock:
            with open(os.path.join(self.root, "events.jsonl"), "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {self.state['phase']} {kind} " + json.dumps(kw, ensure_ascii=False)[:400], flush=True)
        self.save()
        self.backup()


    def backup(self):
        from experiment.runtime.projections import backup_records
        backup_records(self.root, self.backup_root, self.lock)

    def write_research_index(self):
        from experiment.runtime.projections import build_research_index
        write_json(os.path.join(self.root, "research_index.json"), build_research_index(self.root, self.state))
