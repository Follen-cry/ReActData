"""Default cluster executor; jobs.py retains the original cluster transport."""
import os
import threading
import time
from typing import Dict
from experiment.runtime.io import read_json, read_jsonl, write_json

class SlotExecutor:
    def __init__(self, store, protocol, resources, transport, authorize=None):
        self.store, self.P, self.resources, self.transport = store, protocol, resources, transport
        self.state, self.lock, self.event = store.state, store.lock, store.event
        self.authorize = authorize
        self.slots = [transport.Slot(s["name"], s["node"], s["gpus"], s["port"]) for s in protocol["budget"]["slots"]]
        from experiment.runtime.slots import SlotPool
        self.pool = SlotPool(self.slots, protocol["budget"].get("max_concurrent_jobs", len(self.slots)), self.lock, transport.gpus_idle)
        self.slot_free = self.pool.capacity

    def _attachment(self, spec, run):
        """New attachment record, or legacy launch event + pid. Never relaunch an active job."""
        path = os.path.join(run["run_dir"], "executor.json")
        if os.path.isfile(path):
            record = read_json(path)
            if record.get("launch_pending") and not os.path.isfile(os.path.join(run["run_dir"], "pid")):
                raise RuntimeError("job launch was interrupted before a pid was recorded; refusing an ambiguous relaunch")
            return record
        if not os.path.isfile(os.path.join(run["run_dir"], "pid")):
            return None
        events = os.path.join(self.store.root, "events.jsonl")
        launches = [e for e in read_jsonl(events) if e.get("kind") == "job_launched" and e.get("job") == run["run_id"]] if os.path.isfile(events) else []
        if not launches:
            raise RuntimeError(f"job {run['run_id']} has a pid but no recoverable slot; refusing to relaunch")
        event = launches[-1]
        slot = next((s for s in self.slots if s.name == event["slot"]), None)
        if slot is None or slot.node != event["node"]:
            raise RuntimeError("legacy job slot differs from the configured slot")
        return {"run_json": spec, "run_dir": run["run_dir"], "slot": slot.to_json(), "start": event["time"],
                "attempt": event.get("attempt", 0)}

    def execute(self, spec: str) -> str:
        """Runs a job to completion on a free slot; returns done | candidate_failed | infra_failed."""
        run = read_json(spec)
        if self.authorize:
            self.authorize(run)
        elif any(ev.get("name") == "test" for ev in run.get("evals", [])):
            raise RuntimeError("test execution requires an explicit frozen-program authorization")
        d = run["run_dir"]
        if os.path.exists(os.path.join(d, "DONE.json")):
            return "done"
        retries = self.P["search"]["train_infra_retry"]
        attachment = self._attachment(spec, run)
        attempt = attachment.get("attempt", 0) if attachment else 0
        if attachment and attachment.get("retry_pending"):
            attachment = None
        while True:
            self.resources._disk_safe(before_launch=True)
            with self.pool.lease(attachment["slot"] if attachment else None) as slot:
                if attachment:
                    job, attachment = attachment, None
                    self.event("job_attached", job=run["run_id"], slot=slot.name, node=slot.node, attempt=attempt)
                else:
                    # Write intent before launching so a crash after the launch can recover the node.
                    job = {"run_json": spec, "run_dir": d, "slot": slot.to_json(), "start": time.time(), "attempt": attempt,
                           "launch_pending": True}
                    write_json(os.path.join(d, "executor.json"), job)
                    launched = self.transport.launch(slot, spec)
                    job.update(launched)
                    job["launch_pending"] = False
                    write_json(os.path.join(d, "executor.json"), job)
                    self.event("job_launched", job=run["run_id"], slot=slot.name, node=slot.node, attempt=attempt)
                while self.transport.status(job) == "running":
                    time.sleep(30)
                st = self.transport.status(job)
            self.event("job_finished", job=run["run_id"], status=st, wall_s=round(time.time() - job["start"]))
            if st == "done":                       # optimizer state is only needed for resuming an unfinished job
                for ck in os.listdir(os.path.join(d, "checkpoints")) if os.path.isdir(os.path.join(d, "checkpoints")) else []:
                    p = os.path.join(d, "checkpoints", ck, "optimizer.pt")
                    if os.path.exists(p):
                        os.remove(p)
            if st != "infra_failed" or attempt >= retries:
                if st == "infra_failed":
                    self.store.update_state(failures=[{"job": run["run_id"], "kind": "infra", "attempts": attempt + 1,
                                                   "log_tail": open(os.path.join(d, "train.log")).read()[-3000:]}])
                self.resources._disk_safe(protect=(d,))
                return st
            attempt += 1
            self.store.update_state(failures=[{"job": run["run_id"], "kind": "infra", "attempt": attempt, "action": "resume",
                                           "log_tail": open(os.path.join(d, "train.log")).read()[-1500:]}])
            if os.path.exists(os.path.join(d, "exit_code")):
                os.remove(os.path.join(d, "exit_code"))
            # Persist that the next retry has been authorized; a restart cannot reset its budget.
            write_json(os.path.join(d, "executor.json"), {**job, "attempt": attempt, "retry_pending": True})
            # The current invocation will launch that retry; recovery handles the marker below.


    @staticmethod
    def job_cost( spec: str) -> Dict:
        p = os.path.join(read_json(spec)["run_dir"], "cost.json")
        return read_json(p) if os.path.exists(p) else {}


class FakeExecutor:
    """CPU transport: an explicit trusted function writes the task-specific result."""
    def __init__(self, evaluate):
        self.evaluate = evaluate
        self.executed = []

    def execute(self, spec):
        run = read_json(spec)
        if os.path.isfile(os.path.join(run["run_dir"], "DONE.json")):
            return "done"
        result = self.evaluate(run)
        self.executed.append(run["run_id"])
        if result == "done":
            write_json(os.path.join(run["run_dir"], "DONE.json"), {})
        return result
