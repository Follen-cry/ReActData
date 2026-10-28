"""reactdata-v1 Runner (copied from reactdata-v0): the fixed state machine of one experiment (no LLM decides anything here).

v1 changes: two-task dev (equal-weight aggregation, stratified bootstrap), 100-step training, one Builder Agent session
continued across all candidate calls of a run (claude -p --resume; persistent Runner-only agent HOME), checks render
every recipe's text.

    python runner/state_machine.py --run-id r1 [--smoke]

Phases: P0 (pilot / smoke job) -> P1 (fixed baseline on dev with S + S_null; Base on dev) -> P2 (2 rounds x 2 agent
candidates: agent -> checks (+1 repair) -> train + dev -> audit) -> SELECT -> FROZEN -> P3 (formal retraining of the
selected program and the fixed baseline, test for Base / baseline / selected) -> REPORT -> DONE.
Every step is idempotent: rerunning the same command resumes (finished jobs are skipped, running jobs re-attached,
infrastructure failures resumed once from their last checkpoint). --smoke runs the same chain with 4-step training,
4 eval rows and 1 candidate, in a separate run id, to test the machinery end to end.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import API, DATA, EXP, MANIFESTS, RUNS, read_json, read_jsonl, sha256_file, sha256_json, sha256_tree, write_json  # noqa: E402
sys.path.insert(0, API)

import jobs  # noqa: E402
from agent_call import call_agent  # noqa: E402
from audit import audit_run  # noqa: E402
from checks import run_checks  # noqa: E402
from report import write_report  # noqa: E402
import stats  # noqa: E402

PROTO = os.path.join(EXP, "protocol.yaml")
CAND_FILES = ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml", "proposal.md", "memory.md")
BACKUP_EXT = (".json", ".md", ".diff", ".yaml", ".py", ".txt")


class Runner:
    def __init__(self, run_id: str, smoke: bool, replay_src: Optional[str] = None, protocol: str = PROTO,
                 continue_from: Optional[str] = None) -> None:
        self.run_id, self.smoke, self.replay_src = run_id, smoke, replay_src
        self.proto_path = os.path.abspath(protocol)
        self.P = yaml.safe_load(open(self.proto_path))
        self.root = os.path.join(RUNS, run_id)
        os.makedirs(self.root, exist_ok=True)
        self.state_path = os.path.join(self.root, "state.json")
        self.lock = threading.Lock()
        self.disk_lock = threading.Lock()          # manage_disk is called from several job threads
        self.state = read_json(self.state_path) if os.path.exists(self.state_path) else {
            "run_id": run_id, "smoke": smoke, "phase": "INIT", "candidates": {}, "rejects_in_a_row": 0, "failures": []}
        self.slots = [jobs.Slot(s["name"], s["node"], s["gpus"], s["port"]) for s in self.P["budget"]["slots"]]
        self.slot_free = threading.Semaphore(self.P["budget"].get("max_concurrent_jobs", len(self.slots)))
        self.design = self.P.get("design", "v1")
        self.S = self.P["evaluation"]["S"][:1] if smoke else self.P["evaluation"]["S"]
        s_null = self.P["evaluation"].get("S_null", [])
        self.S_null = s_null[:1] if smoke else s_null
        if self.design == "v3":                          # smoke: 2 replicates, 2 confirm seeds, top-1, 1 formal seed
            self.base_seeds = self.P["noise"]["baseline_seeds"][:2] if smoke else self.P["noise"]["baseline_seeds"]
            self.confirm_seeds = self.P["confirm"]["seeds"][:1] if smoke else self.P["confirm"]["seeds"]
            self.top_k = 1 if smoke else self.P["confirm"]["top_k"]
            self.formal_seeds = self.P["final"]["formal_seeds"][:1] if smoke else self.P["final"]["formal_seeds"]
        self.steps = 4 if smoke else self.P["training"]["max_steps"]
        self.max_rows = 4 if smoke else None
        self.n_rounds = 2 if smoke else self.P["search"]["rounds"]      # smoke: 2 x 1 candidate, so the --resume chain is exercised
        self.per_round = 1 if smoke else self.P["search"]["candidates_per_round"]
        self.proto_sha = sha256_file(self.proto_path)
        if "run_json" not in self.state:
            self.state["run_json"] = {"protocol": os.path.relpath(self.proto_path, EXP), "protocol_sha256": self.proto_sha, "runner_sha256": sha256_tree(
                [os.path.join(HERE, f) for f in os.listdir(HERE) if f.endswith(".py")]),
                "api_sha256": sha256_tree([os.path.join(API, f) for f in os.listdir(API) if f.endswith((".py", ".md"))]),
                "builder_agent_md_sha256": sha256_file(os.path.join(EXP, self.P["search"]["agent"].get("instructions", "BUILDER_AGENT.md"))),
                "data_build": read_json(os.path.join(DATA, "BUILD_DONE.json")),
                "scorer": __import__("score").scorer_version()}
            write_json(os.path.join(self.root, "run.json"), self.state["run_json"])
        elif self.state["run_json"]["protocol_sha256"] != self.proto_sha:
            raise SystemExit("the protocol changed since this run started: start a new run id (new protocol version)")
        if continue_from and "inherited_from" not in self.state:
            self.inherit(continue_from)
        elif continue_from and self.state["inherited_from"]["run"] != continue_from:
            raise SystemExit(f"this run already continues {self.state['inherited_from']['run']}")

    # ------------------------------------------------------------------ cross-run continuation
    def inherit(self, src_id: str) -> None:
        """Start this run as a continuation of a finished run: candidates, baselines, delta and the agent session.

        Source records are referenced, never modified: candidate code and job directories stay in the source run, the
        agent HOME (session files) is copied, and the source's base_test / formal_c000 jobs are linked for reuse."""
        src_root = os.path.join(RUNS, src_id)
        src = read_json(os.path.join(src_root, "state.json"))
        if src.get("phase") != "DONE":
            raise SystemExit(f"{src_id} is not finished (phase {src.get('phase')})")
        if src.get("smoke") != self.smoke:
            raise SystemExit("smoke and non-smoke runs cannot continue each other")
        import copy
        cands = copy.deepcopy(src["candidates"])
        for c in cands.values():
            c["inherited_from"] = src_id
        self.state["candidates"] = cands
        for k in ("base_dev", "delta", "estimates", "first_step_check", "agent_session", "last_memory"):
            if k in src:
                self.state[k] = copy.deepcopy(src[k])
        rounds = [c.get("round", 0) or 0 for c in cands.values()]
        idx = [int(k[1:]) for k in cands]
        self.state["round_offset"] = max(rounds) if rounds else 0
        self.state["cand_offset"] = max(idx) if idx else 0
        if os.path.isdir(os.path.join(src_root, "agent_home")):
            shutil.copytree(os.path.join(src_root, "agent_home"), os.path.join(self.root, "agent_home"), symlinks=True)
        linked = []
        for name in os.listdir(os.path.join(src_root, "jobs")):
            if (name == "base_test" or name.startswith("formal_c000-")) and os.path.exists(os.path.join(src_root, "jobs", name, "DONE.json")):
                os.makedirs(os.path.join(self.root, "jobs"), exist_ok=True)
                dst = os.path.join(self.root, "jobs", name)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(src_root, "jobs", name), dst)
                linked.append(name)
        self.state["inherited_from"] = {"run": src_id, "state_sha256": sha256_file(os.path.join(src_root, "state.json")),
                                        "protocol_sha256": src["run_json"]["protocol_sha256"], "runner_sha256": src["run_json"]["runner_sha256"],
                                        "candidates": sorted(cands), "agent_session": src.get("agent_session"), "linked_jobs": linked,
                                        "rounds_done": self.state["round_offset"]}
        self.state["done_upto"] = "P1"          # P0 / P1 results are inherited
        self.state["budget_start"] = time.time()
        self.state["phase"] = "INHERIT"
        self.event("continued_from", **self.state["inherited_from"])

    # ------------------------------------------------------------------ bookkeeping
    def save(self) -> None:
        with self.lock:
            write_json(self.state_path, self.state)

    def event(self, kind: str, **kw) -> None:
        rec = {"time": time.time(), "phase": self.state["phase"], "kind": kind, **kw}
        with self.lock:
            with open(os.path.join(self.root, "events.jsonl"), "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {self.state['phase']} {kind} " + json.dumps(kw, ensure_ascii=False)[:400], flush=True)
        self.save()
        self.backup()

    def backup(self) -> None:
        """Small records -> <exp>/records/<run_id>/ (git); checkpoints / images are referenced, not copied."""
        dst_root = os.path.join(EXP, "records", self.run_id)
        with self.lock:
            for dp, dn, fn in os.walk(self.root):
                dn[:] = [d for d in dn if d not in ("checkpoints", "images", "plugin_scratch", "probes", "input", "work_start",
                                                    "tensorboard", "agent_home") and not d.startswith("agent_home_") and not d.endswith("_checkwork")]
                for f in fn:
                    if not f.endswith(BACKUP_EXT) or f in ("rows_realized.jsonl", "per_row.jsonl", "preds.jsonl"):
                        continue
                    src = os.path.join(dp, f)
                    if os.path.getsize(src) > 2_000_000:
                        continue
                    dst = os.path.join(dst_root, os.path.relpath(src, self.root))
                    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)

    def elapsed_h(self) -> float:
        return (time.time() - self.state["budget_start"]) / 3600 if "budget_start" in self.state else 0.0

    def budget_left_h(self) -> float:
        return self.P["budget"]["wall_hours"] - self.elapsed_h()

    def search_allowed(self, est_h: float) -> bool:
        b = self.P["budget"]
        return self.elapsed_h() + est_h <= b["wall_hours"] - b["reserve_final_hours"] - b["buffer_hours"]

    # ------------------------------------------------------------------ jobs
    def run_spec(self, name: str, cand_dir: Optional[str], seed: int, steps: int, evals: List[Dict]) -> str:
        d = os.path.join(self.root, "jobs", name)
        os.makedirs(d, exist_ok=True)
        spec = os.path.join(d, "run.json")
        if not os.path.exists(spec):
            write_json(spec, {"run_id": name, "run_dir": d, "candidate_dir": cand_dir, "run_seed": seed, "steps": steps,
                              "evals": evals, "max_eval_rows": self.max_rows, "protocol_sha256": self.proto_sha,
                              "overrides": ([f"checkpoint.every_steps={self.P['training']['checkpoint_every']}"]
                                            if self.P["training"].get("checkpoint_every") else []),
                              "gpus": 4})
        return spec

    def evals(self, split: str, seeds: List[int]) -> List[Dict]:
        if split == "test" and self.state["phase"] not in ("P3", "REPORT", "DONE"):
            raise RuntimeError("test set requested before FROZEN")
        if split == "test" and self.smoke:      # the smoke run never touches the real test set
            return [{"name": "test", "file": os.path.join(MANIFESTS, "dev.jsonl"), "seeds": seeds, "smoke_standin": True}]
        return [{"name": split, "file": os.path.join(MANIFESTS, f"{split}.jsonl"), "seeds": seeds}]

    def execute(self, spec: str) -> str:
        """Runs a job to completion on a free slot; returns done | candidate_failed | infra_failed."""
        run = read_json(spec)
        d = run["run_dir"]
        if os.path.exists(os.path.join(d, "DONE.json")):
            return "done"
        retries = self.P["search"]["train_infra_retry"]
        attempt = 0
        while True:
            self._disk_safe(before_launch=True)
            self.slot_free.acquire()
            try:
                slot = None
                while slot is None:
                    for s in self.slots:
                        with self.lock:            # reserve under the lock, check the GPUs outside it (ssh can be slow)
                            if s.job is not None:
                                continue
                            s.job = {"reserved": True}
                        if jobs.gpus_idle(s.node, s.gpus):
                            slot = s
                            break
                        with self.lock:
                            s.job = None
                    if slot is None:
                        time.sleep(60)
                job = jobs.launch(slot, spec)
                self.event("job_launched", job=run["run_id"], slot=slot.name, node=slot.node, attempt=attempt)
                while jobs.status(job) == "running":
                    time.sleep(30)
                st = jobs.status(job)
            finally:
                with self.lock:
                    if slot is not None:
                        slot.job = None
                self.slot_free.release()
            self.event("job_finished", job=run["run_id"], status=st, wall_s=round(time.time() - job["start"]))
            if st == "done":                       # optimizer state is only needed for resuming an unfinished job
                for ck in os.listdir(os.path.join(d, "checkpoints")) if os.path.isdir(os.path.join(d, "checkpoints")) else []:
                    p = os.path.join(d, "checkpoints", ck, "optimizer.pt")
                    if os.path.exists(p):
                        os.remove(p)
            if st != "infra_failed" or attempt >= retries:
                if st == "infra_failed":
                    self.state["failures"].append({"job": run["run_id"], "kind": "infra", "attempts": attempt + 1,
                                                   "log_tail": open(os.path.join(d, "train.log")).read()[-3000:]})
                self._disk_safe(protect=(d,))
                return st
            attempt += 1
            self.state["failures"].append({"job": run["run_id"], "kind": "infra", "attempt": attempt, "action": "resume",
                                           "log_tail": open(os.path.join(d, "train.log")).read()[-1500:]})
            if os.path.exists(os.path.join(d, "exit_code")):
                os.remove(os.path.join(d, "exit_code"))

    # ------------------------------------------------------------------ disk budget
    PEAK_JOB_GB = 25.0      # one running job: trainable (3.6 GB) + optimizer (7.5 GB), twice while a checkpoint is replaced

    def disk_gb(self) -> float:
        tot = 0
        for dp, dn, fn in os.walk(self.root):            # symlinks (inherited jobs) are not followed / not counted
            for f in fn:
                p = os.path.join(dp, f)
                if not os.path.islink(p):
                    try:
                        tot += os.lstat(p).st_size
                    except OSError:
                        pass
        return tot / 1e9

    def _disk_safe(self, **kw) -> None:
        try:
            self.manage_disk(**kw)
        except Exception as e:  # noqa: BLE001  disk housekeeping must not fail a job
            self.state["failures"].append({"kind": "disk_manage_error", "detail": str(e)[:300]})

    def manage_disk(self, before_launch: bool = False, protect=()) -> None:
        """Keep the final trainable weights of the current best candidate's screening run and of every formal (P3) run;
        delete the checkpoints of every other job that is final (DONE.json or FAILED.json). Running jobs, jobs waiting
        for an infrastructure retry and the dirs in ``protect`` (a job whose result is not recorded yet) are never
        touched. Called after every job (search, confirmation, P3) and before every launch; enforced against
        budget.disk_gb when the protocol sets it."""
        with self.disk_lock:
            self._manage_disk(before_launch, {os.path.abspath(x) for x in protect})

    def _manage_disk(self, before_launch: bool, protect: set) -> None:
        limit = self.P["budget"].get("disk_gb")
        jobs_dir = os.path.join(self.root, "jobs")
        if not limit or not os.path.isdir(jobs_dir):
            return
        best = self.state["candidates"].get(self.best_so_far(), {}).get("run_dir")
        freed = 0.0

        def prune(protect_best: bool):
            nonlocal freed
            for name in sorted(os.listdir(jobs_dir)):
                d = os.path.join(jobs_dir, name)
                ck = os.path.join(d, "checkpoints")
                final = os.path.exists(os.path.join(d, "DONE.json")) or os.path.exists(os.path.join(d, "FAILED.json"))
                if os.path.islink(d) or not os.path.isdir(ck) or not final or os.path.abspath(d) in protect:
                    continue                              # inherited, no checkpoints, running / retry pending, or protected
                steps = sorted(x for x in os.listdir(ck) if x.startswith("step_") and not x.endswith(".tmp"))
                keep = protect_best and (os.path.abspath(d) == os.path.abspath(best or "") or name.startswith("formal_"))
                for st in steps:
                    full = os.path.join(ck, st)
                    if keep and st == steps[-1]:
                        opt = os.path.join(full, "optimizer.pt")
                        if os.path.exists(opt):
                            freed += os.path.getsize(opt) / 1e9
                            os.remove(opt)
                        continue
                    freed += sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(full) for f in fs) / 1e9
                    shutil.rmtree(full, ignore_errors=True)

        prune(protect_best=True)
        used = self.disk_gb()
        need = used + (self.PEAK_JOB_GB if before_launch else 0.0)
        if need > limit:                                  # last resort: drop the best candidate's weights too
            prune(protect_best=False)
            used = self.disk_gb()
            self.state["failures"].append({"kind": "disk", "detail": f"over budget before launch: {need:.1f} GB > {limit} GB; "
                                           f"best candidate's weights deleted, now {used:.1f} GB"})
        self.state["disk_gb"] = round(used, 2)
        if freed > 0.05:
            self.event("disk_pruned", freed_gb=round(freed, 2), used_gb=round(used, 2), limit_gb=limit)

    def job_cost(self, spec: str) -> Dict:
        p = os.path.join(read_json(spec)["run_dir"], "cost.json")
        return read_json(p) if os.path.exists(p) else {}

    # ------------------------------------------------------------------ candidates
    def candidate_record(self, cid: str, cdir: str, **kw) -> Dict:
        files = [os.path.join(cdir, f) for f in ("databuilder.py", "analysis.py") if os.path.exists(os.path.join(cdir, f))]
        rec = {"cand_id": cid, "code_sha256": sha256_tree(files) if files else None,
               "config_sha256": sha256_file(os.path.join(cdir, "candidate.yaml")) if os.path.exists(os.path.join(cdir, "candidate.yaml")) else None,
               "templates_sha256": sha256_file(os.path.join(cdir, "text_templates.yaml")) if os.path.exists(os.path.join(cdir, "text_templates.yaml")) else None,
               "protocol_sha256": self.proto_sha, **kw}
        write_json(os.path.join(self.root, "candidates", cid, "candidate.json"), rec)
        return rec

    def program_sha(self, d: str) -> str:
        files = ("databuilder.py", "analysis.py", "text_templates.yaml") if self.P["search"].get("config_only_is_duplicate") \
            else ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml")
        return sha256_tree([os.path.join(d, f) for f in files if os.path.exists(os.path.join(d, f))])

    def trained_entries(self) -> List[Dict]:
        out = []
        for cid, c in sorted(self.state["candidates"].items()):
            e = {"id": cid, "role": c["role"], "status": c["status"], "candidate_dir": c.get("code_dir"),
                 "run_dir": c.get("run_dir"), "seeds": self.S, "cost": c.get("cost_summary"), "audit": c.get("audit"),
                 "comparison": c.get("comparison"), "error": c.get("error")}
            out.append(e)
        return out

    def proposal_errors(self, work: str) -> List[str]:
        """protocol search.proposal: required sections in proposal.md and >= min_item_refs real dev item ids cited
        (a mechanical sign that the per-item results were read; it cannot prove the reasoning)."""
        req = self.P["search"].get("proposal")
        if not req:
            return []
        import re
        p = os.path.join(work, "proposal.md")
        text = open(p).read() if os.path.exists(p) else ""
        errs = [f"proposal.md: missing section '## {h}'" for h in req["sections"] if not re.search(rf"^##\s+{re.escape(h)}", text, re.M | re.I)]
        dev_ids = {r["id"] for r in read_jsonl(os.path.join(MANIFESTS, "dev.jsonl"))}
        cited = set(re.findall(r"dev_(?:scale|move)_\d{3}", text)) & dev_ids
        if len(cited) < req.get("min_item_refs", 0):
            errs.append(f"proposal.md: cites {len(cited)} dev item ids from /input/dev_items/ (at least {req['min_item_refs']} required "
                        "in the Evidence section, e.g. dev_scale_012)")
        return errs

    def workspace_start(self) -> str:
        """protocol search.workspace_start: best_dev (v1 default) or latest_valid (the agent's most recent valid program)."""
        if self.P["search"].get("workspace_start") == "latest_valid":
            mine = sorted(k for k, c in self.state["candidates"].items() if k != "c000" and c["status"] in ("trained", "checked")
                          and not c.get("inherited_from"))
            return mine[-1] if mine else (self.best_so_far() if self.state.get("inherited_from") else "c000")
        return self.best_so_far()

    def gallery_ids(self) -> set:
        """Failure-case images only for the baseline, the current best and the 4 most recent trained candidates."""
        trained = sorted(k for k, c in self.state["candidates"].items() if c["status"] == "trained")
        return {"c000", self.best_so_far()} | set(trained[-4:])

    def best_so_far(self) -> str:
        ok = [(c["dev_mean"], cid) for cid, c in self.state["candidates"].items() if c["status"] == "trained" and "dev_mean" in c]
        return max(ok)[1] if ok else "c000"

    def summarize_job(self, cid: str, spec: str) -> None:
        c = self.state["candidates"][cid]
        run_dir = read_json(spec)["run_dir"]
        c["run_dir"] = run_dir
        cost = self.job_cost(spec)
        c["cost_summary"] = {k: cost.get(k) for k in ("rows", "unique_clips", "repeated_rows", "max_clip_uses", "vit_images", "vit_tokens", "seq_tokens",
                                                      "train_gpu_s", "probe_gpu_s", "eval_gpu_s", "plugin_wall_s")}
        items = stats.item_means(run_dir, "dev", self.S)
        c["dev_mean"] = stats.agg(items)
        c["dev_task_means"] = {t: float(sum(v for k, v in items.items() if stats.task_of(k) == t) /
                                        sum(1 for k in items if stats.task_of(k) == t)) for t in sorted({stats.task_of(k) for k in items})}
        if cid != "c000":
            ref = self.P["search"].get("screen_reference", "screen_seed_run")
            if ref == "replicate_mean" and self.state.get("baseline_mean_items"):
                base = self.state["baseline_mean_items"]          # per-item mean over all baseline replicates
            else:
                base = stats.item_means(self.state["candidates"]["c000"]["run_dir"], "dev", self.S)
            c["comparison"] = stats.paired(items, base)
            c["comparison"]["reference"] = ref
            if self.state.get("noise"):
                c["comparison"]["D_over_sd_train"] = c["comparison"]["D"] / max(1e-9, self.state["noise"]["sd_train"])

    def pool_stats(self) -> Dict:
        pv = read_json(os.path.join(DATA, "pool_view.json"))
        return {"example_clips": [pv["clips"][i]["clip_id"] for i in (0, 1, 2)],
                "summary": {"task": pv["task"], "n_clips": len(pv["clips"]), "frames_per_clip": pv["n_frames"], "fps": pv["fps"],
                            "inner_probe_clips": len(read_jsonl(os.path.join(MANIFESTS, "inner_probe.jsonl"))),
                            "pixel_statistics": read_json(os.path.join(MANIFESTS, "frame_stats_spec.json")),
                            "dev": "50 shape_color_then_scale (same task, new instances) + 50 shape_color_then_move (related task with a "
                                   "different second rule, not in the training pool: cross-task development feedback); dev score = "
                                   "equal-weight mean of the two task means"}}

    def make_candidate(self, cid: str, rnd: int, idx: int, siblings: List[str]) -> bool:
        """Agent session(s) + checks. Returns True if a valid candidate exists."""
        c = self.state["candidates"].setdefault(cid, {"role": "agent", "round": rnd, "status": "new"})
        if c["status"] in ("checked", "trained", "rejected", "failed", "duplicate"):
            return c["status"] in ("checked", "trained")
        rdir = os.path.join(self.root, "rounds", f"{rnd:03d}", cid)
        work = os.path.join(rdir, "work")
        start = self.workspace_start()
        if not os.path.exists(work):
            src = self.state["candidates"][start]["code_dir"]
            os.makedirs(work)
            for f in CAND_FILES:
                if os.path.exists(os.path.join(src, f)):
                    shutil.copy2(os.path.join(src, f), os.path.join(work, f))
            last_mem = self.state.get("last_memory")
            if last_mem and os.path.exists(last_mem):
                shutil.copy2(last_mem, os.path.join(work, "memory.md"))
            shutil.copytree(work, os.path.join(rdir, "work_start"))
        b = getattr(self, "budget_info", None) or {}
        budget_line = (f"Search budget: this is new candidate {b['new_index']} of {b['new_total']} in this run ({b['left_after_this']} left after "
                       f"this one); round {b['round']} of {b['last_round']} ({b['rounds_left_after_this']} rounds left after this one); about "
                       f"{b['wall_h_left']} h of wall time left. The search does not stop early when scores plateau.") if b else ""
        notes = [f"You are writing candidate {cid} (round {rnd}, {idx + 1} of {self.per_round} this round). Your /work starts "
                 f"as a copy of {start} (the best valid program by dev mean so far).", budget_line,
                 f"Training budget per run: {self.steps} optimizer steps x 16 rows from the common initialization; seed {self.P['training'].get('screen_seed', self.P['training'].get('search_run_seed'))}.",
                 f"Dev generation seeds S = {self.S}; dev numbers are selection-biased; the test set is never shown."]
        if self.state.get("noise"):
            nz = self.state["noise"]
            notes.append(f"Training noise (measured): the fixed baseline trained with {nz['k']} different run seeds scored "
                         f"{', '.join(f'{x:.3f}' for x in nz['run_scores'])} on dev (sd {nz['sd_train']:.3f}). A screening result is ONE run: "
                         f"differences smaller than about 2 x {nz['sd_train']:.3f} are indistinguishable from noise. After the search the "
                         f"top {self.top_k} candidates are retrained on {len(self.confirm_seeds)} more seeds and compared with the baseline "
                         "on the same seeds; only that decides. Do not spend candidates on replicates (changing only a seed / salt / "
                         "shuffle is a duplicate and is not trained).")
        notes = [x for x in notes if x]
        if siblings:
            notes.append("Other candidates already written this round (their code is in candidates/): " + ", ".join(siblings)
                         + ". Propose a clearly different idea.")
        sib = [s for s in siblings if self.state["candidates"][s].get("code_dir")]
        inp = write_report(rdir, [e for e in self.trained_entries() if e["id"] not in siblings and e["id"] != cid] + [
            {"id": s, "role": "agent", "status": "training (results not yet available)", "candidate_dir": self.state["candidates"][s]["code_dir"]}
            for s in sib], self.pool_stats(), notes, gallery_ids=self.gallery_ids(),
            instructions=self.P["search"]["agent"].get("instructions", "BUILDER_AGENT.md"),
            baseline_items=(self.state.get("baseline_mean_items"), self.state.get("baseline_sd_items")))
        full_prompt = (open(os.path.join(EXP, self.P["search"]["agent"].get("instructions", "BUILDER_AGENT.md"))).read() +
                       f"\n\n---\nSession: write candidate {cid}. Follow the required workflow: read /input/report.md, "
                       "/input/results.json, /input/dev_items/, /input/analysis_traces/ and /input/api/ first, then write "
                       "proposal.md with all required sections, then edit the files in /work. Finish with a short summary.")
        cont_prompt = (f"Next step of the same search: write candidate {cid} (round {rnd}). {budget_line} The Runner has refreshed /input "
                       f"(new results, failure cases and candidates) and /work now starts as a copy of {start}. Follow the required "
                       "workflow of BUILDER_AGENT.md: first read /input/report.md, /input/results.json, /input/dev_items/ and "
                       "/input/analysis_traces/ completely, then write proposal.md with all required sections (Evidence citing "
                       "dev items by id), then edit the files in /work; finish with a short summary.")
        resume = self.state.get("agent_session") if self.P["search"]["agent"].get("session") == "resume_chain" else None
        prompt = cont_prompt if resume else full_prompt
        home = os.path.join(self.root, "agent_home")
        attempts = 0
        res = None
        if self.replay_src:                    # replay-decisions: no LLM; the source run's candidate code is used as is
            src = os.path.join(RUNS, self.replay_src, "candidates", cid, "code")
            if not os.path.isdir(src):
                c.update(status="rejected", error=f"replay: {cid} has no valid code in {self.replay_src}")
                self.event("candidate_rejected", cand=cid, reason="replay source had no valid candidate")
                return False
            for f in os.listdir(work):
                os.remove(os.path.join(work, f))
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(work, f))
            res = {"ok": True, "replayed_from": self.replay_src, "wall_s": 0.0}
            attempts = 10 ** 6
        while attempts <= self.P["search"]["agent_retry"]:
            res = call_agent(prompt, work, inp, os.path.join(rdir, f"agent_{attempts}"), timeout_s=self.P["search"]["agent"]["timeout_s"],
                             max_turns=self.P["search"]["agent"]["max_turns"], home=home, resume=resume)
            self.event("agent_call", cand=cid, attempt=attempts, ok=res["ok"], models=res.get("models"), cost_usd=res.get("cost_usd"),
                       wall_s=round(res["wall_s"]), resumed_from=resume, session_id=res.get("session_id"))
            if res["ok"]:
                if res.get("session_id"):
                    self.state["agent_session"] = res["session_id"]
                break
            if resume:                          # resume failed: fall back to a fresh session with the full instructions
                self.state["failures"].append({"cand": cid, "kind": "agent_resume_failed", "session": resume,
                                               "detail": (res.get("error") or "")[:500]})
                resume, prompt = None, full_prompt
            attempts += 1
        c.setdefault("agent_calls", []).append({k: res.get(k) for k in ("ok", "models", "cost_usd", "num_turns", "wall_s", "session_id", "replayed_from", "resumed_from")})
        if not res["ok"]:
            c.update(status="failed", error="agent call failed twice")
            self.state["failures"].append({"cand": cid, "kind": "agent_call", "detail": res.get("error", "")[:1000]})
            self.state["rejects_in_a_row"] += 1
            self.event("candidate_failed", cand=cid, reason="agent")
            return False
        chk = run_checks(work, os.path.join(rdir, "check_work"))
        chk["errors"] += self.proposal_errors(work)
        write_json(os.path.join(rdir, "checks.json"), chk)
        if chk["errors"] and self.P["search"]["repair_rounds"] > 0:
            self.event("checks_failed", cand=cid, errors=chk["errors"][:5])
            rp = (prompt + "\n\n---\nThe Runner's checks rejected the files now in /work. Fix every error below and deliver a valid "
                  "candidate (same idea if possible). Errors:\n" + json.dumps(chk["errors"], indent=1)[:6000]
                  + ("\nPlugin log tail:\n" + chk.get("plugin_log_tail", ""))[:3000])
            res = call_agent(rp if not self.state.get("agent_session") else
                             "Repair step: the Runner's checks rejected the files now in /work. Fix every error below and deliver a valid candidate "
                             "(same idea if possible). Errors:\n" + json.dumps(chk["errors"], indent=1)[:6000]
                             + ("\nPlugin log tail:\n" + chk.get("plugin_log_tail", ""))[:3000],
                             work, inp, os.path.join(rdir, "agent_repair"), timeout_s=self.P["search"]["agent"]["timeout_s"],
                             max_turns=self.P["search"]["agent"]["max_turns"], home=home, resume=self.state.get("agent_session"))
            if res.get("ok") and res.get("session_id"):
                self.state["agent_session"] = res["session_id"]
            c["agent_calls"].append({k: res.get(k) for k in ("ok", "models", "cost_usd", "num_turns", "wall_s", "session_id", "resumed_from")})
            self.event("agent_repair", cand=cid, ok=res["ok"], cost_usd=res.get("cost_usd"))
            chk = run_checks(work, os.path.join(rdir, "check_work"))
            chk["errors"] += self.proposal_errors(work)
            write_json(os.path.join(rdir, "checks.json"), chk)
        # patch against the start program
        diff = []
        for f in CAND_FILES:
            a = os.path.join(rdir, "work_start", f)
            b = os.path.join(work, f)
            la = open(a).read().splitlines(True) if os.path.exists(a) else []
            lb = open(b).read().splitlines(True) if os.path.exists(b) else []
            diff += difflib.unified_diff(la, lb, f"{start}/{f}", f"{cid}/{f}")
        open(os.path.join(rdir, "patch.diff"), "w").writelines(diff)
        if chk["errors"]:
            c.update(status="rejected", error="checks: " + "; ".join(chk["errors"])[:2000])
            self.state["rejects_in_a_row"] += 1
            self.state["failures"].append({"cand": cid, "kind": "checks", "errors": chk["errors"][:10]})
            self.event("candidate_rejected", cand=cid, errors=chk["errors"][:5])
            return False
        prog = self.program_sha(work)
        dup = [k for k, v in self.state["candidates"].items() if k != cid and v.get("code_dir") and self.program_sha(v["code_dir"]) == prog]
        if dup:                                # identical program (code + templates + config): not trained again
            c.update(status="duplicate", error=f"identical program to {dup[0]}" + (" (code and templates; config-only changes such as a "
                     "shuffle salt do not make a new program -- replicates are run by the Runner)" if self.P["search"].get("config_only_is_duplicate") else ""))
            self.state["failures"].append({"cand": cid, "kind": "duplicate", "of": dup[0]})
            self.event("candidate_duplicate", cand=cid, of=dup[0])
            return False
        code_dir = os.path.join(self.root, "candidates", cid, "code")
        shutil.rmtree(code_dir, ignore_errors=True)
        shutil.copytree(work, code_dir, ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
        self.state["last_memory"] = os.path.join(code_dir, "memory.md")
        c.update(status="checked", code_dir=code_dir, class_label=chk["class_label"], declared_class=chk["declared_class"],
                 checks_summary={"class_label": chk["class_label"], "tested": chk["tested_behaviour"], "warnings": chk["warnings"],
                                 "sim_stats": chk["sim_stats"]}, parent=start)
        self.candidate_record(cid, code_dir, round=rnd, parent=start, declared_class=chk["declared_class"], class_label=chk["class_label"],
                              agent=c["agent_calls"], status="checked")
        self.state["rejects_in_a_row"] = 0
        self.event("candidate_checked", cand=cid, class_label=chk["class_label"], warnings=chk["warnings"])
        return True

    def train_candidate(self, cid: str) -> None:
        c = self.state["candidates"][cid]
        if c["status"] != "checked":
            return
        seed = self.P["training"].get("screen_seed", self.P["training"].get("search_run_seed"))
        spec = self.run_spec(f"{cid}-s{seed}-g4", c["code_dir"], seed, self.steps, self.evals("dev", self.S))
        st = self.execute(spec)
        run_dir = read_json(spec)["run_dir"]
        if st != "done":
            err = read_json(os.path.join(run_dir, "FAILED.json"))["error"] if st == "candidate_failed" else "infrastructure failure after retry"
            c.update(status="failed", error=err[:3000], run_dir=run_dir)
            self.state["failures"].append({"cand": cid, "kind": st, "error": err[:1500]})
            self.event("candidate_failed", cand=cid, reason=st, error=err[:300])
            return
        au = audit_run(run_dir)
        write_json(os.path.join(run_dir, "audit.json"), au)
        c["audit"] = {k: au.get(k) for k in ("mi_bits", "p_perm", "flag", "templates")}
        if au.get("flag"):
            c.update(status="rejected", error="text-leakage audit flagged", run_dir=run_dir)
            self.state["failures"].append({"cand": cid, "kind": "audit", "audit": c["audit"]})
            self.event("candidate_rejected", cand=cid, reason="audit")
            return
        c["status"] = "trained"
        self.summarize_job(cid, spec)
        try:
            self.manage_disk()
        except Exception as e:  # noqa: BLE001  disk housekeeping must not fail a candidate
            self.state["failures"].append({"kind": "disk_manage_error", "detail": str(e)[:300]})
        self.candidate_record(cid, c["code_dir"], round=c.get("round"), parent=c.get("parent"), class_label=c.get("class_label"),
                              status="trained", run=os.path.basename(run_dir), dev_mean=c["dev_mean"])
        self.event("candidate_trained", cand=cid, dev_mean=round(c["dev_mean"], 4), comparison=c.get("comparison"))

    # ------------------------------------------------------------------ phases
    def p0(self) -> None:
        self.state["phase"] = "P0"
        self.state.setdefault("budget_start", time.time())
        self.event("phase_start")
        spec = os.path.join(self.root, "jobs", "p0_pilot", "run.json")
        if not os.path.exists(spec):
            d = os.path.dirname(spec)
            os.makedirs(d, exist_ok=True)
            write_json(spec, {"run_id": "p0_pilot", "run_dir": d, "candidate_dir": os.path.join(EXP, "seeds", "fixed_f0_final"),
                              "run_seed": 1, "steps": 4, "evals": self.evals("dev", [0]), "max_eval_rows": 8, "overrides": []})
        st = self.execute(spec)
        if st != "done":
            raise SystemExit("P0 pilot failed: blocking (see failures in state.json)")
        cost = self.job_cost(spec)
        fs = read_json(os.path.join(os.path.dirname(spec), "first_step.json"))
        ev = list(cost["evals"].values())[0]
        sec_step = cost["train_wall_s"] / 4
        sec_row = ev["gen_wall_s"] / 8 * 4               # per row per GPU (8 rows on 4 GPUs)
        n_dev = 4 if self.smoke else 100
        est = {"sec_per_step": sec_step, "gen_sec_per_row_gpu": sec_row,
               "candidate_job_h": (cost["load_wall_s"] + sec_step * self.steps + 4 * 15 + len(self.S) * sec_row * n_dev / 4 + 120) / 3600}
        self.state["estimates"] = est
        self.state["first_step_check"] = fs
        self.event("pilot", estimates=est, first_step=fs)
        if not fs.get("ok"):
            raise SystemExit("first-step acceptance failed: blocking")

    def p1(self) -> None:
        if self.design == "v3":
            return self.p1_v3()
        self.state["phase"] = "P1"
        self.event("phase_start")
        cands = self.state["candidates"]
        cands.setdefault("c000", {"role": "fixed_baseline", "status": "checked", "class_label": "fixed",
                                  "code_dir": os.path.join(EXP, "seeds", "fixed_f0_final")})
        self.candidate_record("c000", cands["c000"]["code_dir"], role="fixed_baseline", class_label="fixed")
        seed = self.P["training"]["search_run_seed"]
        base_spec = self.run_spec(f"c000-s{seed}-g4", cands["c000"]["code_dir"], seed, self.steps,
                                  self.evals("dev", self.S + self.S_null))
        model_spec = self.run_spec("base_dev", None, 0, 0, self.evals("dev", self.S))
        results = {}
        ts = [threading.Thread(target=lambda k, s: results.__setitem__(k, self.execute(s)), args=(k, s))
              for k, s in (("c000", base_spec), ("base", model_spec))]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if results["c000"] != "done" or results["base"] != "done":
            raise SystemExit(f"P1 failed {results}: blocking")
        cands["c000"]["status"] = "trained"
        self.summarize_job("c000", base_spec)
        bitems = stats.item_means(read_json(model_spec)["run_dir"], "dev", self.S)
        self.state["base_dev"] = {"run_dir": read_json(model_spec)["run_dir"], "mean": stats.agg(bitems),
                                  "task_means": {t: float(sum(v for k, v in bitems.items() if stats.task_of(k) == t) /
                                                          sum(1 for k in bitems if stats.task_of(k) == t)) for t in sorted({stats.task_of(k) for k in bitems})}}
        bd = read_json(base_spec)["run_dir"]
        nd = stats.null_delta(stats.item_means(bd, "dev", self.S), stats.item_means(bd, "dev", self.S_null))
        self.state["delta"] = nd
        write_json(os.path.join(self.root, "delta.json"), nd)
        cost = self.job_cost(base_spec)
        est = self.state["estimates"]
        est["candidate_job_h_measured"] = (cost["load_wall_s"] + cost["train_wall_s"] + sum(v["gen_wall_s"] for v in cost["evals"].values())
                                           * len(self.S) / len(self.S + self.S_null) + 180) / 3600
        self.event("p1_done", baseline_dev=round(cands["c000"]["dev_mean"], 4), base_dev=round(self.state["base_dev"]["mean"], 4),
                   delta=nd, estimates=est)

    def p2(self) -> None:
        self.state["phase"] = "P2"
        self.event("phase_start")
        est_h = self.state["estimates"].get("candidate_job_h_measured", self.state["estimates"]["candidate_job_h"]) + 0.4
        r0, c0 = self.state.get("round_offset", 0), self.state.get("cand_offset", 0)
        max_new = self.P["search"]["max_agent_candidates"]
        n = c0
        for rnd in range(r0 + 1, r0 + self.n_rounds + 1):
            threads, made = [], []
            for idx in range(self.per_round):
                if n - c0 >= max_new:
                    break
                n += 1
                cid = f"c{n:03d}"
                if self.state["rejects_in_a_row"] >= self.P["search"]["stop_after_consecutive_rejects"]:
                    self.event("stop", reason="consecutive rejects")
                    break
                if not self.search_allowed(est_h) and self.state["candidates"].get(cid, {}).get("status") in (None, "new"):
                    self.event("stop", reason="budget", elapsed_h=round(self.elapsed_h(), 2), est_h=round(est_h, 2))
                    break
                self.manage_disk(before_launch=True)
                self.budget_info = {"new_index": n - c0, "new_total": max_new, "left_after_this": max_new - (n - c0),
                                    "round": rnd, "last_round": r0 + self.n_rounds, "rounds_left_after_this": r0 + self.n_rounds - rnd,
                                    "wall_h_left": round(self.budget_left_h(), 2)}
                if self.make_candidate(cid, rnd, idx, made):
                    t = threading.Thread(target=self._safe_train, args=(cid,))
                    t.start()
                    threads.append(t)
                made.append(cid)
            [t.join() for t in threads]
            self.manage_disk()
            self.event("round_done", round=rnd, disk_gb=round(self.disk_gb(), 1))
            if (n - c0 >= max_new or self.state["rejects_in_a_row"] >= self.P["search"]["stop_after_consecutive_rejects"]
                    or not self.search_allowed(est_h)):
                break

    def _safe_train(self, cid: str) -> None:
        try:
            self.train_candidate(cid)
        except Exception as e:  # noqa: BLE001  a Runner bug must not hang the round; recorded as a failure
            self.state["candidates"][cid].update(status="failed", error=f"runner error: {e}")
            self.state["failures"].append({"cand": cid, "kind": "runner_exception", "trace": traceback.format_exc()[-3000:]})
            self.event("candidate_failed", cand=cid, reason="runner exception")

    def select(self) -> None:
        self.state["phase"] = "SELECT"
        self.event("phase_start")
        if "selection" in self.state:
            return
        cands = self.state["candidates"]
        base = {"id": "c000", "items": stats.item_means(cands["c000"]["run_dir"], "dev", self.S)}
        bgpu = cands["c000"]["cost_summary"]["train_gpu_s"] or 1.0
        pool = [{"id": cid, "class_label": c["class_label"], "compute_ratio": (c["cost_summary"]["train_gpu_s"] or bgpu) / bgpu,
                 "items": stats.item_means(c["run_dir"], "dev", self.S)}
                for cid, c in sorted(cands.items()) if cid != "c000" and c["status"] == "trained"]
        dec = stats.select(pool, base, self.state["delta"]["delta"])
        dec["note"] = "dev comparisons are selection-biased; only the formal retraining evaluated on test is confirmatory (1 training seed)"
        self.state["selection"] = dec
        write_json(os.path.join(self.root, "selection.json"), dec)
        self.event("selected", selected=dec["selected"], rule=dec["rule"], delta=dec["delta"])

    def p3(self) -> None:
        self.state["phase"] = "FROZEN"
        sel = self.state["selection"]["selected"]
        code = self.state["candidates"][sel]["code_dir"]
        frozen = os.path.join(self.root, "frozen", sel)
        if not os.path.exists(frozen):
            shutil.copytree(code, frozen)
        self.state["frozen"] = {"cand": sel, "dir": frozen, "code_sha256": sha256_tree(
            [os.path.join(frozen, f) for f in ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml")])}
        self.event("frozen", **self.state["frozen"])
        self.state["phase"] = "P3"
        self.event("phase_start")
        fseed = self.P["training"]["formal_run_seed"]
        specs = {"baseline_formal": self.run_spec(f"formal_c000-s{fseed}-g4", self.state["candidates"]["c000"]["code_dir"], fseed,
                                                  self.steps, self.evals("test", self.S)),
                 "base_test": self.run_spec("base_test", None, 0, 0, self.evals("test", self.S))}
        if sel != "c000":
            specs["selected_formal"] = self.run_spec(f"formal_{sel}-s{fseed}-g4", frozen, fseed, self.steps, self.evals("test", self.S))
        results = {}
        ts = [threading.Thread(target=lambda k, s: results.__setitem__(k, self.execute(s)), args=(k, s)) for k, s in specs.items()]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.state["p3"] = {k: {"status": v, "run_dir": read_json(specs[k])["run_dir"]} for k, v in results.items()}
        self.event("p3_done", results=results)

    # ------------------------------------------------------------------ design v3
    def run_parallel(self, specs: Dict[str, str]) -> Dict[str, str]:
        results = {}
        ts = [threading.Thread(target=lambda k, sp: results.__setitem__(k, self.execute(sp)), args=(k, sp)) for k, sp in specs.items()]
        [t.start() for t in ts]
        [t.join() for t in ts]
        return results

    def p1_v3(self) -> None:
        """Baseline replicates (one per noise seed) + Base on dev -> training-noise estimate sd_train and delta."""
        self.state["phase"] = "P1"
        self.event("phase_start")
        cands = self.state["candidates"]
        cands.setdefault("c000", {"role": "fixed_baseline", "status": "checked", "class_label": "fixed",
                                  "code_dir": os.path.join(EXP, "seeds", "fixed_f0_final")})
        self.candidate_record("c000", cands["c000"]["code_dir"], role="fixed_baseline", class_label="fixed")
        specs = {f"c000-s{sd}": self.run_spec(f"c000-s{sd}-g4", cands["c000"]["code_dir"], sd, self.steps, self.evals("dev", self.S))
                 for sd in self.base_seeds}
        specs["base"] = self.run_spec("base_dev", None, 0, 0, self.evals("dev", self.S))
        res = self.run_parallel(specs)
        if any(v != "done" for v in res.values()):
            raise SystemExit(f"P1 failed {res}: blocking")
        cands["c000"]["status"] = "trained"
        self.summarize_job("c000", specs[f"c000-s{self.base_seeds[0]}"])          # the screening comparator (seed = screen_seed)
        cands["c000"]["replicates"] = {str(sd): read_json(specs[f"c000-s{sd}"])["run_dir"] for sd in self.base_seeds}
        rep_items = stats.run_means([cands["c000"]["replicates"][str(sd)] for sd in self.base_seeds], "dev", self.S)
        ids = sorted(rep_items[0])
        import numpy as _np
        self.state["baseline_mean_items"] = {i: float(_np.mean([r[i] for r in rep_items])) for i in ids}
        self.state["baseline_sd_items"] = {i: float(_np.std([r[i] for r in rep_items], ddof=1)) if len(rep_items) > 1 else 0.0 for i in ids}
        reps = stats.run_means([cands["c000"]["replicates"][str(sd)] for sd in self.base_seeds], "dev", self.S)
        noise = stats.training_noise(reps)
        noise["seeds"] = self.base_seeds
        self.state["noise"] = noise
        self.state["delta"] = {"delta": noise["delta"], "source": "sd_train of the baseline replicates", **noise}
        write_json(os.path.join(self.root, "noise.json"), noise)
        bitems = stats.item_means(read_json(specs["base"])["run_dir"], "dev", self.S)
        self.state["base_dev"] = {"run_dir": read_json(specs["base"])["run_dir"], "mean": stats.agg(bitems),
                                  "task_means": {t: float(sum(v for k, v in bitems.items() if stats.task_of(k) == t) /
                                                          sum(1 for k in bitems if stats.task_of(k) == t)) for t in sorted({stats.task_of(k) for k in bitems})}}
        cost = self.job_cost(specs[f"c000-s{self.base_seeds[0]}"])
        est = self.state.setdefault("estimates", {})
        est["candidate_job_h_measured"] = (cost["load_wall_s"] + cost["train_wall_s"] + sum(v["gen_wall_s"] for v in cost["evals"].values()) + 120) / 3600
        self.event("p1_done", baseline_runs=[round(x, 4) for x in noise["run_scores"]], sd_train=round(noise["sd_train"], 4),
                   base_dev=round(self.state["base_dev"]["mean"], 4), estimates=est)

    def confirm(self) -> None:
        """Top-k screening candidates retrained on the confirmation seeds; paired with the baseline replicates of the same seeds."""
        self.state["phase"] = "CONFIRM"
        self.event("phase_start")
        cands = self.state["candidates"]
        ranked = sorted([k for k, c in cands.items() if k != "c000" and c["status"] == "trained" and c.get("comparison")],
                        key=lambda k: -cands[k]["comparison"]["D"])
        top = ranked[: self.top_k]
        self.state["confirm_set"] = top
        specs = {}
        for cid in top:
            for sd in self.confirm_seeds:
                specs[f"{cid}-s{sd}"] = self.run_spec(f"{cid}-s{sd}-g4", cands[cid]["code_dir"], sd, self.steps, self.evals("dev", self.S))
        res = self.run_parallel(specs) if specs else {}
        out = {}
        base = stats.run_means([cands["c000"]["replicates"][str(sd)] for sd in self.confirm_seeds], "dev", self.S)
        for cid in top:
            if any(res[f"{cid}-s{sd}"] != "done" for sd in self.confirm_seeds):
                out[cid] = {"error": "confirmation runs failed", "status": {sd: res[f"{cid}-s{sd}"] for sd in self.confirm_seeds}}
                continue
            cr = stats.run_means([read_json(specs[f"{cid}-s{sd}"])["run_dir"] for sd in self.confirm_seeds], "dev", self.S)
            out[cid] = stats.confirm_compare(cr, base, self.state["noise"]["sd_train"], m=max(1, len(top)))
            cands[cid]["confirm"] = out[cid]
        self.state["confirm"] = out
        write_json(os.path.join(self.root, "confirm.json"), {"top_k": top, "seeds": self.confirm_seeds, "results": out})
        self._disk_safe()
        self.event("confirm_done", top=top, results={k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()
                                                         if kk in ("D", "ci_lo", "ci_hi", "error")} for k, r in out.items()})

    def select_v3(self) -> None:
        self.state["phase"] = "SELECT"
        self.event("phase_start")
        if "selection" in self.state:
            return
        delta = self.state["noise"]["delta"]
        ok = [(r["D"], k) for k, r in self.state.get("confirm", {}).items() if "D" in r and r["D"] >= delta and r["ci_lo"] > 0]
        if ok:
            D, sel = max(ok)
            rule = "confirmed candidate with D >= delta (sd_train) and CI lower bound > 0 (training noise included, Bonferroni over top-k)"
        else:
            sel, rule = "c000", "no confirmed candidate passed: fixed baseline"
        dec = {"selected": sel, "rule": rule, "delta": delta, "confirm": self.state.get("confirm", {}),
               "note": "dev is the selection set; only P3's multi-seed formal retraining on test is independent"}
        self.state["selection"] = dec
        write_json(os.path.join(self.root, "selection.json"), dec)
        self.event("selected", selected=sel, rule=rule, delta=round(delta, 4))

    def p3_v3(self) -> None:
        self.state["phase"] = "FROZEN"
        sel = self.state["selection"]["selected"]
        frozen = os.path.join(self.root, "frozen", sel)
        if not os.path.exists(frozen):
            shutil.copytree(self.state["candidates"][sel]["code_dir"], frozen)
        self.state["frozen"] = {"cand": sel, "dir": frozen, "code_sha256": sha256_tree(
            [os.path.join(frozen, f) for f in ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml")])}
        self.event("frozen", **self.state["frozen"])
        self.state["phase"] = "P3"
        self.event("phase_start")
        test_S = self.P["final"]["test_S"]
        specs = {"base_test": self.run_spec("base_test", None, 0, 0, self.evals("test", test_S))}
        arms = ["c000"] + ([sel] if sel != "c000" else [])
        for arm in arms:
            code = self.state["candidates"]["c000"]["code_dir"] if arm == "c000" else frozen
            for sd in self.formal_seeds:
                specs[f"{arm}-s{sd}"] = self.run_spec(f"formal_{arm}-s{sd}-g4", code, sd, self.steps, self.evals("test", test_S))
        res = self.run_parallel(specs)
        self.state["p3"] = {k: {"status": v, "run_dir": read_json(specs[k])["run_dir"]} for k, v in res.items()}
        if all(v == "done" for v in res.values()):
            tm = lambda arm: stats.run_means([read_json(specs[f"{arm}-s{sd}"])["run_dir"] for sd in self.formal_seeds], "test", test_S)  # noqa: E731
            base_formal = tm("c000")
            out = {"base_test": stats.agg(stats.item_means(read_json(specs["base_test"])["run_dir"], "test", test_S)),
                   "baseline_formal_runs": [stats.agg(x) for x in base_formal]}
            if sel != "c000":
                out["selected_vs_baseline"] = stats.confirm_compare(tm(sel), base_formal, self.state["noise"]["sd_train"], m=1)
            self.state["p3_stats"] = out
        self._disk_safe()
        self.event("p3_done", results=res)

    def final_report(self) -> None:
        self.state["phase"] = "REPORT"
        self.event("phase_start")
        self._disk_safe()
        from final_report import build, build_v3
        (build_v3 if self.design == "v3" else build)(self)
        self.state["phase"] = "DONE"
        self.event("done", elapsed_h=round(self.elapsed_h(), 2))

    def run(self) -> None:
        order = ["INIT", "P0", "P1", "P2", "CONFIRM", "SELECT", "FROZEN", "P3", "REPORT", "DONE"]
        done_upto = self.state.get("done_upto", "INIT")
        phases = (("P0", self.p0), ("P1", self.p1), ("P2", self.p2), ("CONFIRM", self.confirm), ("SELECT", self.select_v3),
                  ("P3", self.p3_v3), ("REPORT", self.final_report)) if self.design == "v3" else \
                 (("P0", self.p0), ("P1", self.p1), ("P2", self.p2), ("SELECT", self.select), ("P3", self.p3), ("REPORT", self.final_report))
        for ph, fn in phases:
            if order.index(done_upto) >= order.index(ph):
                continue
            fn()
            self.state["done_upto"] = ph
            self.save()
            self.backup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replay-decisions", default=None, help="source run id: reuse its candidate code, no LLM calls")
    ap.add_argument("--protocol", default=PROTO, help="protocol file (default protocol.yaml)")
    ap.add_argument("--continue-from", default=None, help="finished source run id to continue (candidates, baselines, agent session)")
    a = ap.parse_args()
    Runner(a.run_id, a.smoke, a.replay_decisions, a.protocol, a.continue_from).run()


if __name__ == "__main__":
    main()
