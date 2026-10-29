"""Candidate submission coordinator; filesystem and external-call receipts are separate."""
import os
import shutil
from typing import Dict, List
from experiment.improvement.prompts import PromptContext
from experiment.improvement.submissions import prepare_workspace, validate_submission, write_diff, publish_submission

class CandidateService:
    def __init__(self, store, protocol, contract, policy, evidence, agent, *, seeds, steps, per_round,
                 top_k=0, confirm_seeds=(), replay_src=None, runs_root=None, prompt_builder=None):
        self.store, self.P, self.contract, self.policy = store, protocol, contract, policy
        self.state, self.root, self.event = store.state, store.root, store.event
        self.evidence, self.agent, self.prompts = evidence, agent, prompt_builder
        self.S, self.steps, self.per_round = seeds, steps, per_round
        self.top_k, self.confirm_seeds = top_k, confirm_seeds
        self.replay_src, self.runs_root = replay_src, runs_root
        self.budget_info = {}


    def best_so_far(self):
        return self.policy.best_so_far(self.state)


    def workspace_start(self):
        return self.policy.workspace_start(self.state, self.P["search"])


    def trained_entries(self) -> List[Dict]:
        out = []
        for cid, c in sorted(self.state["candidates"].items()):
            e = {"id": cid, "role": c["role"], "status": c["status"], "candidate_dir": c.get("code_dir"),
                 "run_dir": c.get("run_dir"), "seeds": self.S, "cost": c.get("cost_summary"), "audit": c.get("audit"),
                 "comparison": c.get("comparison"), "error": c.get("error")}
            out.append(e)
        return out


    def _inputs(self, cid, rnd, idx, siblings, parent, directory):
        context = PromptContext(self.P, self.store.snapshot().data, self.per_round, self.steps, self.S,
                                self.top_k, self.confirm_seeds, self.budget_info)
        prompts = self.prompts.build(context, cid, rnd, idx, parent, siblings, self.contract.instructions())
        entries = [e for e in self.trained_entries() if e["id"] not in siblings and e["id"] != cid]
        entries += [{"id": sibling, "role": "agent", "status": "training (results not yet available)",
                     "candidate_dir": self.state["candidates"][sibling]["code_dir"]}
                    for sibling in siblings if self.state["candidates"][sibling].get("code_dir")]
        request = os.path.join(directory, "evidence_request.json")
        inp = self.evidence.rebuild(request) if os.path.isfile(request) else self.evidence.build(directory, entries,
            self.contract.pool_stats(), prompts.notes,
            instructions=self.P["search"]["agent"].get("instructions", "BUILDER_AGENT.md"),
            **self.evidence.search_options(context.state, self.best_so_far()))
        return inp, prompts.full, prompts.continuation

    def _propose(self, cid, work, inp, directory, full, continuation):
        if not self.replay_src:
            return self.agent.propose(cid, work, inp, directory, full, continuation)
        source = os.path.join(self.runs_root, self.replay_src, "candidates", cid, "code")
        if not os.path.isdir(source):
            return {"ok": False, "error": f"replay: {cid} has no valid code in {self.replay_src}"}, full
        for name in os.listdir(work):
            os.remove(os.path.join(work, name))
        for name in os.listdir(source):
            shutil.copy2(os.path.join(source, name), os.path.join(work, name))
        return {"ok": True, "replayed_from": self.replay_src, "wall_s": 0.0}, full

    def _record_call(self, cid, result, key):
        candidate = self.state["candidates"][cid]
        keys = candidate.get("call_keys", [])
        if key in keys:
            return
        record = {k: result.get(k) for k in ("ok", "models", "cost_usd", "num_turns", "wall_s", "session_id", "replayed_from", "resumed_from")}
        self.store.commit("agent_result_recorded", candidates={cid: {
            "agent_calls": candidate.get("agent_calls", []) + [record], "call_keys": keys + [key]}}, cand=cid)

    def _reject(self, cid, status, error, kind, **detail):
        rejects = self.state["rejects_in_a_row"] + (kind in ("agent_call", "checks"))
        self.store.commit("candidate_" + status, candidates={cid: {"status": status, "error": error}},
            updates={"rejects_in_a_row": rejects}, failures=[{"cand": cid, "kind": kind, **detail}], cand=cid, reason=kind)
        return False

    def _validate(self, cid, work, inp, directory, prompt):
        checks = validate_submission(self.contract, work, directory)
        if checks["errors"] and self.P["search"]["repair_rounds"] > 0:
            self.event("checks_failed", cand=cid, errors=checks["errors"][:5])
            result = self.agent.repair(work, inp, directory, self.prompts.repair(prompt, checks, bool(self.state.get("agent_session"))))
            updates = {"agent_session": result["session_id"]} if result.get("ok") and result.get("session_id") else {}
            self.store.commit("agent_repair", updates=updates, cand=cid, ok=result["ok"], cost_usd=result.get("cost_usd"))
            self._record_call(cid, result, "repair")
            checks = validate_submission(self.contract, work, directory)
        return checks

    def make_candidate(self, cid, rnd, idx, siblings):
        existing = self.state["candidates"].get(cid, {})
        if existing.get("status") in ("checked", "trained", "rejected", "failed", "duplicate"):
            return existing["status"] in ("checked", "trained")
        parent = existing.get("parent") or self.workspace_start()
        self.store.commit("submission_started", candidates={cid: {"role": "agent", "round": rnd, "status": "new", "parent": parent}}, cand=cid)
        directory, work = prepare_workspace(self.root, rnd, cid, parent, self.state["candidates"][parent]["code_dir"],
                                             self.contract.files, self.state.get("last_memory"))
        inp, full, continuation = self._inputs(cid, rnd, idx, siblings, parent, directory)
        result, prompt = self._propose(cid, work, inp, directory, full, continuation)
        self._record_call(cid, result, "proposal")
        if not result["ok"]:
            return self._reject(cid, "rejected" if self.replay_src else "failed", result.get("error", "agent call failed twice"),
                                "replay" if self.replay_src else "agent_call")
        # A repair may have completed before its candidate bookkeeping was committed.
        if os.path.isfile(os.path.join(directory, "agent_repair", "operation.json")):
            repaired = self.agent.repair(work, inp, directory, "recover recorded repair")
            if repaired.get("ok") and repaired.get("session_id"):
                self.store.commit("agent_repair_recovered", updates={"agent_session": repaired["session_id"]}, cand=cid)
            self._record_call(cid, repaired, "repair")
        checks = self._validate(cid, work, inp, directory, prompt)
        write_diff(directory, work, self.contract.files, parent, cid)
        if checks["errors"]:
            return self._reject(cid, "rejected", "checks: " + "; ".join(checks["errors"])[:2000], "checks", errors=checks["errors"][:10])
        digest = self.contract.program_sha(work)
        duplicates = [k for k, c in self.state["candidates"].items() if k != cid and c.get("code_dir") and self.contract.program_sha(c["code_dir"]) == digest]
        if duplicates:
            return self._reject(cid, "duplicate", f"identical program to {duplicates[0]}", "duplicate", of=duplicates[0])
        self._publish(cid, rnd, parent, work, checks)
        return True

    def _publish(self, cid, rnd, parent, work, checks):
        code = publish_submission(self.root, cid, work, self.contract.files)
        self.contract.candidate_record(cid, code, round=rnd, parent=parent, declared_class=checks["declared_class"],
            class_label=checks["class_label"], agent=self.state["candidates"][cid]["agent_calls"], status="checked")
        self.store.commit("candidate_checked", updates={"last_memory": os.path.join(code, "memory.md"), "rejects_in_a_row": 0},
            candidates={cid: {"status": "checked", "code_dir": code, "class_label": checks["class_label"], "declared_class": checks["declared_class"],
                "checks_summary": {"class_label": checks["class_label"], "tested": checks["tested_behaviour"], "warnings": checks["warnings"], "sim_stats": checks["sim_stats"]}}},
            cand=cid, class_label=checks["class_label"], warnings=checks["warnings"])
