"""Session continuity and retry are independent of prompt content and CLI transport."""
import os
from experiment.runtime.journal import OperationJournal
from experiment.runtime.io import read_json


class ClaudeBackend:
    def __init__(self, call):
        self._call = call

    def call(self, *args, **options):
        return self._call(*args, **options)


class FakeAgentBackend:
    def __init__(self, respond):
        self.respond, self.calls = respond, []

    def call(self, prompt, work, inp, log_dir, **options):
        self.calls.append({"prompt": prompt, "options": options})
        return self.respond(prompt, work, inp, log_dir, **options)


class AgentService:
    def __init__(self, store, protocol, backend):
        self.store, self.P, self.backend = store, protocol, backend
        self.journal = OperationJournal()

    def _call(self, prompt, work, inp, log_dir, resume):
        config = self.P["search"]["agent"]
        def recover():
            path = os.path.join(log_dir, "output.json")
            return read_json(path) if os.path.isfile(path) else None
        return self.journal.call(log_dir, lambda: self.backend.call(prompt, work, inp, log_dir, timeout_s=config["timeout_s"],
            max_turns=config["max_turns"], home=os.path.join(self.store.root, "agent_home"), resume=resume), recover=recover)

    def propose(self, cid, work, inp, rdir, full_prompt, continuation):
        st = self.store.state
        resume = st.get("agent_session") if self.P["search"]["agent"].get("session") == "resume_chain" else None
        prompt = continuation if resume else full_prompt
        for attempt in range(self.P["search"]["agent_retry"] + 1):
            result = self._call(prompt, work, inp, os.path.join(rdir, f"agent_{attempt}"), resume)
            updates = {"agent_session": result["session_id"]} if result["ok"] and result.get("session_id") else {}
            self.store.commit("agent_call_recovered" if self.journal.reused else "agent_call", updates=updates, cand=cid, attempt=attempt, ok=result["ok"], models=result.get("models"),
                             cost_usd=result.get("cost_usd"), wall_s=round(result["wall_s"]),
                             resumed_from=resume, session_id=result.get("session_id"))
            if result["ok"]:
                break
            if resume:
                self.store.commit("agent_resume_failed", failures=[{"cand": cid, "kind": "agent_resume_failed", "session": resume,
                                       "detail": (result.get("error") or "")[:500]}], cand=cid)
                resume, prompt = None, full_prompt
        return result, prompt

    def repair(self, work, inp, rdir, prompt):
        return self._call(prompt, work, inp, os.path.join(rdir, "agent_repair"), self.store.state.get("agent_session"))
