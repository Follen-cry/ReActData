"""Search scheduling and failure propagation, with task evaluation supplied as a narrow operation."""
import threading
import traceback

class SearchController:
    def __init__(self, store, policy, resources, candidates, evaluate, protocol, rounds, per_round):
        self.store, self.policy, self.resources, self.candidates, self.evaluate = store, policy, resources, candidates, evaluate
        self.state, self.P = store.state, protocol
        self.n_rounds, self.per_round = rounds, per_round
        self.event = store.event

    def run(self) -> None:
        self.store.begin_phase("P2")
        est_h = self.state["estimates"].get("candidate_job_h_measured", self.state["estimates"]["candidate_job_h"]) + 0.4
        r0, c0 = self.state.get("round_offset", 0), self.state.get("cand_offset", 0)
        max_new = self.P["search"]["max_agent_candidates"]
        n = c0
        for rnd, proposals in self.policy.candidates(self.n_rounds, self.per_round, r0, c0):
            threads, made = [], []
            for cid, idx in proposals:
                if n - c0 >= max_new:
                    break
                n += 1
                reason = self.policy.stop_reason(self.store.snapshot().data, self.P["search"], n - c0, self.resources.search_allowed(est_h),
                                                 self.state["candidates"].get(cid, {}).get("status"))
                if reason:
                    if reason == "budget":
                        self.event("stop", reason=reason, elapsed_h=round(self.resources.elapsed_h(), 2), est_h=round(est_h, 2))
                    else:
                        self.event("stop", reason=reason)
                    break
                self.resources.manage_disk(before_launch=True)
                self.candidates.budget_info = {"new_index": n - c0, "new_total": max_new, "left_after_this": max_new - (n - c0),
                                    "round": rnd, "last_round": r0 + self.n_rounds, "rounds_left_after_this": r0 + self.n_rounds - rnd,
                                    "wall_h_left": round(self.resources.budget_left_h(), 2)}
                if self.candidates.make_candidate(cid, rnd, idx, made):
                    t = threading.Thread(target=self._safe_train, args=(cid,))
                    t.start()
                    threads.append(t)
                made.append(cid)
            [t.join() for t in threads]
            self.resources.manage_disk()
            self.event("round_done", round=rnd, disk_gb=round(self.resources.disk_gb(), 1))
            if (n - c0 >= max_new or self.state["rejects_in_a_row"] >= self.P["search"]["stop_after_consecutive_rejects"]
                    or not self.resources.search_allowed(est_h)):
                break


    def _safe_train(self, cid: str) -> None:
        try:
            self.evaluate(cid)
        except Exception as e:  # noqa: BLE001  a Runner bug must not hang the round; recorded as a failure
            self.store.commit("candidate_failed", candidates={cid: {"status": "failed", "error": f"runner error: {e}"}},
                failures=[{"cand": cid, "kind": "runner_exception", "trace": traceback.format_exc()[-3000:]}], cand=cid, reason="runner exception")
