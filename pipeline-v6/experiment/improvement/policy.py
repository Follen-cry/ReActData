"""Candidate order, stopping, lineage and confirmation shortlist."""

class DefaultSearchPolicy:
    def __init__(self, design="v3"):
        self.design = design

    def stages(self):
        # Compatibility view. New applications take their stage plan from ExperimentDesign.
        from experiment.workflow.design import stage_plan
        return stage_plan(self.design)

    def candidates(self, rounds, per_round, round_offset=0, cand_offset=0):
        for rnd in range(round_offset + 1, round_offset + rounds + 1):
            yield rnd, [(f"c{cand_offset + (rnd - round_offset - 1) * per_round + i + 1:03d}", i)
                        for i in range(per_round)]

    @staticmethod
    def stop_reason(state, search, new_count, allowed, status=None):
        if new_count > search["max_agent_candidates"]:
            return "max candidates"
        if state["rejects_in_a_row"] >= search["stop_after_consecutive_rejects"]:
            return "consecutive rejects"
        if not allowed and status in (None, "new"):
            return "budget"
        return None

    @staticmethod
    def best_so_far(state):
        ok = [(c["dev_mean"], cid) for cid, c in state["candidates"].items()
              if c["status"] == "trained" and "dev_mean" in c]
        return max(ok)[1] if ok else "c000"

    def workspace_start(self, state, search):
        if search.get("workspace_start") == "latest_valid":
            mine = sorted(k for k, c in state["candidates"].items() if k != "c000"
                          and c["status"] in ("trained", "checked") and not c.get("inherited_from"))
            return mine[-1] if mine else (self.best_so_far(state) if state.get("inherited_from") else "c000")
        return self.best_so_far(state)

    @staticmethod
    def confirmation_set(candidates, top_k):
        return sorted([k for k, c in candidates.items() if k != "c000" and c["status"] == "trained" and c.get("comparison")],
                      key=lambda k: -candidates[k]["comparison"]["D"])[:top_k]


class LegacySearchPolicy(DefaultSearchPolicy):
    def __init__(self):
        super().__init__("v1")
