"""Statistical acceptance rules, separate from candidate generation."""

class NoiseAwareSelection:
    @staticmethod
    def select(confirm, delta):
        ok = [(r["D"], k) for k, r in confirm.items() if "D" in r and r["D"] >= delta and r["ci_lo"] > 0]
        if ok:
            _, sel = max(ok)
            rule = "confirmed candidate with D >= delta (sd_train) and CI lower bound > 0 (training noise included, Bonferroni over top-k)"
        else:
            sel, rule = "c000", "no confirmed candidate passed: fixed baseline"
        return {"selected": sel, "rule": rule, "delta": delta, "confirm": confirm,
                "note": "dev is the selection set; only P3's multi-seed formal retraining on test is independent"}


class LegacySelection:
    def __init__(self, statistics):
        self.statistics = statistics

    def select(self, candidates, baseline, delta):
        return self.statistics.select(candidates, baseline, delta)
