"""Post-hoc text-leakage audit of a training run (Runner-only; reads generator metadata and pool statistics).

In reactdata-v1 the original description already names the colours and the size change, so a template choice that
follows those (allowed) facts is not leakage. The audited channel is a template choice that follows target-only pixel
information beyond the description: the final frame's foreground area (pool statistics, quartiles), *conditional on* the
description's (scale_from, scale_to). Statistic: conditional mutual information I(template; area bin | stratum) in
bits; null: 1,000 permutations of the template labels within strata (fixed seed). Flag when p < 0.001 and
CMI > 0.02 bits. It can reveal association; it cannot prove the absence of leakage.
"""
from __future__ import annotations

import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, read_json, read_jsonl  # noqa: E402

N_PERM = 1000
P_THR, MI_THR = 0.001, 0.02


def mi(xs: List, ys: List) -> float:
    n = len(xs)
    cx, cy, cxy = Counter(xs), Counter(ys), Counter(zip(xs, ys))
    return sum(c / n * math.log2((c / n) / ((cx[x] / n) * (cy[y] / n))) for (x, y), c in cxy.items())


def cmi(t: List, y: List, z: List) -> float:
    n = len(t)
    by = defaultdict(list)
    for i in range(n):
        by[z[i]].append(i)
    return sum(len(ix) / n * mi([t[i] for i in ix], [y[i] for i in ix]) for ix in by.values())


def audit_run(run_dir: str, *, paths=None) -> Dict:
    data = paths.data if paths else DATA
    rows = read_jsonl(os.path.join(run_dir, "rows_realized.jsonl"))
    if not rows:
        return {"n_rows": 0, "flag": False}
    pv = {c["clip_id"]: c for c in read_json(os.path.join(data, "pool_view.json"))["clips"]}
    area = {cid: pv[cid]["fg_area"][-1] for cid in {r["clip_id"] for r in rows}}
    vals = sorted(area.values())
    q = [vals[int(len(vals) * f)] for f in (0.25, 0.5, 0.75)]
    strat = {}
    for cid in area:
        p = read_json(os.path.join(data, "meta", "train", f"{cid}.json"))["parameters"]
        strat[cid] = (p["scale_from"], p["scale_to"])
    t = [r["template_id"] for r in rows]
    y = [sum(area[r["clip_id"]] > v for v in q) for r in rows]
    z = [strat[r["clip_id"]] for r in rows]
    out = {"n_rows": len(rows), "templates": dict(Counter(t)), "quantity": "final-frame foreground area quartile | (scale_from, scale_to)"}
    if len(set(t)) == 1:
        out.update({"cmi_bits": 0.0, "p_perm": 1.0, "flag": False, "note": "single template: no template-choice channel"})
        return out
    obs = cmi(t, y, z)
    rng = random.Random(20261026)
    idx = defaultdict(list)
    for i, s in enumerate(z):
        idx[s].append(i)
    ge = 0
    for _ in range(N_PERM):
        tp = list(t)
        for ix in idx.values():
            vals_ = [t[i] for i in ix]
            rng.shuffle(vals_)
            for i, v in zip(ix, vals_):
                tp[i] = v
        ge += cmi(tp, y, z) >= obs
    p = (ge + 1) / (N_PERM + 1)
    out.update({"cmi_bits": obs, "p_perm": p, "flag": bool(p < P_THR and obs > MI_THR), "rule": f"flag if p < {P_THR} and CMI > {MI_THR} bits",
                "mi_bits": obs})
    return out
