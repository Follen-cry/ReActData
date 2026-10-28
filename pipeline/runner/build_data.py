"""Build the reactdata-v5 data from the frozen reactdata-v4 data (refuses to rebuild once data/BUILD_DONE.json exists).

  train pool   16 clips of the v4 pool (256 colour-corrected shape_color_then_scale clips), stratified over the 12
               (scale_from, scale_to) combinations (largest remainder; seed 20261028); the stratification reads metadata
               (Runner only). With 100 steps x 16 rows each clip appears ~100 times if only final-frame rows are used.
  frames       hard links to the v4 frames (F0 / final PNG endpoints, colour-corrected mp4 frames)
  inner probe / dev / test: unchanged (v4 = v1)
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, FRAMES, MANIFESTS, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402

V4 = "/scratch/network/ssd/junlin/reactdata-v4"
V4_EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reactdata-v4")
N_POOL, SEED = 16, 20261028


def link_clip(cid):
    src, dst = os.path.join(V4, "data", "frames512", cid), os.path.join(FRAMES, cid)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not os.path.exists(os.path.join(dst, f)):
            os.link(os.path.join(src, f), os.path.join(dst, f))


def main():
    done = os.path.join(DATA, "BUILD_DONE.json")
    if os.path.exists(done):
        raise SystemExit(f"{done} exists; data is frozen")
    pool = read_jsonl(os.path.join(V4_EXP, "manifests", "train_pool.jsonl"))
    probe = read_jsonl(os.path.join(V4_EXP, "manifests", "inner_probe.jsonl"))
    strata = defaultdict(list)
    for r in pool:
        p = read_json(os.path.join(V4, "data", "meta", "train", f"{r['clip_id']}.json"))["parameters"]
        strata[(p["scale_from"], p["scale_to"])].append(r["clip_id"])
    keys = sorted(strata)
    quota = {k: N_POOL * len(strata[k]) / len(pool) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rng = random.Random(SEED)
    rest = sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), rng.random()))
    for k in rest[: N_POOL - sum(alloc.values())]:
        alloc[k] += 1
    chosen = sorted(c for k in keys for c in rng.sample(sorted(strata[k]), alloc[k]))
    assert len(chosen) == N_POOL
    for cid in chosen + [r["clip_id"] for r in probe]:
        link_clip(cid)
    keep = set(chosen)
    for r in pool + probe:
        r["frames"] = os.path.join(FRAMES, r["clip_id"])
    os.makedirs(MANIFESTS, exist_ok=True)
    write_jsonl(os.path.join(MANIFESTS, "train_pool.jsonl"), [r for r in pool if r["clip_id"] in keep])
    write_jsonl(os.path.join(MANIFESTS, "inner_probe.jsonl"), probe)
    pv = read_json(os.path.join(V4, "data", "pool_view.json"))
    pv["clips"] = [c for c in pv["clips"] if c["clip_id"] in keep]
    pv["note"] = "reactdata-v5: 16-clip subset of the v4 pool"
    write_json(os.path.join(DATA, "pool_view.json"), pv)
    os.makedirs(os.path.join(DATA, "meta", "train"), exist_ok=True)
    for cid in chosen:
        os.link(os.path.join(V4, "data", "meta", "train", f"{cid}.json"), os.path.join(DATA, "meta", "train", f"{cid}.json"))
    v4c = {c["clip_id"]: c for c in read_json(os.path.join(V4_EXP, "manifests", "data_checks.json"))["per_clip"]}
    summary = {"n_pool": N_POOL, "seed": SEED, "strata": {f"{k[0]}->{k[1]}": alloc[k] for k in keys}, "clips": chosen,
               "colour_correction": [v4c[c] for c in chosen]}
    write_json(os.path.join(MANIFESTS, "data_checks.json"), summary)
    write_json(done, {"counts": {"train_pool": N_POOL, "inner_probe": len(probe), "dev": 100, "test": 100}, "from": "reactdata-v4 data"})
    print(json.dumps({k: v for k, v in summary.items() if k != "colour_correction"}, indent=1))


if __name__ == "__main__":
    main()
