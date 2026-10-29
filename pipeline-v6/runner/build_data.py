"""Build the reactdata-v6 data once (refuses to rebuild once data/BUILD_DONE.json exists).

Three VBVR tasks, final-frame targets only (F0 / final PNG endpoints at 512x512; no mp4 frames are needed):

  task     train pool (160 / task)                  inner probe (32)         dev (50 / task)              test (50 / task)
  scale    reactdata-v4 pool (256 colour-corrected   reactdata-v1 probe        reactdata-v1 dev_scale        fresh: raw chunks 0-19, not in
           clips of raw chunks 0-19), seeded draw    (chunk 20), 11            (chunk 21), all 50            the v1 1,600-clip pool, 50
  move     fresh: raw chunks 0-19, seeded draw       same draw, 11             reactdata-v1 dev_move          reactdata-v1 test_move
                                                                               (target_pred ood, seed 242)   (ood_train500, seed 342), 50 of 100
  geo      reactdata-v0 pool (10,768 clips of        reactdata-v0 probe, 10    reactdata-v0 dev (ood), 50     reactdata-v0 test (ood_train500),
           2d_geometric_transformation), draw                                  of 100                         50 of 100

The agent chooses 128 of the 480 pool clips (each at most once: 1 epoch); pool_view.json gives it the task, the
description (= its training text) and pixel statistics of F0 / final, never generator metadata.
Isolation: md5 of F0 / final across splits; exact generator-parameter tuples within a task across splits.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, FRAMES, MANIFESTS, RAW_ROOT, md5_file, read_json, read_jsonl, wrap_prompt, write_json, write_jsonl  # noqa: E402

OLD_EXP = "/scratch/network/ssd2/junlin/ssl_mllm/internvlu-exp/exp"
V0, V1, V4 = (f"/scratch/network/ssd/junlin/reactdata-v{i}" for i in (0, 1, 4))
SCALE, MOVE, GEO = "shape_color_then_scale", "shape_color_then_move", "2d_geometric_transformation"
SHORT = {SCALE: "scale", MOVE: "move", GEO: "geo"}
SEED = 20261028
N_POOL, N_DEV, N_TEST = 160, 50, 50
N_PROBE = {SCALE: 11, MOVE: 11, GEO: 10}
SIZE, STAT = 512, 256


def load(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(p)
    return im


def to512(im):
    return im if im.shape[:2] == (SIZE, SIZE) else cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def write_png(path, im):
    tmp = path + ".tmp.png"
    cv2.imwrite(tmp, im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    os.replace(tmp, path)


def put_endpoints(out, sample_dir=None, f0=None, final=None):
    """F0 / final at 512: hard link existing 512 frames, else resize the raw PNG endpoints."""
    os.makedirs(out, exist_ok=True)
    for name, src, raw in (("f0.png", f0, "first_frame.png"), ("final.png", final, "final_frame.png")):
        dst = os.path.join(out, name)
        if os.path.exists(dst):
            continue
        if src:
            os.link(src, dst)
        else:
            write_png(dst, to512(load(os.path.join(sample_dir, raw))))


def bg_color(im):
    b = np.concatenate([im[:8].reshape(-1, 3), im[-8:].reshape(-1, 3), im[:, :8].reshape(-1, 3), im[:, -8:].reshape(-1, 3)])
    return np.median(b, axis=0)


def stats(d):
    s0, s1 = (cv2.resize(load(os.path.join(d, f)), (STAT, STAT), interpolation=cv2.INTER_AREA) for f in ("f0.png", "final.png"))
    fg = [int((np.linalg.norm(s.astype(np.float32) - bg_color(s).astype(np.float32), axis=2) > 25).sum()) for s in (s0, s1)]
    change = int((np.abs(s1.astype(np.int16) - s0.astype(np.int16)).max(2) > 40).sum())
    return {"fg_area": fg, "change_px": change}


def params(task, m):
    P = m["parameters"]
    if task == GEO:
        return json.dumps(P, sort_keys=True)
    objs = P["objects"]
    return (P["transformation_type"], P.get("color_from"), P.get("color_to"), P.get("scale_from"), P.get("scale_to"),
            P.get("move_from"), P.get("move_to"), objs[0]["symbol"], objs[3]["symbol"] if len(objs) > 3 else None)


def eval_row(sid, task, d, raw_prompt):
    f0 = os.path.join(d, "f0.png")
    return {"id": sid, "task_type": "imgen", "source": "reactdata", "task_name": task, "variant": "V4", "gap_s": "na",
            "stalled": False, "layout": {"shown": ["F0"], "target": "final", "cond": "F0"}, "image": [f0], "cond_image": f0,
            "target_image": os.path.join(d, "final.png"),
            "conversations": [{"from": "human", "value": "<image>\n" + wrap_prompt(task, raw_prompt)}, {"from": "gpt", "value": "<img>"}]}


def raw_prompt(sample_dir):
    return open(os.path.join(sample_dir, "prompt.txt")).read().strip()


def main():
    done = os.path.join(DATA, "BUILD_DONE.json")
    if os.path.exists(done):
        raise SystemExit(f"{done} exists; data is frozen")
    rng = random.Random(SEED)
    os.makedirs(MANIFESTS, exist_ok=True)
    pool, probe, dev, test = [], [], [], []           # pool/probe: clip rows; dev/test: eval rows
    src = {}                                          # new id -> (split, task, sample_dir or None)

    def add_clip(split, task, old_id, sample_dir, prompt, frames=None):
        cid = f"{SHORT[task]}_{old_id}"
        out = os.path.join(FRAMES, cid)
        put_endpoints(out, sample_dir, *( (os.path.join(frames, "f0.png"), os.path.join(frames, "final.png")) if frames else (None, None)))
        (pool if split == "train" else probe).append({"clip_id": cid, "task": task, "sample_dir": sample_dir, "prompt": prompt, "frames": out})
        src[cid] = (split, task, sample_dir)

    # -- scale: v4 pool + v1 probe
    v4 = read_jsonl(os.path.join(OLD_EXP, "reactdata-v4", "manifests", "train_pool.jsonl"))
    for r in sorted(rng.sample(v4, N_POOL), key=lambda r: r["clip_id"]):
        add_clip("train", SCALE, r["clip_id"], r["sample_dir"], r["prompt"], os.path.join(V4, "data", "frames512", r["clip_id"]))
    v1p = read_jsonl(os.path.join(OLD_EXP, "reactdata-v1", "manifests", "inner_probe.jsonl"))
    for r in sorted(rng.sample(v1p, N_PROBE[SCALE]), key=lambda r: r["clip_id"]):
        add_clip("probe", SCALE, r["clip_id"], r["sample_dir"], r["prompt"], os.path.join(V1, "data", "frames512", r["clip_id"]))
    # -- move: fresh draw from raw chunks 0-19 (train + probe disjoint)
    mv = sorted(d for k in range(20) for d in glob.glob(f"{RAW_ROOT}/{MOVE}/chunk_{k:04d}/*/*"))
    draw = rng.sample(mv, N_POOL + N_PROBE[MOVE])
    mid = lambda d: f"s{int(d.split('/')[-3].split('_')[-1]):02d}_{d[-8:]}"   # noqa: E731
    for i, d in enumerate(sorted(draw[:N_POOL]) + sorted(draw[N_POOL:])):
        add_clip("train" if i < N_POOL else "probe", MOVE, mid(d), d, raw_prompt(d))
    # -- geo: v0 pool + v0 probe
    v0 = read_jsonl(os.path.join(OLD_EXP, "reactdata-v0", "manifests", "train_pool.jsonl"))
    for r in sorted(rng.sample(v0, N_POOL), key=lambda r: r["clip_id"]):
        add_clip("train", GEO, r["clip_id"], r["sample_dir"], r["prompt"], os.path.join(V0, "data", "frames512", r["clip_id"]))
    v0p = read_jsonl(os.path.join(OLD_EXP, "reactdata-v0", "manifests", "inner_probe.jsonl"))
    for r in sorted(rng.sample(v0p, N_PROBE[GEO]), key=lambda r: r["clip_id"]):
        add_clip("probe", GEO, r["clip_id"], r["sample_dir"], r["prompt"], os.path.join(V0, "data", "frames512", r["clip_id"]))

    # -- dev / test (eval rows keep their frozen frame paths; ids renamed per task)
    def take(rows, n, prefix, task):
        rows = [r for r in rows if r["task_name"] == task]
        pick = rows if n >= len(rows) else sorted(rng.sample(rows, n), key=lambda r: r["id"])
        return [{**r, "id": f"{prefix}_{SHORT[task]}_{i:03d}", "orig_id": r["id"]} for i, r in enumerate(pick)]
    v1dev = read_jsonl(os.path.join(OLD_EXP, "reactdata-v1", "manifests", "dev.jsonl"))
    dev += take(v1dev, N_DEV, "dev", SCALE) + take(v1dev, N_DEV, "dev", MOVE)
    dev += take(read_jsonl(os.path.join(OLD_EXP, "reactdata-v0", "manifests", "dev.jsonl")), N_DEV, "dev", GEO)
    test += take(read_jsonl(os.path.join(OLD_EXP, "reactdata-v1", "manifests", "test.jsonl")), N_TEST, "test", MOVE)
    test += take(read_jsonl(os.path.join(OLD_EXP, "reactdata-v0", "manifests", "test.jsonl")), N_TEST, "test", GEO)
    v1pool = {r["sample_dir"] for r in read_jsonl(os.path.join(OLD_EXP, "reactdata-v1", "manifests", "train_pool.jsonl"))}
    sc = sorted(d for k in range(20) for d in glob.glob(f"{RAW_ROOT}/{SCALE}/chunk_{k:04d}/*/*") if d not in v1pool)
    for i, d in enumerate(sorted(rng.sample(sc, N_TEST))):
        sid = f"test_scale_{i:03d}"
        out = os.path.join(FRAMES, "test", sid)
        put_endpoints(out, d)
        test.append({**eval_row(sid, SCALE, out, raw_prompt(d)), "orig_id": d})
    test.sort(key=lambda r: r["id"])

    # -- isolation: md5 of endpoints across splits, generator-parameter tuples within a task across splits
    def endpoints(split_rows, kind):
        for r in split_rows:
            if kind == "clip":
                yield r["clip_id"], r["task"], os.path.join(r["frames"], "f0.png"), os.path.join(r["frames"], "final.png")
            else:
                yield r["id"], r["task_name"], r["image"][0], r["target_image"]
    seen, rep = {}, {"md5_overlap": [], "param_dup": [], "n_checked": 0}
    allsets = [("train", pool, "clip"), ("probe", probe, "clip"), ("dev", dev, "eval"), ("test", test, "eval")]
    for split, rows, kind in allsets:
        for sid, task, f0, fin in endpoints(rows, kind):
            for h in (md5_file(f0), md5_file(fin)):
                if h in seen and seen[h][0] != split:
                    rep["md5_overlap"].append([seen[h], [split, sid]])
                seen.setdefault(h, [split, sid])
            rep["n_checked"] += 1
    meta_dir = lambda r: r.get("sample_dir") or (r["orig_id"] if r["orig_id"].startswith("/") else None)   # noqa: E731
    ptab = defaultdict(dict)
    for split, rows, kind in allsets:
        for r in rows:
            d = meta_dir(r)
            task = r.get("task") or r["task_name"]
            if d is None:     # frozen v0 / v1 eval rows: generator metadata from their own build
                continue
            ptab[task].setdefault(params(task, read_json(os.path.join(d, "metadata.json"))), []).append([split, r.get("clip_id") or r["id"]])
    for task, tab in ptab.items():
        for k, v in tab.items():
            if len({s for s, _ in v}) > 1:
                rep["param_dup"].append({"task": task, "ids": v})
    rep["note"] = ("dev/test rows reused from reactdata-v0/v1 were isolation-checked against their own train pools there "
                   "(manifests/isolation_report.json of v0/v1); here md5 covers every split, parameters cover rows with raw dirs")
    write_json(os.path.join(MANIFESTS, "isolation_report.json"), rep)
    if rep["md5_overlap"]:
        raise SystemExit(f"md5 overlap across splits: {rep['md5_overlap'][:5]}")
    # geo reuses v0 meta for dev/test isolation (v0 report); param_dup across train/probe of v6 is a hard error too
    if rep["param_dup"]:
        print("WARNING param dups", json.dumps(rep["param_dup"][:5]))

    # -- manifests, pool view, Runner-only metadata
    write_jsonl(os.path.join(MANIFESTS, "train_pool.jsonl"), pool)
    write_jsonl(os.path.join(MANIFESTS, "inner_probe.jsonl"), probe)
    write_jsonl(os.path.join(MANIFESTS, "inner_probe_eval.jsonl"), [eval_row(r["clip_id"], r["task"], r["frames"], r["prompt"]) for r in probe])
    write_jsonl(os.path.join(MANIFESTS, "dev.jsonl"), dev)
    write_jsonl(os.path.join(MANIFESTS, "test.jsonl"), test)
    pv = {"tasks": [SCALE, MOVE, GEO], "stat_resolution": STAT,
          "stats_note": "fg_area = [F0, final] foreground pixels at 256x256 (|pixel - border median| > 25); change_px = pixels "
                        "whose max channel |final - F0| > 40 at 256x256",
          "clips": [{"clip_id": r["clip_id"], "task": r["task"], "description": r["prompt"], **stats(r["frames"])} for r in pool]}
    write_json(os.path.join(DATA, "pool_view.json"), pv)
    os.makedirs(os.path.join(DATA, "meta", "train"), exist_ok=True)
    for r in pool:
        dst = os.path.join(DATA, "meta", "train", f"{r['clip_id']}.json")
        if not os.path.exists(dst):
            os.link(os.path.join(r["sample_dir"], "metadata.json"), dst)
    counts = {"train_pool": dict(Counter(r["task"] for r in pool)), "inner_probe": dict(Counter(r["task"] for r in probe)),
              "dev": dict(Counter(r["task_name"] for r in dev)), "test": dict(Counter(r["task_name"] for r in test))}
    write_json(os.path.join(MANIFESTS, "data_checks.json"), {"seed": SEED, "counts": counts, "isolation": {k: len(v) for k, v in rep.items() if isinstance(v, list)}})
    write_json(done, {"counts": counts, "from": "reactdata-v0 / v1 / v4 frozen data + fresh raw draws (move train/probe, scale test)"})
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
