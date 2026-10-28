"""Rule scoring (Runner-only): reactdata-v1 analogy-grid scorer (scorers/analogy_grid.py), per row task.

score_item(task, f0, candidate, gt) -> {"score", "sub", "valid", "categories", "gates"}; ``valid`` is False for a
missing, unreadable, black or flat image (scored 0).

    python runner/score.py <eval_dir> <eval_jsonl>      # scores <eval_dir>/images/*.png -> <eval_dir>/scores.json
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CVLIB_DIR, SCORER_DIR, read_jsonl, sha256_file, write_json  # noqa: E402
sys.path.insert(0, SCORER_DIR)
import analogy_grid as G  # noqa: E402

SUBS = G.SUBS


def scorer_version() -> Dict[str, str]:
    return {"version": G.VERSION, "analogy_grid_sha256": sha256_file(os.path.join(SCORER_DIR, "analogy_grid.py")),
            "cvlib_sha256": sha256_file(os.path.join(CVLIB_DIR, "cvlib.py"))}


def score_item(task: str, f0: str, cand: str, gt: str) -> Dict:
    zero = {k: 0.0 for k in SUBS}
    if cand is None or not os.path.isfile(cand):
        return {"score": 0.0, "sub": zero, "valid": False, "categories": ["missing"], "gates": {}}
    im = cv2.imread(cand, cv2.IMREAD_COLOR)
    if im is None:
        return {"score": 0.0, "sub": zero, "valid": False, "categories": ["unreadable"], "gates": {}}
    if float(im.max()) < 8 or float(im.std()) < 1.0:
        return {"score": 0.0, "sub": zero, "valid": False, "categories": ["black_or_flat"], "gates": {}}
    sub, dbg = G.score_images(task, G.load(f0), G.load(im), G.load(gt))
    gates = {"ceiling": dbg.get("ceiling"), "context_gate": dbg.get("context_gate"), "transform": dbg.get("transform"),
             "n_objects": dbg.get("counts", {}).get("cand")}
    return {"score": G.overall(sub), "sub": {k: float(v) for k, v in sub.items()}, "valid": True,
            "categories": G.categories(sub, dbg), "gates": gates}


def score_dir(eval_dir: str, rows: List[Dict]) -> Dict:
    items = []
    for r in rows:
        cand = os.path.join(eval_dir, "images", r["id"] + ".png")
        items.append({"id": r["id"], "task": r["task_name"], **score_item(r["task_name"], r["image"][0], cand, r["target_image"])})
    by_task = {}
    for t in sorted({x["task"] for x in items}):
        xs = [x for x in items if x["task"] == t]
        by_task[t] = {"n": len(xs), "mean": float(np.mean([x["score"] for x in xs])),
                      "sub_mean": {k: float(np.mean([x["sub"][k] for x in xs])) for k in SUBS},
                      "n_invalid": sum(not x["valid"] for x in xs)}
    out = {"n": len(items), "task_means": {t: v["mean"] for t, v in by_task.items()},
           "mean": float(np.mean(list({t: v["mean"] for t, v in by_task.items()}.values()))) if items else float("nan"),
           "mean_note": "equal-weight mean of the task means", "by_task": by_task,
           "n_invalid": sum(not x["valid"] for x in items),
           "sub_mean": {k: float(np.mean([x["sub"][k] for x in items])) for k in SUBS} if items else {},
           "version": scorer_version(), "items": items}
    write_json(os.path.join(eval_dir, "scores.json"), out)
    return out


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in score_dir(sys.argv[1], read_jsonl(sys.argv[2])).items() if k != "items"}, indent=1))
