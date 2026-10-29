"""Rule scoring (Runner-only), per row task (reactdata-v6: three tasks):
  shape_color_then_scale / shape_color_then_move  reactdata-v1 analogy-grid scorer (scorers/analogy_grid.py)
  2d_geometric_transformation                     VBVR-CustomEval 2d_geometric_transformation scorer (as in reactdata-v0)
Sub-score names are task-specific (TASK_SUBS); means over sub-scores are taken per task.

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

GEO = "2d_geometric_transformation"
GEO_FILE = os.path.join(CVLIB_DIR, "2d_geometric_transformation.py")
GEO_SUBS = ("rotation_angle", "position_alignment", "shape_fidelity")
TASK_SUBS = {"shape_color_then_scale": G.SUBS, "shape_color_then_move": G.SUBS, GEO: GEO_SUBS}
SUBS = tuple(dict.fromkeys(k for v in TASK_SUBS.values() for k in v))
_GEO = None


def geo():
    global _GEO
    if _GEO is None:
        import importlib.util
        if CVLIB_DIR not in sys.path:
            sys.path.insert(0, CVLIB_DIR)
        spec = importlib.util.spec_from_file_location("rd_geo_scorer", GEO_FILE)
        _GEO = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_GEO)
    return _GEO


def scorer_version() -> Dict[str, str]:
    return {"version": G.VERSION + " + VBVR-CustomEval 2d_geometric_transformation",
            "analogy_grid_sha256": sha256_file(os.path.join(SCORER_DIR, "analogy_grid.py")),
            "geo_scorer_sha256": sha256_file(GEO_FILE), "cvlib_sha256": sha256_file(os.path.join(CVLIB_DIR, "cvlib.py"))}


def geo_categories(sub: Dict[str, float], dbg: Dict) -> List[str]:
    """reactdata-v0 categories for the geo scorer."""
    cats = []
    found = dbg.get("found", {})
    if dbg.get("error") or not found.get("cand", True):
        cats.append("no_shape")
    if dbg.get("ceiling", 1.0) < 1.0 or dbg.get("blankness", {}).get("ceiling", 1.0) < 1.0:
        cats.append("blank_or_sparse")
    if dbg.get("colour_ok") is False:
        cats.append("wrong_colour")
    if "no_shape" not in cats:
        if sub["position_alignment"] < 0.5:
            cats.append("wrong_position")
        if sub["rotation_angle"] < 0.5:
            cats.append("wrong_rotation")
        if sub["shape_fidelity"] < 0.5:
            cats.append("deformed_shape")
        if dbg.get("residual", {}).get("outline_retained", 0.0) > 0.3:
            cats.append("outline_left")
    return cats


def score_item(task: str, f0: str, cand: str, gt: str) -> Dict:
    zero = {k: 0.0 for k in TASK_SUBS[task]}
    if cand is None or not os.path.isfile(cand):
        return {"score": 0.0, "sub": zero, "valid": False, "categories": ["missing"], "gates": {}}
    im = cv2.imread(cand, cv2.IMREAD_COLOR)
    if im is None:
        return {"score": 0.0, "sub": zero, "valid": False, "categories": ["unreadable"], "gates": {}}
    if float(im.max()) < 8 or float(im.std()) < 1.0:
        return {"score": 0.0, "sub": zero, "valid": False, "categories": ["black_or_flat"], "gates": {}}
    if task == GEO:
        m = geo()
        sub, dbg = m._score(m._coerce(f0), m._coerce(im), m._coerce(gt))
        sub = {k: float(sub[k]) for k in GEO_SUBS}
        gates = {"ceiling": dbg.get("ceiling"), "colour_ok": dbg.get("colour_ok"),
                 "outline_retained": dbg.get("residual", {}).get("outline_retained"),
                 "best_fit_deg": dbg.get("rotation", {}).get("best_fit_deg"), "position_err_px": dbg.get("position", {}).get("err_px")}
        return {"score": float(m.overall(sub)), "sub": sub, "valid": True, "categories": geo_categories(sub, dbg), "gates": gates}
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
                      "sub_mean": {k: float(np.mean([x["sub"][k] for x in xs])) for k in TASK_SUBS[t]},
                      "n_invalid": sum(not x["valid"] for x in xs)}
    out = {"n": len(items), "task_means": {t: v["mean"] for t, v in by_task.items()},
           "mean": float(np.mean(list({t: v["mean"] for t, v in by_task.items()}.values()))) if items else float("nan"),
           "mean_note": "equal-weight mean of the task means", "by_task": by_task,
           "n_invalid": sum(not x["valid"] for x in items),
           "sub_mean": {k: float(np.mean([x["sub"][k] for x in items if k in x["sub"]])) for k in SUBS
                        if any(k in x["sub"] for x in items)},
           "version": scorer_version(), "items": items}
    write_json(os.path.join(eval_dir, "scores.json"), out)
    return out


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in score_dir(sys.argv[1], read_jsonl(sys.argv[2])).items() if k != "items"}, indent=1))
