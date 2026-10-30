"""reactdata-v1 rule scorer for the VBVR colour-then-X analogy tasks (shape_color_then_scale, shape_color_then_move).

Scene: a 2x3 grid on white. Top row = worked example A -> B -> C; bottom row = D -> E -> F. F0 shows A, B, C, D
(4 objects); the final frame adds E (= D recoloured like A -> B) and F (= E transformed like B -> C: rescaled for
scale, vertically displaced for move). Measured on 60 GT frames per task at the 512 working size: GT always has 6
objects and F0 always 4; rows split at y = 256; scale F/E area ratio 0.15 .. 6.4; move F - E vertical offset in
{0, +-12, +-13, +-20, ... +-50} px with equal area; smallest object 320 px^2.

Only the cells the model has to produce are scored. A, B, C, D already exist in the input, so they act as a
multiplicative context gate instead of earning credit (a copy of the input scores 0 by construction; the VBVR-CustomEval
move scorer gave 0.40 + 0.05 weight to preservation / count and scored a copy of the input about 0.5).

Sub-scores (all 0..1; total = gates x (0.35 E_cell + 0.40 F_cell + 0.25 F_transform)):
  E_cell       E is present at its cell, same glyph and size as GT's E (centred IoU), GT's colour.
  F_cell       the same for F against GT's F (position, glyph+size, colour).
  F_transform  the task's transform of F relative to the candidate's own E, against GT's:
                 scale: |log(area_F / area_E) - log(GT ratio)|, full credit <= log 1.15, zero at the error of
                        "no rescale" (max(|log GT ratio|, log 1.3)) -> leaving F at E's size scores 0;
                 move:  |(y_F - y_E) - GT offset|, full credit <= 4 px, zero at max(|GT offset|, 10 px)
                        -> leaving F at E's height scores 0 (offset-0 samples: zero at 10 px).
Gates: context = mean cell score of A, B, C, D (ramp: 1 above 0.75, 0 at 0.25); blank / sparse foreground ceiling
(cvlib.blankness); repainted background (dE > 40) -> ceiling 0.15.
Invalid image (missing / unreadable / black / flat) -> 0 everywhere (handled by runner/score.py).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_CV = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/scorers"
if _CV not in sys.path:
    sys.path.insert(0, _CV)
import cvlib as C  # noqa: E402

VERSION = "analogy_grid v1.0"
WEIGHTS = {"E_cell": 0.35, "F_cell": 0.40, "F_transform": 0.25}
SUBS = tuple(WEIGHTS)
FG_TOL = 25.0
MIN_AREA = 150.0              # smallest real object 320 px^2; arrows between cells are ~25 px^2
ROW_SPLIT = 256.0
POS_GOOD_PX, POS_BAD_PX = 10.0, 45.0       # column centres jitter ~4 px; 45 px = half the column pitch minus a half glyph
IOU_GOOD, IOU_BAD = 0.85, 0.60             # same thresholds as the VBVR-CustomEval move scorer (measured there)
CTX_GOOD, CTX_BAD = 0.75, 0.25
SCALE_GOOD = math.log(1.15)
SCALE_MIN_BAD = math.log(1.3)
MOVE_GOOD_PX, MOVE_MIN_BAD_PX = 4.0, 10.0


def detect(img: np.ndarray) -> List[Dict]:
    return C.objects(img, C.clean(C.foreground_mask(img, tol=FG_TOL), open_k=3, close_k=3), min_area=MIN_AREA)


def grid(objs: List[Dict]) -> Dict[str, Dict]:
    top = sorted([o for o in objs if o["centroid"][1] < ROW_SPLIT], key=lambda o: o["centroid"][0])
    bot = sorted([o for o in objs if o["centroid"][1] >= ROW_SPLIT], key=lambda o: o["centroid"][0])
    cells = dict(zip("ABC", top))
    cells.update(zip("DEF", bot))
    return cells


def _dist(a: Dict, b: Dict) -> float:
    return float(np.hypot(a["centroid"][0] - b["centroid"][0], a["centroid"][1] - b["centroid"][1]))


def cell_report(gt: Dict, cd: Optional[Dict]) -> Dict:
    if cd is None:
        return {"present": False, "score": 0.0}
    d = _dist(gt, cd)
    pos = C.band(d, POS_GOOD_PX, POS_BAD_PX)
    iou = C.centred_iou(gt, cd)
    shape = C.band(1.0 - iou, 1.0 - IOU_GOOD, 1.0 - IOU_BAD)
    de = C.dE(gt["lab"], cd["lab"])
    ok = de <= C.SAME_COLOR_DE
    return {"present": True, "dist_px": round(d, 1), "pos": round(pos, 3), "iou": round(iou, 3), "shape": round(shape, 3),
            "colour_dE": round(de, 1), "colour_ok": bool(ok), "score": float(pos * shape * (1.0 if ok else 0.0))}


def transform_score(task: str, gt: Dict[str, Dict], e: Optional[Dict], f: Optional[Dict]) -> Tuple[float, Dict]:
    if e is None or f is None:
        return 0.0, {"reason": "E or F missing"}
    if task == "shape_color_then_scale":
        want = math.log(gt["F"]["area"] / gt["E"]["area"])
        got = math.log(max(f["area"], 1.0) / max(e["area"], 1.0))
        bad = max(abs(want), SCALE_MIN_BAD)
        return C.band(abs(got - want), SCALE_GOOD, bad), {"log_ratio_gt": round(want, 3), "log_ratio_cand": round(got, 3)}
    if task == "shape_color_then_move":
        want = gt["F"]["centroid"][1] - gt["E"]["centroid"][1]
        got = f["centroid"][1] - e["centroid"][1]
        bad = max(abs(want), MOVE_MIN_BAD_PX)
        return C.band(abs(got - want), MOVE_GOOD_PX, bad), {"dy_gt": round(want, 1), "dy_cand": round(got, 1)}
    raise ValueError(task)


def score_images(task: str, inp: np.ndarray, cand: np.ndarray, gt_img: np.ndarray) -> Tuple[Dict[str, float], Dict]:
    dbg: Dict = {"version": VERSION}
    gt_objs, cd_objs = detect(gt_img), detect(cand)
    gt = grid(gt_objs)
    dbg["counts"] = {"gt": len(gt_objs), "cand": len(cd_objs)}
    gate = C.blankness(cand, gt_img)
    bg_de = C.dE(C.to_lab(C.background_bgr(gt_img)), C.to_lab(C.background_bgr(cand)))
    ceiling = gate["ceiling"] if bg_de <= 40.0 else min(gate["ceiling"], 0.15)
    dbg.update(ceiling=ceiling, background_dE=round(bg_de, 1), blankness=gate)
    if len(gt) < 6:
        dbg["error"] = f"GT yielded {len(gt)} cells"
        return {k: 0.0 for k in SUBS}, dbg
    pool = list(cd_objs)
    matched, rep = {}, {}
    for name in "ABCDEF":                    # greedy nearest within POS_BAD_PX, one candidate object per cell
        best = min(pool, key=lambda o: _dist(o, gt[name])) if pool else None
        if best is not None and _dist(best, gt[name]) <= POS_BAD_PX:
            pool.remove(best)
            matched[name] = best
        else:
            matched[name] = None
        rep[name] = cell_report(gt[name], matched[name])
    context = float(np.mean([rep[n]["score"] for n in "ABCD"]))
    ctx_gate = C.band(1.0 - context, 1.0 - CTX_GOOD, 1.0 - CTX_BAD)     # 1 for context >= 0.75, 0 at <= 0.25
    tr, trd = transform_score(task, gt, matched["E"], matched["F"])
    sub = {"E_cell": rep["E"]["score"], "F_cell": rep["F"]["score"], "F_transform": tr}
    mult = ceiling * ctx_gate
    sub = {k: float(np.clip(v, 0.0, 1.0) * mult) for k, v in sub.items()}
    dbg.update(cells=rep, context=round(context, 3), context_gate=round(ctx_gate, 3), transform=trd,
               extra_objects=len(pool))
    return sub, dbg


def overall(sub: Dict[str, float]) -> float:
    return float(sum(sub[k] * w for k, w in WEIGHTS.items()))


def load(path_or_img):
    return C.load(path_or_img, C.WORK) if isinstance(path_or_img, str) else C.resize_to(path_or_img, C.WORK)


def categories(sub: Dict[str, float], dbg: Dict) -> List[str]:
    """Error categories for failure-case selection (from sub-scores and gates)."""
    cats = []
    if dbg.get("ceiling", 1.0) < 1.0:
        cats.append("blank_or_repainted")
    if dbg.get("context_gate", 1.0) < 1.0:
        cats.append("context_damaged")
    cells = dbg.get("cells", {})
    for n, key in (("E", "E_wrong"), ("F", "F_wrong")):
        c = cells.get(n, {})
        if not c.get("present"):
            cats.append(f"{n}_missing")
        elif not c.get("colour_ok", True):
            cats.append(f"{n}_wrong_colour")
        elif c.get("score", 0) < 0.5:
            cats.append(key)
    if sub.get("F_transform", 0) < 0.5 and cells.get("F", {}).get("present"):
        cats.append("F_wrong_transform")
    if dbg.get("extra_objects", 0) > 0:
        cats.append("extra_objects")
    return cats
