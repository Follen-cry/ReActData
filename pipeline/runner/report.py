"""Builder Agent input report of one round (Runner-only writer; the agent reads the result under /input).

Contents: protocol summary for the agent, dev results of Base / fixed baseline / earlier candidates (total, sub-scores,
category counts, paired comparison with the baseline, marked as selection-biased), fixed-rule failure cases (images:
F0, generation, dev GT, seed 0), training-loss summaries, probe usage, cost, earlier candidates' code, and a few
training-video example frames. Nothing from test, metadata or the scorer source.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import API, EXP, FRAMES, MANIFESTS, read_json, read_jsonl, write_json  # noqa: E402

CATS = ("blank_or_repainted", "context_damaged", "E_missing", "F_missing", "E_wrong_colour", "F_wrong_colour", "E_wrong",
        "F_wrong", "F_wrong_transform", "extra_objects")
PER_CAT = 1          # per task and category
MAX_FAIL = 12
N_SUCCESS = 1        # per task
EXAMPLE_FRAMES = (0, 12, 20, 28, 36, 44, 49)
CAND_FILES = ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml", "proposal.md")


def link(src: str, dst: str) -> None:
    """Hard link (same file system, no extra space); copy as a fallback."""
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def dev_summary(run_dir: str, seeds: List[int]) -> Dict:
    out = {"per_seed": {}, "by_task": {}, "n_invalid": 0}
    per_task = {}
    for s in seeds:
        sc = read_json(os.path.join(run_dir, "eval", "dev", f"g{s}", "scores.json"))
        out["per_seed"][s] = round(sc["mean"], 4)
        out["n_invalid"] += sc["n_invalid"]
        for t, v in sc["by_task"].items():
            pt = per_task.setdefault(t, {"means": [], "subs": [], "cats": Counter()})
            pt["means"].append(v["mean"])
            pt["subs"].append(v["sub_mean"])
        if s == seeds[0]:
            for x in sc["items"]:
                per_task[x["task"]]["cats"].update(x["categories"])
    for t, pt in per_task.items():
        out["by_task"][t] = {"mean_over_seeds": round(float(np.mean(pt["means"])), 4),
                             "sub_mean": {k: round(float(np.mean([d[k] for d in pt["subs"]])), 4) for k in pt["subs"][0]},
                             "categories_seed0": dict(pt["cats"])}
    out["mean_over_seeds"] = round(float(np.mean(list(out["per_seed"].values()))), 4)
    return out


def failure_cases(run_dir: str, seed: int, dest: str, *, emit=None, manifests=None) -> List[Dict]:
    """Fixed rule: per task and error category the lowest-scoring item (ties: id), at most MAX_FAIL distinct items,
    plus the highest-scoring item of each task."""
    sc = read_json(os.path.join(run_dir, "eval", "dev", f"g{seed}", "scores.json"))
    rows = {r["id"]: r for r in read_jsonl(os.path.join(manifests or MANIFESTS, "dev.jsonl"))}
    picked, used = [], set()
    for task in sorted({x["task"] for x in sc["items"]}):
        items = sorted([x for x in sc["items"] if x["task"] == task], key=lambda x: (x["score"], x["id"]))
        for c in CATS:
            for x in [x for x in items if c in x["categories"] and x["id"] not in used][:PER_CAT]:
                if len(picked) < MAX_FAIL:
                    picked.append((c, x))
                    used.add(x["id"])
        for x in sorted(items, key=lambda x: (-x["score"], x["id"]))[:N_SUCCESS]:
            if x["id"] not in used:
                picked.append(("high_score_example", x))
                used.add(x["id"])
    if emit is None:
        os.makedirs(dest, exist_ok=True)
    emit = emit or link
    out = []
    for cat, x in picked:
        r = rows[x["id"]]
        base = os.path.join(dest, x["id"])
        emit(r["image"][0], base + "_input_F0.png")
        emit(os.path.join(run_dir, "eval", "dev", f"g{seed}", "images", x["id"] + ".png"), base + "_generated.png")
        emit(r["target_image"], base + "_ground_truth.png")
        out.append({"id": x["id"], "task": x["task"], "selected_for": cat, "score": round(x["score"], 4),
                    "sub": {k: round(v, 3) for k, v in x["sub"].items()}, "categories": x["categories"],
                    "images": [os.path.basename(base) + s for s in ("_input_F0.png", "_generated.png", "_ground_truth.png")]})
    return out


def loss_summary(run_dir: str) -> Dict:
    p = os.path.join(run_dir, "signals.jsonl")
    if not os.path.exists(p):
        return {}
    sig = read_jsonl(p)
    loss = [s["metrics"].get("gen_loss") for s in sig if s.get("metrics")]
    win = 20
    return {"gen_loss_mean_per_20_steps": [round(float(np.mean(loss[i:i + win])), 5) for i in range(0, len(loss), win)]}


def dev_items_file(run_dir: str, seeds: List[int], base_mean=None, base_sd=None) -> List[Dict]:
    """Per-item dev results of one run (all items, generation seed(s) S), with the baseline replicate mean / sd."""
    per = {}
    for s in seeds:
        sc = read_json(os.path.join(run_dir, "eval", "dev", f"g{s}", "scores.json"))
        for x in sc["items"]:
            d = per.setdefault(x["id"], {"id": x["id"], "task": x["task"], "scores": [], "sub": {}, "categories": x["categories"]})
            d["scores"].append(round(x["score"], 4))
            for k, v in x["sub"].items():
                d["sub"].setdefault(k, []).append(round(v, 4))
    out = []
    for i in sorted(per):
        d = per[i]
        d["score"] = round(float(np.mean(d["scores"])), 4)
        d["sub"] = {k: round(float(np.mean(v)), 4) for k, v in d["sub"].items()}
        if base_mean:
            d["baseline_mean"] = round(base_mean[i], 4)
            d["baseline_sd"] = round(base_sd[i], 4)
            d["diff_vs_baseline_mean"] = round(d["score"] - base_mean[i], 4)
        out.append(d)
    return out


def analysis_trace(run_dir: str, every: int = 10) -> Dict:
    """What the candidate's own analysis.py reported during training (its summary dicts) and the probe results."""
    sig = os.path.join(run_dir, "signals.jsonl")
    trace = []
    if os.path.exists(sig):
        rows = read_jsonl(sig)
        for r in rows:
            if r["step"] % every == 0 or r["step"] == rows[-1]["step"]:
                summ = r.get("summary")
                txt = json.dumps(summ, ensure_ascii=False) if summ is not None else "null"
                trace.append({"step": r["step"], "gen_loss": round(r["metrics"].get("gen_loss", float("nan")), 5),
                              "summary": summ if len(txt) <= 2000 else txt[:2000] + " ...(truncated)"})
    pr = os.path.join(run_dir, "probes.jsonl")
    probes = read_jsonl(pr) if os.path.exists(pr) else []
    return {"analysis_summary_every_10_steps": trace, "probe_results": probes}


def collect_report(round_dir: str, entries: List[Dict], pool_stats: Dict, notes: List[str], gallery_ids=None,
                 instructions: str = "BUILDER_AGENT.md", baseline_items=None, visibility=None, *, paths=None):
    """entries: [{"id", "role", "run_dir", "seeds", "candidate_dir", "comparison", "checks", "cost", "audit", "status"}]."""
    api, exp = (paths.api, paths.exp) if paths else (API, EXP)
    frames = os.path.join(paths.data, "frames512") if paths else FRAMES
    manifests = paths.manifests if paths else MANIFESTS
    inp = os.path.join(round_dir, "input")
    from experiment.evidence.authorized import AuthorizedEvidence, OutputFile
    outputs = []
    def emit(kind, source, destination):
        outputs.append(OutputFile(os.path.relpath(destination, inp), kind, source))
    def copy(source, destination):
        emit("copy", source, destination)
    def link(source, destination):
        emit("link", source, destination)
    def write_json(destination, data):
        emit("json", data, destination)
    allows = visibility.allows if visibility else lambda section: True
    for f in (("recipe.py", "text_slots.py", "plugin_api.py", "scorer_description.md") if allows("api") else ()):
        copy(os.path.join(api, f), os.path.join(inp, "api", f))
    if allows("instructions"):
        copy(os.path.join(exp, instructions), os.path.join(inp, "BUILDER_AGENT.md"))
    # training-video examples: 3 pool clips, frames EXAMPLE_FRAMES + final
    ex = os.path.join(inp, "train_examples")
    for cid in (pool_stats["example_clips"] if allows("train_examples") else ()):
        for t in EXAMPLE_FRAMES:
            src = os.path.join(frames, cid, "f0.png" if t == 0 else f"mp4_{t:02d}.png")
            link(src, os.path.join(ex, f"{cid}_frame{t:02d}.png"))
        link(os.path.join(frames, cid, "final.png"), os.path.join(ex, f"{cid}_final.png"))
    lines = ["# Round input report", "", "All dev numbers below were used for selection and are therefore selection-biased; "
             "they are not confirmatory evidence. The test set is never shown.", "",
             "Files: `results.json` (all candidates: dev scores, comparisons, costs, training rows, failure cases), "
             "`dev_items/<id>.json` (every dev item of every trained program: score, sub-scores, error categories, and the "
             "baseline replicate mean / sd and the difference to it), `dev_items/baseline_replicates.json`, "
             "`analysis_traces/<id>.json` (what each program's own analysis.py reported every 10 training steps, and its "
             "probe results), `failures/`, `candidates/`, `train_examples/`, `api/`.", ""]
    lines += [f"- {n}" for n in (notes if allows("results") else [])] + [""]
    lines += ["## Training pool", "", "```json", json.dumps(pool_stats["summary"] if allows("train_examples") else {}, indent=1), "```", "",
              "Example frames of three training clips are in `train_examples/` (frame index in the file name; 16 fps, 60 frames).", ""]
    data = []
    bmean, bsd = baseline_items if baseline_items else (None, None)
    if bmean and allows("dev_items"):
        write_json(os.path.join(inp, "dev_items", "baseline_replicates.json"),
                   [{"id": i, "baseline_mean": round(bmean[i], 4), "baseline_sd": round(bsd[i], 4)} for i in sorted(bmean)])
    for e in entries:
        rec = {"id": e["id"], "role": e["role"], "status": e["status"]}
        if e.get("run_dir") and e["status"] == "trained" and allows("results"):
            rec["dev"] = dev_summary(e["run_dir"], e["seeds"])
            rec["training_loss"] = loss_summary(e["run_dir"])
            if allows("dev_items"):
                write_json(os.path.join(inp, "dev_items", f"{e['id']}.json"), dev_items_file(e["run_dir"], e["seeds"], bmean, bsd))
            if allows("analysis_traces"):
                write_json(os.path.join(inp, "analysis_traces", f"{e['id']}.json"), analysis_trace(e["run_dir"]))
            if allows("failures") and (gallery_ids is None or e["id"] in gallery_ids):
                rec["failure_cases"] = failure_cases(e["run_dir"], e["seeds"][0], os.path.join(inp, "failures", e["id"]), emit=link, manifests=manifests)
            rr = os.path.join(e["run_dir"], "rows_realized.jsonl")
            if os.path.exists(rr):
                rows_ = read_jsonl(rr)
                tg = Counter(r.get("target_frame", 59) for r in rows_)
                rec["training_rows"] = {"rows": len(rows_), "unique_clips": len({r["clip_id"] for r in rows_}),
                                        "non_final_target_rows": sum(v for k, v in tg.items() if k != 59),
                                        "target_frame_counts": {str(k): v for k, v in sorted(tg.items())},
                                        "rows_with_context": sum(1 for r in rows_ if r["context_frames"])}
            pr = os.path.join(e["run_dir"], "probes.jsonl")
            rec["probes_run"] = dict(Counter(p["kind"] for p in read_jsonl(pr))) if os.path.exists(pr) else {}
        for k in (("comparison", "cost", "checks_summary", "audit", "error") if allows("results") else ()):
            if e.get(k) is not None:
                rec[k] = e[k]
        if e.get("candidate_dir") and allows("candidates"):
            d = os.path.join(inp, "candidates", e["id"])
            for f in CAND_FILES + ("memory.md",):
                if os.path.exists(os.path.join(e["candidate_dir"], f)):
                    copy(os.path.join(e["candidate_dir"], f), os.path.join(d, f))
        data.append(rec)
    write_json(os.path.join(inp, "results.json"), data)
    lines += ["## Results so far", ""]
    for r in data:
        lines.append(f"### {r['id']} ({r['role']}, {r['status']})")
        if "dev" in r:
            d = r["dev"]
            lines.append(f"- dev score (equal-weight mean of the two task means), mean over generation seeds {list(d['per_seed'])}: "
                         f"**{d['mean_over_seeds']}** (per seed {d['per_seed']}); invalid images {d['n_invalid']}")
            for t, v in d["by_task"].items():
                lines.append(f"  - {t}: {v['mean_over_seeds']}; sub-scores {v['sub_mean']}; error categories (seed "
                             f"{list(d['per_seed'])[0]}): {v['categories_seed0']}")
            if r.get("failure_cases"):
                lines.append(f"- failure cases: `failures/{r['id']}/` (see results.json for their scores)")
            if r.get("training_rows"):
                tr = r["training_rows"]
                lines.append(f"- training rows: {tr['rows']} rows, {tr['unique_clips']} unique clips, {tr['non_final_target_rows']} rows with a "
                             f"non-final target, {tr['rows_with_context']} rows with context frames")
            if r.get("training_loss"):
                lines.append(f"- training flow-matching loss, mean per 20 steps: {r['training_loss']['gen_loss_mean_per_20_steps']}")
        if r.get("comparison"):
            c = r["comparison"]
            lines.append(f"- vs fixed baseline: D = {c['D']:+.4f}, 95% bootstrap CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] "
                         f"(selection-biased dev numbers); per task: "
                         + "; ".join(f"{t} {v['D']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]" for t, v in c.get("per_task", {}).items()))
        if r.get("cost"):
            lines.append(f"- cost: {r['cost']}")
        if r.get("error"):
            lines.append(f"- error: {r['error'][:1500]}")
        if allows("candidates") and any(e["id"] == r["id"] and e.get("candidate_dir") for e in entries):
            lines.append(f"- code: `candidates/{r['id']}/`")
        lines.append("")
    outputs.append(OutputFile("report.md", "text", "\n".join(lines)))
    return AuthorizedEvidence(inp, tuple(outputs))


def write_report(*args, **options):
    """Historical entry point; collection and rendering are now separate operations."""
    from experiment.evidence.authorized import render_authorized
    return render_authorized(collect_report(*args, **options))
