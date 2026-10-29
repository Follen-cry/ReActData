"""Final report of a run: <run_root>/report.md + summary.json (also backed up to <exp>/records/<run_id>/)."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, MANIFESTS, read_json, read_jsonl, write_json  # noqa: E402
from runner import stats  # noqa: E402


def _cost(run_dir):
    p = os.path.join(run_dir, "cost.json")
    return read_json(p) if os.path.exists(p) else {}


def build(r, *, paths=None) -> None:
    data = paths.data if paths else DATA
    manifests = paths.manifests if paths else MANIFESTS
    st, S = r.state, r.S
    root = r.root
    jobs_dir = os.path.join(root, "jobs")
    gpu = defaultdict(float)                  # phase -> GPU-hours by kind
    for name in sorted(os.listdir(jobs_dir)):
        if os.path.islink(os.path.join(jobs_dir, name)):      # inherited from the source run: its cost is reported there
            continue
        c = _cost(os.path.join(jobs_dir, name))
        ph = "final" if name.startswith(("formal_", "base_test")) else "search"
        gpu[f"{ph}_train"] += c.get("train_gpu_s", 0) / 3600
        gpu[f"{ph}_probe"] += c.get("probe_gpu_s", 0) / 3600
        gpu[f"{ph}_eval"] += c.get("eval_gpu_s", 0) / 3600
        gpu[f"{ph}_load"] += c.get("load_wall_s", 0) * c.get("world_size", 4) / 3600
    agent_usd = sum((a.get("cost_usd") or 0) for c in st["candidates"].values() for a in c.get("agent_calls", []))
    models = sorted({m for c in st["candidates"].values() for a in c.get("agent_calls", []) for m in (a.get("models") or [])})
    ev = read_jsonl(os.path.join(root, "events.jsonl"))
    phase_t = {}
    for e in ev:
        if e["kind"] == "phase_start":
            phase_t.setdefault(e["phase"], e["time"])
    t0 = st["budget_start"]
    lines = [f"# {r.P.get('experiment', 'reactdata')} {r.run_id}{' (SMOKE: 4 steps, 4 rows, not a result)' if r.smoke else ''}", "",
             f"Wall time from P0 start: {(ev[-1]['time'] - t0) / 3600:.2f} h (budget {r.P['budget']['wall_hours']} h). "
             f"Phase starts (h): " + ", ".join(f"{k} {(v - t0) / 3600:.2f}" for k, v in phase_t.items()), "",
             "## Dev (selection set; selection-biased)", "",
             f"Generation seeds S = {S}. Null comparison of the fixed baseline (S vs {r.S_null}): "
             f"D_null = {st['delta']['D']:+.4f}, CI [{st['delta']['ci_lo']:+.4f}, {st['delta']['ci_hi']:+.4f}] -> delta = {st['delta']['delta']:.4f}"
             " (generation noise only; training-seed noise is not included).", "",
             "Dev = 50 shape_color_then_scale (training task, new instances) + 50 shape_color_then_move (not in the training pool: "
             "cross-task development feedback); dev mean = equal-weight mean of the two task means.", "",
             "| id | role | class | status | dev mean | dev scale | dev move | D vs baseline [95% CI] | D scale | D move | train GPU-h | ViT images | rows / unique clips |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
             f"| Base | untrained | - | - | {st['base_dev']['mean']:.4f} | {st['base_dev']['task_means'].get('dev_scale', 0):.4f} | "
             f"{st['base_dev']['task_means'].get('dev_move', 0):.4f} | | | | | | |"]
    for cid, c in sorted(st["candidates"].items()):
        cs = c.get("cost_summary") or {}
        cmp = c.get("comparison") or {}
        tm = c.get("dev_task_means", {})
        pt = cmp.get("per_task", {})
        cell = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"   # noqa: E731
        lines.append(f"| {cid} | {c['role']} | {c.get('class_label', '-')} | {c['status']} | {cell(c.get('dev_mean'))} | "
                     f"{cell(tm.get('dev_scale'))} | {cell(tm.get('dev_move'))} | "
                     + (f"{cmp['D']:+.4f} [{cmp['ci_lo']:+.4f}, {cmp['ci_hi']:+.4f}] | " if cmp else " | ")
                     + (f"{pt['dev_scale']['D']:+.4f} | {pt['dev_move']['D']:+.4f} | " if 'dev_scale' in pt and 'dev_move' in pt else " | | ")
                     + f"{(cs.get('train_gpu_s') or 0) / 3600:.2f} | {cs.get('vit_images', '')} | {cs.get('rows', '')} / {cs.get('unique_clips', '')} |")
    sel = st["selection"]
    lines += ["", f"**Selection**: {sel['selected']} — {sel['rule']} (delta {sel['delta']:.4f}).", ""]
    summary = {"run_id": r.run_id, "smoke": r.smoke, "delta": st["delta"], "selection": sel, "base_dev": st["base_dev"],
               "candidates": {k: {kk: v.get(kk) for kk in ("role", "status", "class_label", "declared_class", "dev_mean", "comparison",
                                                            "cost_summary", "audit", "error", "parent")} for k, v in st["candidates"].items()}}
    # test
    p3 = st.get("p3", {})
    lines += ["## Test (independent of selection; 1 training seed)", ""]
    if all(v["status"] == "done" for v in p3.values()) and p3:
        it = {k: stats.item_means(v["run_dir"], "test", S) for k, v in p3.items()}
        n = len(next(iter(it.values())))
        lines += [f"{n} shape_color_then_move test datapoints (new instances of the dev move task; not an unseen-task transfer "
                  f"test), generation seeds S = {S}. Copies of vbvr_target_pred/ood_train500 (training data of earlier, unrelated "
                  "checkpoints; see manifests/isolation_report.json): not a fresh never-used test set.", "", "| model | test mean |", "|---|---|"]
        for k, v in it.items():
            lines.append(f"| {k} | {sum(v.values()) / n:.4f} |")
        comps = {}
        if "selected_formal" in it:
            comps["selected_formal - baseline_formal"] = stats.paired(it["selected_formal"], it["baseline_formal"])
            comps["selected_formal - Base"] = stats.paired(it["selected_formal"], it["base_test"])
        comps["baseline_formal - Base"] = stats.paired(it["baseline_formal"], it["base_test"])
        lines += ["", "| comparison | D | 95% bootstrap CI (datapoints) |", "|---|---|---|"]
        for k, v in comps.items():
            lines.append(f"| {k} | {v['D']:+.4f} | [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}] |")
        summary["test"] = {"means": {k: sum(v.values()) / n for k, v in it.items()}, "comparisons": comps}
    else:
        lines.append(f"P3 incomplete: {p3}")
    lines += ["", "## Cost", "", "| item | GPU-h |", "|---|---|"]
    for k in sorted(gpu):
        lines.append(f"| {k} | {gpu[k]:.2f} |")
    inh = st.get("inherited_from")
    own = {k: c for k, c in st["candidates"].items() if not c.get("inherited_from")}
    own_usd = sum((a.get("cost_usd") or 0) for c in own.values() for a in c.get("agent_calls", []))
    if inh:
        lines.insert(2, f"Continuation of run {inh['run']} (candidates {', '.join(inh['candidates'])}, agent session {inh['agent_session']}, "
                        f"linked jobs {inh['linked_jobs']}); the selection pool includes the inherited candidates. Costs below are "
                        "this run's own jobs only.\n")
    lines += ["", f"Builder Agent (this run): {sum(len(c.get('agent_calls', [])) for c in own.values())} calls, models {models}, "
              f"${own_usd:.2f}" + (f" (all candidates incl. inherited: ${agent_usd:.2f})" if inh else "") + ".",
              f"Disk at the end: {st.get('disk_gb')} GB (limit {r.P['budget'].get('disk_gb')}).", "", "## Failures and retries", ""]
    lines += [f"- {json.dumps(f, ensure_ascii=False)[:600]}" for f in st["failures"]] or ["- none"]
    lines += ["", "## Checks", "", f"- first optimizer step: {st.get('first_step_check')}",
              f"- data: {read_json(os.path.join(data, 'BUILD_DONE.json'))['counts']}; isolation ok: "
              f"{read_json(os.path.join(manifests, 'isolation_report.json'))['isolation_ok']}; PNG/mp4 mixing allowed: "
              f"{read_json(os.path.join(manifests, 'data_checks.json'))['png_vs_mp4']['mixing_allowed']} (policy: F0 / VAE / target from PNG, "
              "context frames from mp4)", f"- scorer: {read_json(os.path.join(manifests, 'scorer_validation.json'))['version']}, validation "
              f"passed: {read_json(os.path.join(manifests, 'scorer_validation.json'))['all_passed']}", ""]
    summary["gpu_hours"] = dict(gpu)
    summary["agent"] = {"usd": agent_usd, "models": models}
    summary["failures"] = st["failures"]
    with open(os.path.join(root, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    write_json(os.path.join(root, "summary.json"), summary)


TASK_KEYS = ("dev_scale", "dev_move", "dev_geo")       # v6 task strata (item id prefix)


def _task_rows(root, cid, P) -> str:
    """v6: rows per task (clip-id prefix) of the candidate's screening run."""
    f = os.path.join(root, "jobs", f"{cid}-s{P['training']['screen_seed']}-g4", "rows_realized.jsonl")
    if not os.path.exists(f):
        return ""
    n = Counter(row["clip_id"].split("_", 1)[0] for row in read_jsonl(f))
    return " / ".join(f"{k} {v}" for k, v in sorted(n.items()))


def build_v3(r, *, paths=None) -> None:
    """Design v3 report: training noise, screening (single runs), confirmation (paired seeds), selection, P3 multi-seed test."""
    st = r.state
    root = r.root
    nz = st["noise"]
    jobs_dir = os.path.join(root, "jobs")
    gpu = defaultdict(float)
    for name in sorted(os.listdir(jobs_dir)):
        if os.path.islink(os.path.join(jobs_dir, name)):
            continue
        c = _cost(os.path.join(jobs_dir, name))
        ph = "final" if name.startswith(("formal_", "base_test")) else ("confirm" if any(
            name.endswith(f"-s{sd}-g4") for sd in r.confirm_seeds) and not name.startswith("c000") else "search_and_noise")
        for k in ("train_gpu_s", "probe_gpu_s", "eval_gpu_s"):
            gpu[f"{ph}_{k.split('_')[0]}"] += c.get(k, 0) / 3600
    agent_calls = [a for c in st["candidates"].values() if not c.get("inherited_from") for a in c.get("agent_calls", [])]
    ev = read_jsonl(os.path.join(root, "events.jsonl"))
    t0 = st["budget_start"]
    L = [f"# {r.P.get('experiment', 'reactdata')} {r.run_id} (design v3){' (SMOKE: not a result)' if r.smoke else ''}", "",
         f"Wall time: {(ev[-1]['time'] - t0) / 3600:.2f} h (budget {r.P['budget']['wall_hours']} h).", "",
         "## Training noise (P1)", "",
         f"Fixed baseline, run seeds {nz['seeds']}: dev {', '.join(f'{x:.4f}' for x in nz['run_scores'])}; mean {nz['mean']:.4f}, "
         f"sd_train {nz['sd_train']:.4f} (= delta). Base (untrained) dev {st['base_dev']['mean']:.4f}.", "",
         "## Screening (one run per candidate, seed = screen seed; NOT evidence on its own)", "",
         f"D is against {'the per-item mean of the baseline replicates' if r.P['search'].get('screen_reference') == 'replicate_mean' else 'the baseline run of the same seed'}.", "",
         "| id | parent | class | status | dev | " + " | ".join(t.replace("_", " ") for t in TASK_KEYS) + " | D vs baseline | rows / unique clips | task rows |",
         "|---|---|---|---|---|" + "---|" * len(TASK_KEYS) + "---|---|---|"]
    for cid, c in sorted(st["candidates"].items()):
        cmp, tm, cs = c.get("comparison") or {}, c.get("dev_task_means", {}), c.get("cost_summary") or {}
        f = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"   # noqa: E731
        L.append(f"| {cid} | {c.get('parent', '-')} | {c.get('class_label', '-')} | {c['status']} | {f(c.get('dev_mean'))} | "
                 + " | ".join(f(tm.get(t)) for t in TASK_KEYS) + f" | {('%+.4f' % cmp['D']) if cmp else '-'} | "
                 f"{cs.get('rows', '')} / {cs.get('unique_clips', '')} | {_task_rows(root, cid, r.P)} |")
    L += ["", f"## Confirmation (top {r.top_k}, seeds {r.confirm_seeds}, paired with the baseline replicates of the same seeds)", "",
          "| id | D | per-seed D | SE_train | SE_item | CI (Bonferroni) | " + " | ".join(t + " D" for t in TASK_KEYS) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(TASK_KEYS)]
    for cid, x in st.get("confirm", {}).items():
        if "D" not in x:
            L.append(f"| {cid} | failed: {x.get('error')} | | | | | | |")
            continue
        pt = x.get("per_task", {})
        L.append(f"| {cid} | {x['D']:+.4f} | {', '.join('%+.3f' % d for d in x['D_per_seed'])} | {x['SE_train']:.4f} | {x['SE_item']:.4f} | "
                 f"[{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | " + " | ".join(f"{pt.get(t, {}).get('D', 0):+.4f}" for t in TASK_KEYS) + " |")
    sel = st["selection"]
    L += ["", f"**Selection**: {sel['selected']} — {sel['rule']} (delta {sel['delta']:.4f}).", "", "## Test (P3, formal seeds "
          f"{r.formal_seeds}; existing set with usage history, not fresh)", ""]
    ps = st.get("p3_stats")
    if ps:
        L.append(f"- Base: {ps['base_test']:.4f}; fixed baseline runs: {', '.join(f'{x:.4f}' for x in ps['baseline_formal_runs'])}")
        if "selected_vs_baseline" in ps:
            x = ps["selected_vs_baseline"]
            L.append(f"- {sel['selected']} vs baseline: D {x['D']:+.4f}, CI [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] (training + item noise)")
    else:
        L.append(f"P3 incomplete: {st.get('p3')}")
    L += ["", "## Cost", "", "| item | GPU-h |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in sorted(gpu.items())]
    L += ["", f"Builder Agent: {len(agent_calls)} calls, ${sum((a.get('cost_usd') or 0) for a in agent_calls):.2f}. "
          f"Disk at the end: {st.get('disk_gb')} GB.", "", "## Failures", ""]
    L += [f"- {json.dumps(f_, ensure_ascii=False)[:600]}" for f_ in st["failures"]] or ["- none"]
    with open(os.path.join(root, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    write_json(os.path.join(root, "summary.json"), {"run_id": r.run_id, "design": "v3", "noise": nz, "selection": sel,
                                                     "confirm": st.get("confirm"), "p3": ps, "gpu_hours": dict(gpu)})
