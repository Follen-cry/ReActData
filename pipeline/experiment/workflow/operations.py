"""Shared experiment use cases. Tasks supply jobs, measurements and artifacts only."""
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from experiment.runtime.io import read_json, write_json
from experiment.contracts import JobRequest, TaskAdapter


def prepare_jobs(store, task: TaskAdapter, design, requests):
    snapshot = store.snapshot().data
    specs = {}
    for request in requests:
        cid = request.candidate_id
        code = snapshot["candidates"][cid]["code_dir"] if cid else None
        if request.formal and cid != "c000":
            code = snapshot["frozen"]["dir"]
        specs[request.key] = task.build_spec(store, request, code, design.settings)
    return specs


def execute_many(specs, execute):
    if not specs:
        return {}
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        jobs = {key: pool.submit(execute, spec) for key, spec in specs.items()}
        # Exceptions propagate instead of disappearing inside worker threads.
        return {key: future.result() for key, future in jobs.items()}


def summarize(task: TaskAdapter, spec, seeds):
    directory = read_json(spec)["run_dir"]
    items = task.item_means(directory, "dev", seeds)
    metrics = task.statistics
    groups = sorted({metrics.task_of(i) for i in items})
    return {"run_dir": directory, "dev_mean": metrics.agg(items),
            "dev_task_means": {g: float(sum(v for i, v in items.items() if metrics.task_of(i) == g) /
                                       sum(1 for i in items if metrics.task_of(i) == g)) for g in groups},
            "cost_summary": task.cost_summary(spec)}


def run_baseline(store, design, task: TaskAdapter, execute):
    store.begin_phase("P1")
    s, metrics = design.settings, task.statistics
    if "c000" not in store.state["candidates"]:
        store.commit("baseline_registered", candidates={"c000": {"role": "fixed_baseline", "status": "checked",
                     "class_label": "fixed", "code_dir": task.baseline_dir}})
    task.contract.candidate_record("c000", store.state["candidates"]["c000"]["code_dir"], role="fixed_baseline", class_label="fixed")
    specs = prepare_jobs(store, task, design, design.baseline_plan())
    result = execute_many(specs, execute)
    if any(status != "done" for status in result.values()):
        raise RuntimeError(f"P1 failed {result}: blocking")
    key = f"c000-s{s.base_seeds[0]}" if design.name == "v3" else "c000"
    candidate = {"status": "trained", **summarize(task, specs[key], s.S)}
    base_items = task.item_means(read_json(specs["base"])["run_dir"], "dev", s.S)
    groups = sorted({metrics.task_of(i) for i in base_items})
    updates = {"base_dev": {"run_dir": read_json(specs["base"])["run_dir"], "mean": metrics.agg(base_items),
               "task_means": {g: float(sum(v for i, v in base_items.items() if metrics.task_of(i) == g) /
                                       sum(1 for i in base_items if metrics.task_of(i) == g)) for g in groups}}}
    if design.name == "v3":
        candidate["replicates"] = {str(seed): read_json(specs[f"c000-s{seed}"])["run_dir"] for seed in s.base_seeds}
        replicates = [task.item_means(candidate["replicates"][str(seed)], "dev", s.S) for seed in s.base_seeds]
        updates["baseline_mean_items"] = {i: float(np.mean([r[i] for r in replicates])) for i in sorted(replicates[0])}
        updates["baseline_sd_items"] = {i: float(np.std([r[i] for r in replicates], ddof=1)) if len(replicates) > 1 else 0.0 for i in sorted(replicates[0])}
        noise = {**metrics.training_noise(replicates), "seeds": list(s.base_seeds)}
        updates.update(noise=noise, delta={"delta": noise["delta"], "source": "sd_train of the baseline replicates", **noise})
        write_json(os.path.join(store.root, "noise.json"), noise)
    else:
        updates["delta"] = metrics.null_delta(task.item_means(candidate["run_dir"], "dev", s.S),
                                               task.item_means(candidate["run_dir"], "dev", s.S_null))
        write_json(os.path.join(store.root, "delta.json"), updates["delta"])
    updates["estimates"] = {**store.state.get("estimates", {}),
                            "candidate_job_h_measured": task.estimate_job_hours(task.cost(specs[key]), s, design.name)}
    store.commit("p1_done", updates=updates, candidates={"c000": candidate}, baseline_dev=round(candidate["dev_mean"], 4))


def run_screening(store, design, task: TaskAdapter, resources, protocol, execute, cid):
    c = store.snapshot().data["candidates"][cid]
    if c["status"] != "checked":
        return
    s = design.settings
    request = JobRequest(cid, cid, s.screen_seed, s.steps, "dev", s.S)
    spec = task.build_spec(store, request, c["code_dir"], s)
    status = execute(spec)
    directory = read_json(spec)["run_dir"]
    if status != "done":
        error = read_json(os.path.join(directory, "FAILED.json"))["error"] if status == "candidate_failed" else "infrastructure failure after retry"
        store.commit("candidate_failed", candidates={cid: {"status": "failed", "error": error[:3000], "run_dir": directory}},
                     failures=[{"cand": cid, "kind": status, "error": error[:1500]}], cand=cid, reason=status)
        return
    audit = task.audit(directory)
    write_json(os.path.join(directory, "audit.json"), audit)
    audit = task.audit_summary(audit)
    if audit.get("flag"):
        store.commit("candidate_rejected", candidates={cid: {"status": "rejected", "error": "text-leakage audit flagged",
                     "run_dir": directory, "audit": audit}}, failures=[{"cand": cid, "kind": "audit", "audit": audit}], cand=cid, reason="audit")
        return
    measured = {"status": "trained", "audit": audit, **summarize(task, spec, s.S)}
    snapshot = store.snapshot().data
    reference = protocol["search"].get("screen_reference", "screen_seed_run")
    baseline = snapshot.get("baseline_mean_items") if reference == "replicate_mean" else None
    if not baseline:
        baseline = task.item_means(snapshot["candidates"]["c000"]["run_dir"], "dev", s.S)
    comparison = {**task.statistics.paired(task.item_means(directory, "dev", s.S), baseline), "reference": reference}
    if snapshot.get("noise"):
        comparison["D_over_sd_train"] = comparison["D"] / max(1e-9, snapshot["noise"]["sd_train"])
    measured["comparison"] = comparison
    store.commit("candidate_trained", candidates={cid: measured}, cand=cid, dev_mean=round(measured["dev_mean"], 4), comparison=comparison)
    resources._disk_safe()
    task.contract.candidate_record(cid, c["code_dir"], round=c.get("round"), parent=c.get("parent"),
        class_label=c.get("class_label"), status="trained", run=os.path.basename(directory), dev_mean=measured["dev_mean"])


def run_confirmation(store, design, task: TaskAdapter, resources, search_policy, execute):
    store.begin_phase("CONFIRM")
    snapshot, s = store.snapshot().data, design.settings
    top = search_policy.confirmation_set(snapshot["candidates"], s.top_k)
    specs = prepare_jobs(store, task, design, design.confirmation_plan(top))
    results = execute_many(specs, execute)
    base = [task.item_means(snapshot["candidates"]["c000"]["replicates"][str(seed)], "dev", s.S) for seed in s.confirm_seeds]
    out = {}
    for cid in top:
        if any(results[f"{cid}-s{seed}"] != "done" for seed in s.confirm_seeds):
            out[cid] = {"error": "confirmation runs failed", "status": {seed: results[f"{cid}-s{seed}"] for seed in s.confirm_seeds}}
        else:
            cand = [task.item_means(read_json(specs[f"{cid}-s{seed}"])["run_dir"], "dev", s.S) for seed in s.confirm_seeds]
            out[cid] = task.statistics.confirm_compare(cand, base, snapshot["noise"]["sd_train"], m=max(1, len(top)))
    write_json(os.path.join(store.root, "confirm.json"), {"top_k": top, "seeds": list(s.confirm_seeds), "results": out})
    store.commit("confirm_done", updates={"confirm_set": top, "confirm": out},
                 candidates={cid: {"confirm": r} for cid, r in out.items() if "D" in r}, top=top)
    resources._disk_safe()


def run_selection(store, design, task: TaskAdapter, selection_policy):
    store.begin_phase("SELECT")
    st = store.snapshot().data
    if "selection" in st:
        return
    if design.name == "v3":
        # Pass ordinary JSON data to the decision function, never mutable runtime state.
        decision = selection_policy.select(store.state.get("confirm", {}).copy(), st["noise"]["delta"])
    else:
        s, candidates = design.settings, st["candidates"]
        baseline = candidates["c000"]
        bgpu = baseline["cost_summary"]["train_gpu_s"] or 1.0
        pool = [{"id": cid, "class_label": c["class_label"], "compute_ratio": (c["cost_summary"]["train_gpu_s"] or bgpu) / bgpu,
                 "items": task.item_means(c["run_dir"], "dev", s.S)} for cid, c in sorted(candidates.items()) if cid != "c000" and c["status"] == "trained"]
        decision = selection_policy.select(pool, {"id": "c000", "items": task.item_means(baseline["run_dir"], "dev", s.S)}, st["delta"]["delta"])
        decision["note"] = "dev comparisons are selection-biased; only the formal retraining evaluated on test is confirmatory (1 training seed)"
    write_json(os.path.join(store.root, "selection.json"), decision)
    store.commit("selected", updates={"selection": decision}, selected=decision["selected"], rule=decision["rule"], delta=decision["delta"])


def run_final(store, design, task: TaskAdapter, resources, execute):
    task.freeze(store, store.state["selection"]["selected"])
    store.begin_phase("P3")
    specs = prepare_jobs(store, task, design, design.final_plan(store.state["selection"]["selected"]))
    results = execute_many(specs, execute)
    updates = {"p3": {k: {"status": status, "run_dir": read_json(specs[k])["run_dir"]} for k, status in results.items()}}
    if design.name == "v3" and all(status == "done" for status in results.values()):
        selected, s = store.state["selection"]["selected"], design.settings
        runs = lambda cid: [task.item_means(read_json(specs[f"{cid}-s{seed}"])["run_dir"], "test", design.test_seeds) for seed in s.formal_seeds]
        baseline = runs("c000")
        stats = {"base_test": task.statistics.agg(task.item_means(read_json(specs["base_test"])["run_dir"], "test", design.test_seeds)),
                 "baseline_formal_runs": [task.statistics.agg(r) for r in baseline]}
        if selected != "c000":
            stats["selected_vs_baseline"] = task.statistics.confirm_compare(runs(selected), baseline, store.state["noise"]["sd_train"], m=1)
        updates["p3_stats"] = stats
    store.commit("p3_done", updates=updates, results=results)
    resources._disk_safe()
