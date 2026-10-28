"""Derived index and bounded backup; neither owns experiment state."""
import os
import shutil
BACKUP_EXT = (".json", ".md", ".diff", ".yaml", ".py", ".txt")

def build_research_index(root, state):
    """Measurements, agent-authored references and protocol decisions stay distinct."""
    records = []
    for cid, c in sorted(state.get("candidates", {}).items()):
        code = c.get("code_dir")
        rd = os.path.join(root, "rounds", f"{c.get('round', 0) or 0:03d}", cid)
        source = code or os.path.join(rd, "work")
        def ref(p):
            return p if p and os.path.isfile(p) else None
        run = c.get("run_dir")
        artifacts = c.get("artifacts", {})
        records.append({
            "candidate_id": cid, "parent_id": c.get("parent"), "status": c.get("status"),
            "agent_authored": {"proposal": ref(os.path.join(source, "proposal.md")),
                              "memory": ref(os.path.join(source, "memory.md")),
                              "diff": ref(os.path.join(rd, "patch.diff")),
                              "explanation": c.get("agent_explanation")},
            "measurements": {"evaluation": artifacts.get("evaluation", os.path.join(run, "eval", "dev") if run else None),
                             "analysis": ref(os.path.join(run, "signals.jsonl")) if run else None,
                             "probes": ref(os.path.join(run, "probes.jsonl")) if run else None,
                             "checks": ref(os.path.join(rd, "checks.json")), "audit": c.get("audit"),
                             "replicates": c.get("replicates", {})},
            "protocol_decisions": {"comparison": c.get("comparison"), "confirmation": state.get("confirm", {}).get(cid, c.get("confirm")),
                                   "rejection_reason": c.get("error"),
                                   "selected": state.get("selection", {}).get("selected") == cid},
        })
    jobs = {}
    jd = os.path.join(root, "jobs")
    if os.path.isdir(jd):
        for name in sorted(os.listdir(jd)):
            d = os.path.join(jd, name)
            if os.path.isdir(d):
                jobs[name] = {f: os.path.join(d, f) for f in ("run.json", "DONE.json", "FAILED.json", "cost.json")
                              if os.path.isfile(os.path.join(d, f))}
    return {
        "schema_version": 1, "candidates": records, "jobs": jobs,
        "selection": state.get("selection"), "frozen": state.get("frozen")}


def backup_records(root, dst_root, lock) -> None:
    """Small records -> <exp>/records/<run_id>/ (git); checkpoints / images are referenced, not copied."""
    if dst_root is None:
        return
    with lock:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in ("checkpoints", "images", "plugin_scratch", "probes", "input", "work_start",
                                                "tensorboard", "agent_home") and not d.startswith("agent_home_") and not d.endswith("_checkwork")]
            for f in fn:
                if not f.endswith(BACKUP_EXT) or f in ("rows_realized.jsonl", "per_row.jsonl", "preds.jsonl"):
                    continue
                src = os.path.join(dp, f)
                if os.path.getsize(src) > 2_000_000:
                    continue
                dst = os.path.join(dst_root, os.path.relpath(src, root))
                if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
