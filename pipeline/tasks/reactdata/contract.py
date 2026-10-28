"""Reactdata candidate files, task checks, pool summary and instruction resources."""
import os
from typing import Dict, List
from experiment.runtime.io import read_json, read_jsonl, sha256_file, sha256_tree, write_json

CAND_FILES = ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml", "proposal.md", "memory.md")

class ReactDataContract:
    files = CAND_FILES

    def __init__(self, root, protocol, protocol_sha, checks, *, store=None, exp, data, manifests):
        self.root, self.P, self.proto_sha = root, protocol, protocol_sha
        self._check = checks
        self.store, self.exp, self.data, self.manifests = store, exp, data, manifests

    def verify_frozen(self, frozen):
        files = [os.path.join(frozen["dir"], f) for f in ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml")]
        return all(os.path.isfile(p) for p in files) and sha256_tree(files) == frozen["code_sha256"]

    def frozen_hash(self, directory):
        return sha256_tree([os.path.join(directory, f) for f in ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml")])

    def candidate_record(self, cid: str, cdir: str, **kw) -> Dict:
        files = [os.path.join(cdir, f) for f in ("databuilder.py", "analysis.py") if os.path.exists(os.path.join(cdir, f))]
        rec = {"cand_id": cid, "code_sha256": sha256_tree(files) if files else None,
               "config_sha256": sha256_file(os.path.join(cdir, "candidate.yaml")) if os.path.exists(os.path.join(cdir, "candidate.yaml")) else None,
               "templates_sha256": sha256_file(os.path.join(cdir, "text_templates.yaml")) if os.path.exists(os.path.join(cdir, "text_templates.yaml")) else None,
               "protocol_sha256": self.proto_sha, **kw}
        if self.store:
            self.store.candidate_record(cid, rec)
        else:
            write_json(os.path.join(self.root, "candidates", cid, "candidate.json"), rec)
        return rec


    def program_sha(self, d: str) -> str:
        files = ("databuilder.py", "analysis.py", "text_templates.yaml") if self.P["search"].get("config_only_is_duplicate") \
            else ("databuilder.py", "analysis.py", "text_templates.yaml", "candidate.yaml")
        return sha256_tree([os.path.join(d, f) for f in files if os.path.exists(os.path.join(d, f))])


    def proposal_errors(self, work: str) -> List[str]:
        """protocol search.proposal: required sections in proposal.md and >= min_item_refs real dev item ids cited
        (a mechanical sign that the per-item results were read; it cannot prove the reasoning)."""
        req = self.P["search"].get("proposal")
        if not req:
            return []
        import re
        p = os.path.join(work, "proposal.md")
        text = open(p).read() if os.path.exists(p) else ""
        errs = [f"proposal.md: missing section '## {h}'" for h in req["sections"] if not re.search(rf"^##\s+{re.escape(h)}", text, re.M | re.I)]
        dev_ids = self.valid_dev_ids()
        cited = {i for i in dev_ids if re.search(r"(?<![\w])" + re.escape(i) + r"(?![\w])", text)}
        if len(cited) < req.get("min_item_refs", 0):
            errs.append(f"proposal.md: cites {len(cited)} dev item ids from /input/dev_items/ (at least {req['min_item_refs']} required "
                        "in the Evidence section, e.g. dev_scale_012)")
        return errs


    def pool_stats(self) -> Dict:
        pv = read_json(os.path.join(self.data, "pool_view.json"))
        return {"example_clips": [pv["clips"][i]["clip_id"] for i in (0, 1, 2)],
                "summary": {"task": pv["task"], "n_clips": len(pv["clips"]), "frames_per_clip": pv["n_frames"], "fps": pv["fps"],
                            "inner_probe_clips": len(read_jsonl(os.path.join(self.manifests, "inner_probe.jsonl"))),
                            "pixel_statistics": read_json(os.path.join(self.manifests, "frame_stats_spec.json")),
                            "dev": "50 shape_color_then_scale (same task, new instances) + 50 shape_color_then_move (related task with a "
                                   "different second rule, not in the training pool: cross-task development feedback); dev score = "
                                   "equal-weight mean of the two task means"}}


    def valid_dev_ids(self):
        return {r["id"] for r in read_jsonl(os.path.join(self.manifests, "dev.jsonl"))}


    def instructions(self):
        return open(os.path.join(self.exp, self.P["search"]["agent"].get("instructions", "BUILDER_AGENT.md"))).read()


    def check_candidate(self, work, check_work):
        return self._check(work, check_work)
