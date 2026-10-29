"""reactdata-v1 training entry (copied from reactdata-v0; max 100 steps): N0 recipe, rows chosen by a sandboxed candidate program, evaluation at the end.

    torchrun --nproc_per_node=4 runner/train_entry.py --spec <run_dir>/run.json

run.json (written by the Runner):
  {"run_id", "run_dir", "candidate_dir", "run_seed", "steps" (0 = evaluation only, e.g. Base),
   "evals": [{"name": "dev", "file": ..., "seeds": [0, 1, 2]}], "max_eval_rows": null, "overrides": []}

Per step k: rank 0 asks the plugin for 16 recipes, checks them (api/recipe.py), broadcasts them; every rank renders
and trains global rows j = m * world + rank (micro-batch m); per-row noise / dropout seeded by (run_seed, k, j) and the
CFG-drop state by (run_seed, k, j), so rows and noise do not depend on the GPU count (numerics do: §11.3). Per-row
stats are gathered to rank 0 -> Analysis.update -> Builder.observe -> Analysis.request_probes; accepted probes run
on all ranks with no gradient and a saved/restored RNG state. Checkpoints (every 50 steps) store model, optimizer,
scheduler, RNG, plugin states, probe scheduler and record offsets.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import API, EXP, TRAIN_SRC, read_json, read_jsonl, sha256_tree, write_json, write_jsonl, implementation_hash  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, API)
sys.path.insert(0, TRAIN_SRC)
sys.path.insert(0, os.path.dirname(TRAIN_SRC))      # internvlu (pipeline package) lives in src/

import plugin_api as PA                                                   # noqa: E402
from recipe import RecipeError, check_batch, check_recipe, check_unique   # noqa: E402
from text_slots import TemplateError                                     # noqa: E402

from internvlu.processing_internvlu import DropCondType                  # noqa: E402
from internvlu_train import factory                                      # noqa: E402
from internvlu_train.checkpoint import CheckpointCallback, CheckpointManager  # noqa: E402
from internvlu_train.cli.train import seed_everything, setup_logging     # noqa: E402
from internvlu_train.config import TrainingPlan, config_to_dict, dump_config, load_config  # noqa: E402
from internvlu_train.data import Collator, ShardedBatchSampler, load_meta  # noqa: E402
from internvlu_train.engine import ConsoleLogger, NonFiniteLossGuard, Tracker, Trainer, init_distributed, shutdown  # noqa: E402
from internvlu_train.evaluation import Predictor                         # noqa: E402
from internvlu_train.evaluation.runner import evaluate_set               # noqa: E402
from internvlu_train.losses import Objective                             # noqa: E402
from internvlu_train.losses.objective import gen_term                    # noqa: E402
from internvlu_train.model import apply_adapters, build_train_model, load_components  # noqa: E402
from internvlu_train.optim import build_optimizer, grad_norms            # noqa: E402
from internvlu_train.preprocess import sample_drop_state                 # noqa: E402

from runner.materialize import Materializer                                     # noqa: E402
from plugin_host import PluginError, PluginHost                          # noqa: E402

logger = logging.getLogger("reactdata")
WARMUP = 6                      # default; v6 job specs carry "warmup_steps" (protocol training.warmup_steps)


def unified_lambda(total: int, warmup: int = WARMUP):
    """lr multiplier of optimizer step k (0-based): (k+1)/warmup for k < warmup, then cosine to ~0 at k = total-1."""
    def f(k: int) -> float:
        if k < warmup:
            return (k + 1) / warmup
        p = (k - warmup + 1) / max(1, total - warmup + 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return f


def crc(s: str) -> int:
    return zlib.crc32(s.encode())


class FailRun(RuntimeError):
    pass


class ReplayHost:
    """replay-data mode: serves the recipes of a recorded rows_realized.jsonl step by step; no plugin, no probes."""

    def __init__(self, rows_path: str) -> None:
        self.by_step = {}
        for r in read_jsonl(rows_path):
            self.by_step.setdefault(r["step"], []).append(r)
        self.calls, self.seconds = 0, 0.0

    def call(self, op, timeout=None, **kw):
        self.calls += 1
        if op == "next_batch":
            rows = sorted(self.by_step[kw["step"]], key=lambda r: r["j"])
            return [{"clip_id": r["clip_id"], "context_frames": r["context_frames"], "template_id": r["template_id"]} for r in rows]
        if op == "after_step":
            return {"summary": {"replay": True}, "requests": []}
        if op == "state":
            return {"builder": {}, "analysis": {}}
        return "ok"

    def close(self):
        pass


def bcast(obj, dist):
    if dist.world_size == 1:
        return obj
    box = [obj]
    tdist.broadcast_object_list(box, src=0)
    return box[0]


class RunCheckpoints(CheckpointManager):
    """Adds run_state.json (plugin states, probe scheduler, pending probe results, cost); required to resume."""

    def list(self):
        return [d for d in super().list() if os.path.isfile(os.path.join(d, "run_state.json"))]

    def save(self, trainer, extra_meta=None):
        st = trainer.run_state() if self.dist.is_main else None
        final = super().save(trainer, extra_meta)
        if self.dist.is_main:
            write_json(os.path.join(final, "run_state.json"), st)
            # the base class pruned before run_state.json existed, so it never saw the new checkpoint as complete and
            # kept the old one too; prune again now (keep_last complete checkpoints)
            import shutil
            for old in self.list()[:-self.keep_last] if self.keep_last > 0 else []:
                shutil.rmtree(old, ignore_errors=True)
        self.dist.barrier()
        return final

    def load(self, trainer, path):
        super().load(trainer, path)
        trainer.load_run_state(read_json(os.path.join(path, "run_state.json")))


class DynTrainer(Trainer):
    def __init__(self, *, run, mat, host, builder, probe_rows, probe_predictor, p_drop, p_all, **kw):
        super().__init__(**kw)
        self.p_drop, self.p_all = p_drop, p_all
        self.run, self.mat, self.host, self.ib = run, mat, host, builder
        self.seed = int(run["run_seed"])
        self.probe_rows = probe_rows                     # inner-probe eval rows (F0 + original text -> final)
        self.probe_ids = sorted(r["id"] for r in probe_rows)
        self.probe_predictor = probe_predictor
        from tasks.reactdata.probes import ProbeController
        self.probes = ProbeController(self.probe_ids, self.mat.template_ids, self.mat.render)
        self.cost = {"rows": 0, "vit_images": 0, "seq_tokens": 0, "train_wall_s": 0.0, "probe_wall_s": 0.0,
                     "plugin_wall_s": 0.0, "clip_uses": {}}
        self.pool = ThreadPoolExecutor(4)
        from training_journal import TrainingJournal
        self.journal = TrainingJournal(self.run["run_dir"])
        self.clip_uses = {}                  # v6 one-epoch rule (api/recipe.py check_unique), rank 0 only

    # -- resumable state
    def run_state(self):
        probes = self.probes.state_dict()
        return {"plugin": self.host.call("state"), "pending": probes["pending"], "sched": probes["sched"], "cost": self.cost,
                "step": self.step, "clip_uses": self.clip_uses}

    def load_run_state(self, st):
        if self.dist.is_main:
            self.host.call("load", state=st["plugin"])
        self.probes.load_state_dict(st)
        self.cost = st["cost"]
        self.clip_uses = dict(st.get("clip_uses", {}))

    # -- one optimizer step on this rank's rows
    def _train_rows(self, rows_items, js, step):
        n = len(rows_items)
        sums, per = {}, []
        weights = {}
        for i, (item, j) in enumerate(zip(rows_items, js)):
            batch = self.collator([item]).to(self.dist.device)
            h = crc(f"{self.seed}:{step}:{j}")
            torch.manual_seed(h)
            torch.cuda.manual_seed(h)
            last = i == n - 1
            sync = contextlib.nullcontext() if last or not hasattr(self.ddp_model, "no_sync") else self.ddp_model.no_sync()
            with sync:
                with self._autocast():
                    out = self.ddp_model(batch)
                    loss = self.objective(out, step)
                (loss.total / n).backward()
            sums["loss"] = sums.get("loss", 0.0) + float(loss.total.detach()) / n
            for k, v in loss.terms.items():
                sums[k] = sums.get(k, 0.0) + float(v) / n
            weights = loss.weights
            per.append([float(loss.terms["gen_loss"]), float(out.sigma[0]), float(batch.input_ids.shape[1]),
                        float(batch.pixel_values.shape[0]) if batch.pixel_values is not None else 0.0])
        norms = grad_norms(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(self.trainable, self.max_grad_norm)
        skipped = not math.isfinite(float(total_norm))
        if skipped:
            self._consecutive_skips += 1
            if self._consecutive_skips > 10:
                raise FloatingPointError("non-finite gradient 10 steps in a row")
        else:
            self._consecutive_skips = 0
            self.optimizer.step()
        lr_used = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self.dist.mean(dict(sums))
        metrics.update({k.replace("_loss", "_weight"): w for k, w in weights.items()})
        metrics["grad_norm"] = float(total_norm)
        metrics["skipped_nonfinite_grad"] = float(skipped)
        metrics.update({f"grad_norm/{k}": v for k, v in norms.items()})
        metrics["lr"] = lr_used                          # the lr this step's update used
        return metrics, per

    def _gather(self, obj):
        if self.dist.world_size == 1:
            return [obj]
        out = [None] * self.dist.world_size
        tdist.all_gather_object(out, obj)
        return out

    # -- probes (all ranks)
    def _run_probes(self, requests, step):
        from tasks.reactdata.gpu_probes import execute_probes
        return execute_probes(requests, step, model=self.model, item_builder=self.ib, materializer=self.mat, collator=self.collator,
            predictor=self.probe_predictor, probe_rows=self.probe_rows, dist=self.dist, gather=self._gather,
            run_dir=self.run["run_dir"], cost=self.cost, autocast=self._autocast)
    def fit(self):
        plan, W, rank = self.plan, self.dist.world_size, self.dist.rank
        self.ddp_model.train()
        for cb in self.callbacks:
            cb.on_train_start(self)
        if self.dist.is_main:
            for name in ("rows_realized.jsonl", "signals.jsonl", "probes.jsonl"):
                self.journal.open(name, self.step)
        t0, step0 = time.time(), self.step
        total = plan.total_steps
        while self.step < total:
            step = self.step
            ts = time.time()
            # 1. recipes from the plugin (rank 0), checked and broadcast
            msg = None
            if self.dist.is_main:
                try:
                    tp = time.time()
                    raw = self.host.call("next_batch", step=step)
                    self.cost["plugin_wall_s"] += time.time() - tp
                    recipes = check_batch(raw, self.mat.clip_ids, self.mat.template_ids)
                    check_unique(recipes, self.clip_uses)
                    texts = [self.mat.render(r) for r in recipes]
                    msg = {"recipes": recipes, "texts": texts}
                except (PluginError, RecipeError, TemplateError) as e:
                    msg = {"error": f"{type(e).__name__}: {e}"[:4000]}
            msg = bcast(msg, self.dist)
            if "error" in msg:
                raise FailRun(f"step {step}: {msg['error']}")
            recipes, texts = msg["recipes"], msg["texts"]
            assert len(recipes) == plan.grad_accum * W
            js = [m * W + rank for m in range(plan.grad_accum)]
            drops = [sample_drop_state(random.Random(f"{self.seed}:{step}:{j}:drop"), self.p_drop, self.p_all) for j in range(len(recipes))]
            items = list(self.pool.map(lambda j: self.ib.build(self.mat.row(recipes[j], f"{step}|{j}"), drops[j]), js))
            if step == 0:     # acceptance evidence (§4.2): the first optimizer step must change the parameters
                watch = [p for p in self.trainable if p.numel() <= 1 << 22][:8]
                before = [p.detach().float().clone() for p in watch]
            metrics, per = self._train_rows(items, js, step)
            if step == 0 and self.dist.is_main:
                delta = sum(float((p.detach().float() - b).norm()) for p, b in zip(watch, before))
                write_json(os.path.join(self.run["run_dir"], "first_step.json"), {
                    "lr_used": metrics["lr"], "gen_weight": metrics.get("gen_weight"), "grad_norm": metrics["grad_norm"],
                    "param_delta_norm_8_tensors": delta, "ok": bool(metrics["lr"] > 0 and metrics.get("gen_weight") == 0.5
                                                                    and math.isfinite(metrics["grad_norm"]) and metrics["grad_norm"] > 0
                                                                    and delta > 0)})
                del before
            gathered = self._gather([(j, p) for j, p in zip(js, per)])
            stats = dict(x for part in gathered for x in part)
            # 2. plugin feedback (rank 0)
            msg = None
            if self.dist.is_main:
                rows = []
                for j, r in enumerate(recipes):
                    loss, sigma, seq, nimg = stats[j]
                    rows.append({"j": j, **r, "loss": loss, "sigma": sigma, "seq_len": int(seq), "vit_images": int(nimg),
                                 "cfg_drop": drops[j].name})
                info = {"step": step, "lr": metrics["lr"], "grad_norm": metrics["grad_norm"], "gen_weight": metrics.get("gen_weight"),
                        "sec_per_step": (time.time() - t0) / max(1, step - step0 + 1), "total_steps": total}
                budget = self.probes.budget()
                try:
                    tp = time.time()
                    res = self.host.call("after_step", step=step, rows=rows, step_info=info, probe_results=self.probes.pending, budget=budget)
                    self.cost["plugin_wall_s"] += time.time() - tp
                    reqs = res["requests"] if self.run.get("probes", True) else []
                    ok, rejected = self.probes.check(reqs, step)
                    msg = {"ok": ok, "rejected": rejected, "summary": res["summary"]}
                except PluginError as e:
                    msg = {"error": f"{type(e).__name__}: {e}"[:4000]}
                for r, text in zip(rows, texts):
                    self.journal.write("rows_realized.jsonl", {"step": step, **r, "text": text})
                self.journal.write("signals.jsonl", {"step": step, "info": info, "metrics": {k: v for k, v in metrics.items() if isinstance(v, float)},
                                              "summary": msg.get("summary"), "error": msg.get("error")})
                for r in recipes:
                    self.cost["clip_uses"][r["clip_id"]] = self.cost["clip_uses"].get(r["clip_id"], 0) + 1
                self.cost["rows"] += len(recipes)
                self.cost["vit_images"] += sum(1 + len(r["context_frames"]) for r in recipes)
                self.cost["seq_tokens"] += sum(int(stats[j][2]) for j in range(len(recipes)))
            msg = bcast(msg, self.dist)
            if "error" in msg:
                raise FailRun(f"step {step}: {msg['error']}")
            self.probes.pending = list(msg["rejected"])
            if msg["ok"]:
                self.probes.accept(msg["ok"], step)
                out = self._run_probes(msg["ok"], step)
                self.probes.pending += out
                if self.dist.is_main:
                    for o in out:
                        self.journal.write("probes.jsonl", {"step": step, **o})
            self.step += 1
            self.cost["train_wall_s"] += time.time() - ts
            metrics["sec_per_step"] = (time.time() - t0) / (self.step - step0)
            self.tracker.log(metrics, self.step)
            if self.dist.is_main:
                self.journal.flush()
                write_json(os.path.join(self.run["run_dir"], "progress.json"), {"step": self.step, "total": total, "time": time.time()})
            for cb in self.callbacks:
                cb.on_step_end(self, metrics)
            if os.environ.get("REACTDATA_TEST_CRASH_AFTER") == str(self.step):     # tests/: simulated crash for resume
                raise SystemExit(17)
        for cb in self.callbacks:
            cb.on_train_end(self)
        self.journal.close()


def run_evals(run, cfg, model, comps, entries, dist, rows_limit):
    """Generation on each eval set at each seed, then rule scoring on rank 0."""
    from score import score_dir
    out = {}
    eb = factory.eval_item_builder(cfg, comps.tokenizer, model.num_image_token, entries)
    for ev in run.get("evals", []):
        rows = read_jsonl(ev["file"])
        if rows_limit:                                   # smoke / pilot: an equal share of rows per task
            tasks = sorted({r["task_name"] for r in rows})
            k = max(1, rows_limit // len(tasks))
            rows = [r for t in tasks for r in [x for x in rows if x["task_name"] == t][:k]]
        path = ev["file"]
        if rows_limit:
            path = os.path.join(run["run_dir"], "eval", f"{ev['name']}_rows.jsonl")
            if dist.is_main:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                write_jsonl(path, rows)
            dist.barrier()
        for s in ev["seeds"]:
            d = os.path.join(run["run_dir"], "eval", ev["name"], f"g{s}")
            done = os.path.join(d, "scores.json")
            if os.path.exists(done):                      # resumed after the evaluation of this seed
                continue
            pred = Predictor(model, comps.tokenizer, eb, comps.scheduler, diffusion_steps=cfg.eval.diffusion_steps, seed=s,
                             all_cfg_scale=cfg.eval.all_cfg_scale, part_cfg_scale=cfg.eval.part_cfg_scale, generation_mode="image")
            t0 = time.time()
            evaluate_set(pred, path, d, dist, size=cfg.gen_image.size, resize_mode=cfg.gen_image.resize_mode)
            if dist.is_main:
                sc = score_dir(d, rows)
                sc_meta = {"name": ev["name"], "seed": s, "mean": sc["mean"], "n": sc["n"], "n_invalid": sc["n_invalid"],
                           "gen_wall_s": time.time() - t0, "world": dist.world_size}
                write_json(os.path.join(d, "eval_meta.json"), sc_meta)
                out[f"{ev['name']}/g{s}"] = sc_meta
                logger.info(f"[eval] {ev['name']} seed {s}: mean {sc['mean']:.4f} (invalid {sc['n_invalid']}) {sc_meta['gen_wall_s']:.0f} s")
            dist.barrier()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    run = read_json(args.spec)
    from experiment.contracts import RunPaths
    paths = RunPaths(**run["resources"]) if "resources" in run else None
    steps = int(run["steps"])
    if not 0 <= steps <= 100:
        raise SystemExit("steps must be in 0..100 (protocol v1)")
    ov = [f"train.output_dir={run['run_dir']}", f"train.max_steps={max(steps, 1)}"] + list(run.get("overrides", []))
    cfg = load_config(os.path.join(paths.exp if paths else EXP, "configs", "train.yaml"), ov, [])
    if cfg.train.global_batch_size != 16 or cfg.loss.lm_weight != 0.0 or cfg.loss.gen_warmup_steps != 0:
        raise SystemExit("config violates the protocol (global batch 16, lm_weight 0, gen weight constant)")
    dist = init_distributed(cfg.dist.backend)
    setup_logging(dist.is_main)
    if dist.is_main:
        os.makedirs(run["run_dir"], exist_ok=True)
        dump_config(cfg, os.path.join(run["run_dir"], "config.yaml"))
    seed_everything(cfg.train.seed)                     # common initialization (LoRA init) for every arm
    t_load = time.time()
    comps = load_components(cfg.model, cfg.data)
    model = build_train_model(comps, cfg)
    apply_adapters(model, cfg.adapters)
    model.to(dist.device)
    entries = load_meta(cfg.data.train_meta)
    load_s = time.time() - t_load
    cost = {"load_wall_s": load_s, "world_size": dist.world_size}
    eval_cost = {}
    trainer = None
    if steps > 0:
        cand = run["candidate_dir"]
        mat = Materializer(cand, paths=paths)
        ib = factory.item_builder(cfg, comps.tokenizer, model.num_image_token, max_dynamic_patch=factory.entry_max_dynamic_patch(cfg, entries[0]))
        plan = TrainingPlan.build(cfg, num_rows=steps * 16, world_size=dist.world_size)
        optimizer = build_optimizer(model, cfg.optim)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, unified_lambda(steps, int(run.get("warmup_steps", WARMUP))))
        ddp_model = DDP(model, device_ids=[dist.local_rank], find_unused_parameters=cfg.dist.find_unused_parameters,
                        static_graph=cfg.dist.static_graph) if dist.world_size > 1 else model
        tracker = Tracker(backends=cfg.tracking.backends, log_dir=run["run_dir"], project=cfg.tracking.project,
                          run_name=run["run_id"], config=config_to_dict(cfg), enabled=dist.is_main)
        ckpt = RunCheckpoints(run["run_dir"], dist, keep_last=cfg.checkpoint.keep_last)
        host = None
        if dist.is_main:
            cy = mat.candidate_yaml
        if dist.is_main and run.get("replay_rows"):
            host = ReplayHost(run["replay_rows"])
        elif dist.is_main:
            host = PluginHost(cand, mat.pool_view_path, os.path.join(run["run_dir"], "plugin_scratch"),
                              os.path.join(run["run_dir"], "plugin.log"), api=paths.api if paths else None)
            host.call("init", timeout=120, builder_config={**(cy.get("builder") or {}), "templates": sorted(mat.template_ids)},
                      analysis_config={**(cy.get("analysis") or {}), "templates": sorted(mat.template_ids)},
                      seed=int(run["run_seed"]), total_steps=steps, inner_probe_ids=mat.probe_ids)
        eb = factory.eval_item_builder(cfg, comps.tokenizer, model.num_image_token, entries)
        probe_pred = Predictor(model, comps.tokenizer, eb, comps.scheduler, diffusion_steps=cfg.eval.diffusion_steps, seed=0,
                               all_cfg_scale=cfg.eval.all_cfg_scale, part_cfg_scale=cfg.eval.part_cfg_scale, generation_mode="image")
        dummy = ShardedBatchSampler(max(dist.world_size, 1) * 4, batch_size=1, seed=0, rank=dist.rank, world_size=dist.world_size)
        trainer = DynTrainer(run=run, mat=mat, host=host, builder=ib, probe_rows=mat.probe_eval_rows, probe_predictor=probe_pred,
                             p_drop=cfg.gen_image.cfg_dropout, p_all=cfg.gen_image.cfg_dropout_all_frac,
                             model=model, ddp_model=ddp_model, objective=Objective.from_config(cfg.loss), optimizer=optimizer,
                             scheduler=scheduler, dataset=None, sampler=dummy, collator=Collator(comps.tokenizer.pad_token_id),
                             plan=plan, dist=dist, tracker=tracker,
                             callbacks=[NonFiniteLossGuard(), ConsoleLogger(cfg.train.log_every),
                                        CheckpointCallback(ckpt, cfg.checkpoint.every_steps, {"name": run["run_id"]})],
                             max_grad_norm=cfg.optim.max_grad_norm, num_workers=0)
        resume = ckpt.resolve(cfg.checkpoint.resume)
        if resume:
            ckpt.load(trainer, resume)
        if dist.is_main:
            write_json(os.path.join(run["run_dir"], "hashes.json"), {
                "candidate_files_sha256": sha256_tree([os.path.join(cand, f) for f in sorted(os.listdir(cand))
                                                       if os.path.isfile(os.path.join(cand, f))]),
                "runner_sha256": implementation_hash(),
                "api_sha256": sha256_tree([os.path.join(API, f) for f in os.listdir(API) if f.endswith(".py")])})
        try:
            trainer.fit()
        except FailRun as e:
            if dist.is_main:
                write_json(os.path.join(run["run_dir"], "FAILED.json"), {"kind": "candidate", "error": str(e), "step": trainer.step})
                logger.error(str(e))
            dist.barrier()
            if host:
                host.close()
            shutdown()
            sys.exit(3)
        tracker.close()
        if host:
            cost["plugin_calls"] = host.calls
            host.close()
        cost.update({k: v for k, v in trainer.cost.items() if k != "clip_uses"})
        uses = trainer.cost["clip_uses"]
        cost.update({"unique_clips": len(uses), "max_clip_uses": max(uses.values()) if uses else 0,
                     "repeated_rows": sum(v - 1 for v in uses.values())})
    eval_cost = run_evals(run, cfg, model, comps, entries, dist, run.get("max_eval_rows"))
    if dist.is_main:
        cost["evals"] = eval_cost
        cost["vit_tokens"] = cost.get("vit_images", 0) * model.num_image_token
        cost["train_gpu_s"] = cost.get("train_wall_s", 0.0) * dist.world_size
        cost["probe_gpu_s"] = cost.get("probe_wall_s", 0.0) * dist.world_size
        cost["eval_gpu_s"] = sum(v["gen_wall_s"] for v in eval_cost.values()) * dist.world_size
        write_json(os.path.join(run["run_dir"], "cost.json"), cost)
        write_json(os.path.join(run["run_dir"], "DONE.json"), {"time": time.time(), "steps": trainer.step if trainer else 0})
    dist.barrier()
    shutdown()


if __name__ == "__main__":
    main()
