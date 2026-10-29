"""GPU probe effects; RNG/model-mode restoration stays in one explicit scope."""
import os
import time
import zlib
import torch
from experiment.runtime.io import read_jsonl, write_jsonl
from internvlu.processing_internvlu import DropCondType
from internvlu_train.evaluation.runner import evaluate_set
from internvlu_train.losses.objective import gen_term

def crc(s: str) -> int:
    return zlib.crc32(s.encode())          # same per-clip noise seed as runner/train_entry.crc


def execute_probes(reqs, step, *, model, item_builder, materializer, collator, predictor, probe_rows, dist, gather, run_dir, cost, autocast):
    results = []
    t0 = time.time()
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        for q in reqs:
            if q["kind"] == "loss":
                mine = []
                model.eval()
                with torch.no_grad():
                    for idx in range(dist.rank, len(q["items"]), dist.world_size):
                        it = q["items"][idx]
                        item = item_builder.build(materializer.row(it, f"probe|{it['clip_id']}"), DropCondType.none)
                        batch = collator([item]).to(dist.device)
                        h = crc(f"probe:{it['clip_id']}")
                        torch.manual_seed(h)
                        torch.cuda.manual_seed(h)
                        with autocast():
                            out = model(batch)
                        mine.append((idx, {**it, "loss": float(gen_term(out)), "sigma": float(out.sigma[0])}))
                model.train()
                allr = sorted(x for part in gather(mine) for x in part)
                results.append({"kind": "loss", "step": step, "items": [r for _, r in allr]})
            else:
                d = os.path.join(run_dir, "probes", f"gen_step{step:03d}")
                path = os.path.join(d, "rows.jsonl")
                if dist.is_main:
                    os.makedirs(d, exist_ok=True)
                    want = set(q["clip_ids"])
                    write_jsonl(path, [r for r in probe_rows if r["id"] in want])
                dist.barrier()
                evaluate_set(predictor, path, d, dist, size=512, resize_mode="area")
                items = []
                if dist.is_main:
                    from runner.score import score_dir
                    sc = score_dir(d, read_jsonl(path))
                    items = [{"clip_id": x["id"], "task": x["task"], "score": x["score"], "sub": x["sub"], "valid": x["valid"]} for x in sc["items"]]
                results.append({"kind": "generate", "step": step, "items": items})
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        model.train()
    cost["probe_wall_s"] += time.time() - t0
    return results
