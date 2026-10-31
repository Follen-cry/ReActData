"""Static, fixed-order InternVL-U adapter. No imports from historical pipelines."""
import argparse, contextlib, json, math, os, random, sys, time, zlib
from pathlib import Path
from storage import read, write, digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True); a=ap.parse_args()
    s=read(a.spec); c=s['config']; ext=c['external']; out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[ext['model_library'],str(Path(ext['model_library']).parent)]
    import torch
    import torch.distributed as td
    from torch.nn.parallel import DistributedDataParallel as DDP
    from internvlu_train import factory
    from internvlu_train.config import load_config, dump_config
    from internvlu_train.cli.train import seed_everything
    from internvlu_train.model import load_components, build_train_model, apply_adapters
    from internvlu_train.data import Row, Collator, ensure_gen_token
    from internvlu_train.engine import init_distributed, shutdown
    from internvlu_train.optim import build_optimizer
    from internvlu_train.losses import Objective
    from internvlu_train.preprocess import sample_drop_state
    from internvlu_train.checkpoint.state import trainable_state_dict, load_trainable_state_dict
    from internvlu_train.evaluation import Predictor
    cfg=load_config(ext['training_config'],[],[])
    cfg.train.output_dir=str(out)
    t=c['training']; e=c['evaluation']; steps=t['max_steps']; batch_size=t['global_batch']; warmup=t['warmup_steps']
    if cfg.train.global_batch_size!=batch_size or cfg.train.max_steps!=steps:
        raise ValueError('Resolved training config disagrees with task batch/steps')
    if cfg.loss.gen_weight!=t['gen_weight'] or cfg.loss.lm_weight!=t['lm_weight'] or cfg.loss.gen_warmup_steps!=t['gen_warmup_steps']:
        raise ValueError('Resolved loss disagrees with task contract')
    if cfg.optim.lr!=t['lora_lr'] or cfg.optim.gen_decoder_lr!=t['gen_decoder_lr']:
        raise ValueError('Resolved learning rates disagree with task contract')
    d=init_distributed(timeout_minutes=15); start=time.monotonic()
    if d.device.type!='cuda': raise RuntimeError('CUDA required; no CPU fallback')
    seed_everything(c['model_init_seed'])
    comps=load_components(cfg.model,cfg.data); model=build_train_model(comps,cfg);apply_adapters(model,cfg.adapters);model.to(d.device)
    ib=factory.item_builder(cfg,comps.tokenizer,model.num_image_token,max_dynamic_patch=1)
    if d.is_main: dump_config(cfg,str(out/'resolved_train.yaml'))
    if s['mode']=='train':
        rows=read(s['rows'])
        if len(rows)!=c['rows_per_trial'] or len(rows)!=steps*batch_size:
            raise ValueError('Rows do not match task one-pass recipe')
        rows=[Row(**{**r,'conversations':ensure_gen_token(r['conversations'])},raw=r) for r in rows]
        coll=Collator(comps.tokenizer.pad_token_id); opt=build_optimizer(model,cfg.optim)
        def lr(k):
            return (k+1)/warmup if k<warmup else .5*(1+math.cos(math.pi*min(1,(k-warmup+1)/(steps-warmup+1))))
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lr)
        ddp=DDP(model,device_ids=[d.local_rank],find_unused_parameters=cfg.dist.find_unused_parameters,static_graph=cfg.dist.static_graph) if d.world_size>1 else model
        objective=Objective.from_config(cfg.loss); trainable=[p for p in model.parameters() if p.requires_grad]; ddp.train()
        if batch_size%d.world_size: raise ValueError('GPU count must divide global batch')
        for step in range(steps):
            ts=time.monotonic(); total=0.;js=list(range(d.rank,batch_size,d.world_size));opt.zero_grad(set_to_none=True)
            for k,j in enumerate(js):
                row=rows[step*batch_size+j]
                drop=sample_drop_state(random.Random(f"{c['train_seed']}:{step}:{j}:drop"),cfg.gen_image.cfg_dropout,cfg.gen_image.cfg_dropout_all_frac)
                batch=coll([ib.build(row,drop)]).to(d.device)
                h=zlib.crc32(f"{c['train_seed']}:{step}:{j}".encode());torch.manual_seed(h);torch.cuda.manual_seed(h)
                ctx=ddp.no_sync() if k<len(js)-1 and hasattr(ddp,'no_sync') else contextlib.nullcontext()
                with ctx:
                    with torch.autocast('cuda',dtype=torch.bfloat16): loss=objective(ddp(batch),step)
                    if not torch.isfinite(loss.total): raise FloatingPointError('nonfinite loss')
                    (loss.total/len(js)).backward();total+=float(loss.total.detach())/len(js)
            norm=torch.nn.utils.clip_grad_norm_(trainable,cfg.optim.max_grad_norm)
            if not math.isfinite(float(norm)): raise FloatingPointError('nonfinite gradient')
            used=opt.param_groups[0]['lr'];opt.step();sch.step()
            rec=d.mean({'loss':total,'grad_norm':float(norm)});rec.update(step=step+1,lr=used,seconds=time.monotonic()-ts)
            if d.is_main:
                with open(out/'loss.jsonl','a') as f:f.write(json.dumps(rec)+'\n')
                print(json.dumps(rec),flush=True)
        if d.is_main:
            tmp=out/'last.pt.tmp';torch.save(trainable_state_dict(model),tmp);os.replace(tmp,out/'last.pt')
            write(out/'checkpoint.json',{'path':str(out/'last.pt'),'sha256':digest(out/'last.pt'),'step':steps,'format':'trainable_parameters; frozen weights from fixed base','base':ext['weights']})
        d.barrier()
    else:
        if not s.get('base_only', False):
            if digest(s['checkpoint'])!=s['checkpoint_sha256']: raise ValueError('checkpoint hash mismatch')
            load_trainable_state_dict(model,torch.load(s['checkpoint'],map_location='cpu',weights_only=True))
        model.eval(); pred=Predictor(model,comps.tokenizer,ib,comps.scheduler,diffusion_steps=e['diffusion_steps'],seed=c['eval_seed'],all_cfg_scale=e['all_cfg_scale'],part_cfg_scale=e['part_cfg_scale'],generation_mode='image')
        rows=[json.loads(x) for x in Path(s['manifest']).read_text().splitlines() if x.strip()]
        expected=e.get('test_samples',e['samples']) if out.name=='test' else e['samples']
        if len(rows)!=expected: raise ValueError('Evaluation manifest count disagrees with task')
        (out/'images').mkdir(exist_ok=True)
        for raw in rows[d.rank::d.world_size]:
            result=pred.predict(Row.parse(raw));result['image'].save(out/'images'/f"{raw['id']}.png")
            print('generated',raw['id'],flush=True)
        d.barrier()
        if d.is_main:
            from scorer_loader import load_cvlib
            load_cvlib(ext['scorer_cvlib'])
            from real_score import score
            score(rows,out)
            write(out/'reload.json',{'checkpoint':s['checkpoint'],'sha256':s['checkpoint_sha256'],'loaded':True})
    costs=d.mean({'wall_seconds':time.monotonic()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated()})
    peaks=[None]*d.world_size
    if d.world_size>1: td.all_gather_object(peaks,torch.cuda.max_memory_allocated())
    else: peaks=[torch.cuda.max_memory_allocated()]
    if d.is_main: write(out/'cost.json',{**costs,'world_size':d.world_size,'peak_memory_per_rank':peaks,'simulated':False})
    d.barrier();shutdown()
if __name__=='__main__':main()
