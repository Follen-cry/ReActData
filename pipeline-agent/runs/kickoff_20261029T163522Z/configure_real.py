import sys,json,pathlib,datetime
p=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(p.parents[1]))
from storage import write,digest
from real_score import item
rows={s:[json.loads(l) for l in (p/f'{s}_sources.jsonl').read_text().splitlines()] for s in ('train','dev','test')}
for split in ('dev','test'):
    converted=[dict(id=r['id'],image=[r['f0']],cond_image=r['f0'],target_image=r['final'],conversations=[{'from':'human','value':r['eval_text']},{'from':'gpt','value':'<img>'}]) for r in rows[split]]
    (p/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in converted))
# Scorer self-check uses training examples only; no held-out outcomes are consulted.
checks=[dict(id=r['id'],gt=item(r['f0'],r['final'],r['final'])['score'],copy_f0=item(r['f0'],r['f0'],r['final'])['score']) for r in rows['train'][:20]]
write(p/'scorer_selfcheck.json',checks)
if min(r['gt'] for r in checks)<.95 or max(r['copy_f0'] for r in checks)>.05: raise RuntimeError('scorer selfcheck failed')
lib='/scratch/network/ssd2/junlin/ssl_mllm/internvlu-exp/src/internvlu-train'
(p/'train.yaml').write_text(f'''extends: {lib}/configs/v3/N0.yaml
name: pipeline-agent-real
train:
  max_steps: 8
  seed: 42
  global_batch_size: 16
  per_device_batch_size: 1
loss:
  gen_weight: 0.5
  lm_weight: 0.0
  gen_warmup_steps: 0
optim:
  lr: 0.00002
  gen_decoder_lr: 0.0001
data:
  dynamic_image_size: false
  max_dynamic_patch: 1
gen_image:
  size: 512
  resize_mode: area
eval:
  sets: []
  at_start: false
  diffusion_steps: 20
  all_cfg_scale: 4.5
  part_cfg_scale: 2.0
tracking:
  backends: [jsonl]
checkpoint:
  resume: null
  export_at_end: false
''')
c=dict(mode='real',task='shape_color_then_scale',campaign=str(p/'campaign.json'),estimated_trial_seconds=3600,sources={r['id']:{k:r[k] for k in ('video','f0','final','frame_count','description','eval_text')} for r in rows['train']},external=dict(model_library=lib,weights='/scratch/network/ssd2/junlin/models/InternVL-U',training_config=str(p/'train.yaml'),dev_manifest=str(p/'dev.jsonl'),test_manifest=str(p/'test.jsonl'),prompt_wrapper='/scratch/network/ssd2/junlin/ssl_mllm/VBVR-DataGeneration/target_pred/vbvr_target_pred_prompt_wrap.py',scorer_cvlib='/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/scorers/cvlib.py'))
write(p/'real_config.json',c)
ledger=json.loads((p/'campaign.json').read_text());ledger.update(status='implementing',gpu_allocation=[0,1,2,3]);ledger['previous_stop']={k:ledger.pop(k) for k in ('ended_at','elapsed_seconds','stop_reason') if k in ledger};write(p/'campaign.json',ledger)
print('scorer check passed; configs prepared')
