"""Backend row conversion and distributed process launch."""
import json
from storage import write


def row_specs(dataset):
    """Consume manifest order exactly once; backend must not add another shuffle."""
    rows = []
    for line in (dataset / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        rows.append(dict(id=r['sample_id'],
                         conversations=[{'from':'human','value':r['text']}, {'from':'gpt','value':'<img>'}],
                         images=[str(dataset / p) for p in r['input_images']],
                         target_image=str(dataset / r['target_image']),
                         cond_image=str(dataset / r['input_images'][0])))
    return rows



def launch(trial, config, mode, output, **extra):
    from pathlib import Path
    import sys
    import campaign
    from storage import read
    ledger=read(config['campaign'])
    spec={'config':config,'mode':mode,'output':str(output),**extra}
    path=trial/(mode+'_'+output.name+'_spec.json');write(path,spec)
    args=[sys.executable,'-m','torch.distributed.run','--standalone',
          '--nproc_per_node='+str(len(ledger['gpu_allocation'])),str(Path(__file__).parent/'real_backend.py'),'--spec',str(path)]
    reserve=1800 if output.name=='test' else ledger['report_reserve_seconds']
    timeout=config['training'].get('timeout_seconds') if mode=='train' else config['evaluation'].get('test_timeout_seconds' if output.name=='test' else 'dev_timeout_seconds')
    rc=campaign.run_process(config['campaign'],args,trial/(output.name+'.log'),reserve_seconds=reserve,timeout_seconds=timeout)
    if rc: raise RuntimeError(f'{mode} backend exited {rc}; see {output.name}.log')
