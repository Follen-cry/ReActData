"""Explicit backend boundary; receipts and static Row specifications work on CPU."""
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


def train_and_dev(trial, config):
    write(trial / 'training_contract.json', {
        'training':config['training'], 'evaluation':config['evaluation'],
        'seeds':{k:config[k] for k in ('model_init_seed','train_seed','build_seed','eval_seed')},
        'external':config['external'], 'simulated':config['mode']=='simulation'})
    write(trial / 'rows.json', row_specs(trial / 'dataset'))
    if config['mode'] != 'simulation':
        return real_train(trial, config)
    result={'simulated':True,'kind':'simulation_receipt','dev':{'synthetic_score':0.5},'checkpoint':None}
    write(trial / 'simulation.json',result)
    return result


def test(trial,config):
    if config['mode']!='simulation':
        return real_test(trial, config)
    return {'simulated':True,'synthetic_score':0.5,'checkpoint':None}


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
    rc=campaign.run_process(config['campaign'],args,trial/(output.name+'.log'),reserve_seconds=reserve)
    if rc: raise RuntimeError(f'{mode} backend exited {rc}; see {output.name}.log')


def real_train(trial,config):
    from storage import read, digest
    # Missing resources fail explicitly; never fallback to synthetic scores.
    for k in ('model_library','weights','training_config','dev_manifest','test_manifest'):
        if k not in config['external']: raise ValueError('REAL_RESOURCE_MISSING: '+k)
    launch(trial,config,'train',trial/'train',rows=str(trial/'rows.json'))
    ck=read(trial/'train/checkpoint.json')
    if digest(ck['path'])!=ck['sha256']: raise ValueError('saved checkpoint hash mismatch')
    launch(trial,config,'eval',trial/'dev',checkpoint=ck['path'],checkpoint_sha256=ck['sha256'],manifest=config['external']['dev_manifest'])
    return dict(simulated=False,checkpoint=ck['path'],checkpoint_sha256=ck['sha256'],dev=read(trial/'dev/metrics.json'),
                cost={'train':read(trial/'train/cost.json'),'dev':read(trial/'dev/cost.json')},
                config_sha256=digest(trial/'config.json'),data_sha256=digest(trial/'dataset/manifest.jsonl'),logs=[str(trial/'train.log'),str(trial/'dev.log')])


def real_test(trial,config):
    from storage import read,digest
    result=read(trial/'result.json')['result']
    if digest(result['checkpoint'])!=result['checkpoint_sha256']: raise ValueError('selected checkpoint changed')
    launch(trial,config,'eval',trial/'test',checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],manifest=config['external']['test_manifest'])
    return dict(simulated=False,**read(trial/'test/metrics.json'),checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],cost=read(trial/'test/cost.json'))
