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
        raise NotImplementedError('REAL_BACKEND_NOT_IMPLEMENTED: load fixed base, ensure_gen_token/Row, train 8 steps, save last checkpoint, evaluate 50 dev items; no GPU process started')
    result={'simulated':True,'kind':'simulation_receipt','dev':{'synthetic_score':0.5},'checkpoint':None}
    write(trial / 'simulation.json',result)
    return result


def test(trial,config):
    if config['mode']!='simulation':
        raise NotImplementedError('REAL_TEST_NOT_IMPLEMENTED: load selected existing checkpoint, never retrain')
    return {'simulated':True,'synthetic_score':0.5,'checkpoint':None}
