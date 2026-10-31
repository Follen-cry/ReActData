"""Resolve the single task contract and explicit external resource paths."""
from pathlib import Path
import yaml
import json
from storage import digest


def resolve(config_path):
    path = Path(config_path).resolve()
    text=path.read_text(encoding='utf-8-sig')
    c = json.loads(text) if path.suffix.lower()=='.json' else yaml.safe_load(text)
    allowed = {'mode', 'task', 'sources', 'external', 'max_dataset_bytes',
               'max_text_chars', 'simulation_samples', 'simulation_max_trials', 'campaign', 'estimated_trial_seconds', 'frame_policy', 'task_contract'}
    if not isinstance(c, dict) or set(c) - allowed:
        raise ValueError('unknown configuration keys; use the current example config and a new experiment')
    if c.get('mode', 'real') not in ('real', 'simulation'):
        raise ValueError('mode')
    c.setdefault('mode', 'real')
    spec = c.pop('task_contract', None)
    if spec is None:
        # Default contract for explicit standalone CPU fixtures.
        spec = yaml.safe_load((Path(__file__).parent / 'tasks/default.yaml').read_text(encoding='utf-8'))
    else:
        from recipe import validate
        validate(spec)
    if c.get('task') != spec['task']:
        raise ValueError('only shape_color_then_scale is supported')
    c.update({k: v for k, v in spec.items() if k != 'task'})
    from frame_policy import policy
    c['frame_policy'] = policy(c)
    if c['mode'] == 'real' and any(k.startswith('simulation_') for k in c):
        raise ValueError('simulation overrides forbidden in real mode')
    if c['mode'] == 'simulation':
        c['rows_per_trial'] = c.get('simulation_samples', c['rows_per_trial'])
        c['max_trials'] = c.get('simulation_max_trials', c['max_trials'])
    if c['mode'] == 'real':
        if not c.get('campaign') or not isinstance(c.get('estimated_trial_seconds'), (int, float)) or c['estimated_trial_seconds'] <= 0:
            raise ValueError('real mode requires campaign ledger and positive estimated_trial_seconds')
        cp = Path(c['campaign'])
        c['campaign'] = str((path.parent / cp).resolve() if not cp.is_absolute() else cp.resolve())
        if not Path(c['campaign']).is_file(): raise ValueError('campaign ledger missing')
    c.setdefault('max_dataset_bytes', 500000000)
    c.setdefault('max_text_chars', 8000)
    for key in ('rows_per_trial', 'max_trials', 'max_dataset_bytes', 'max_text_chars'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(key)
    c['source_hashes'] = {}
    seen = set()
    def resource(value):
        p = Path(value)
        if not p.is_absolute(): p = path.parent / p
        if not p.is_file(): raise ValueError('missing resource: ' + str(p))
        return str(p.resolve())
    if not isinstance(c.get('sources'), dict) or not c['sources']:
        raise ValueError('source pool required')
    for sid, src in c['sources'].items():
        if not isinstance(sid, str) or not sid or not isinstance(src, dict):
            raise ValueError('source ID and object required')
        required = {'video', 'f0', 'final', 'frame_count', 'description', 'eval_text'}
        if not required <= set(src) or set(src) - required - {'frames'}:
            raise ValueError('source requires video/f0/final/frame_count/description/eval_text')
        if type(src['frame_count']) is not int or src['frame_count'] < 2:
            raise ValueError('frame_count')
        for key in ('description', 'eval_text'):
            if not isinstance(src[key], str) or not src[key].strip() or len(src[key]) > c['max_text_chars']:
                raise ValueError('invalid source text')
        for key in ('video', 'f0', 'final'):
            src[key] = resource(src[key])
        if src['video'] in seen: raise ValueError('source aliases must not repeat a video')
        seen.add(src['video'])
        c['source_hashes'][sid] = {key: digest(src[key]) for key in ('f0', 'final')}
        bank = src.get('frames', {})
        if not isinstance(bank, dict): raise ValueError('frames must map zero-based index strings to PNG paths')
        for index, value in bank.items():
            if not isinstance(index, str) or not index.isdecimal() or str(int(index)) != index or not 0 < int(index) < src['frame_count'] - 1:
                raise ValueError('frame bank keys must be canonical intermediate indices; endpoints use f0/final')
            bank[index] = resource(value)
        if c['frame_policy']['mode'] == 'select':
            c['source_hashes'][sid]['frames'] = {index: digest(value) for index, value in bank.items()}
            c['source_hashes'][sid]['video'] = digest(src['video'])
    if len(seen) < c['rows_per_trial']: raise ValueError('source pool smaller than required rows')
    ext = c.setdefault('external', {})
    if set(ext) - {'model_library', 'weights', 'training_config', 'dev_manifest', 'test_manifest', 'prompt_wrapper', 'scorer_cvlib'}:
        raise ValueError('unknown external dependency')
    for key, value in ext.items():
        p = Path(value)
        ext[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return c
