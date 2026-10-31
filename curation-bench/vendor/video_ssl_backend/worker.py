"""Separated method operations; executed with the configured InternVL-U Python."""
import argparse
import json
from pathlib import Path

from storage import read, write, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['resolve', 'audit', 'train', 'eval', 'test', 'preinit'])
    p.add_argument('--config', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--dataset')
    p.add_argument('--checkpoint')
    a = p.parse_args()
    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.action == 'resolve':
        from task import resolve
        write(root / 'backend.json', resolve(a.config))
        return {'status':'resolved'}
    c = read(a.config)
    if a.action == 'audit':
        from dataset import validate
        return {'status':'valid', 'samples':len(validate(a.dataset, c)),
                'scope':'source whitelist, unique videos, frozen frames/pixels/text and temporal ordering; not semantic deduplication'}
    import training
    if a.action == 'train':
        import campaign
        from dataset import validate
        validate(a.dataset, c)
        campaign.register(c['campaign'], root, root.name, 'train', c['estimated_trial_seconds'])
        try:
            campaign.update(c['campaign'], root, 'train', 'running')
            write(root / 'rows.json', training.row_specs(Path(a.dataset)))
            write(root / 'training_contract.json', {k:c[k] for k in ['training','evaluation','frame_policy','model_init_seed','train_seed','build_seed','eval_seed']})
            training.launch(root, c, 'train', root / 'train', rows=str(root / 'rows.json'))
            ck = read(root / 'train' / 'checkpoint.json')
            if digest(ck['path']) != ck['sha256']:
                raise ValueError('checkpoint hash mismatch')
            campaign.update(c['campaign'], root, 'train', 'completed')
            return {'status':'completed', 'checkpoint':ck}
        except BaseException:
            campaign.update(c['campaign'], root, 'train', 'failed')
            raise
    if a.action == 'preinit':
        training.launch(root, c, 'eval', root / 'dev', base_only=True, checkpoint=None,
                        checkpoint_sha256=None, manifest=c['external']['dev_manifest'])
        return {'status':'completed', 'dev':read(root / 'dev' / 'metrics.json'), 'base_only':True}
    ck = read(a.checkpoint)
    if digest(ck['path']) != ck['sha256']:
        raise ValueError('checkpoint hash mismatch')
    split = 'test' if a.action == 'test' else 'dev'
    training.launch(root, c, 'eval', root / split, checkpoint=ck['path'], checkpoint_sha256=ck['sha256'],
                    manifest=c['external'][split + '_manifest'])
    return {'status':'completed', split:read(root / split / 'metrics.json'), 'checkpoint':ck}


if __name__ == '__main__':
    try:
        result = main()
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({'status':'failed','error':str(exc)}))
        raise SystemExit(1)
