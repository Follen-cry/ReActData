"""Method lifecycle and persistent CLI semantics; fake worker, no GPU claims."""
import json
from pathlib import Path
import time

import pytest

from video_ssl.tools import suite
from video_ssl.tools.session import Session
from video_ssl.tools.evaluation import finalize
from video_ssl.core.runtime import write, read, digest


@pytest.fixture
def method(tmp_path, monkeypatch):
    backend = Path(__file__).resolve().parents[1] / 'vendor' / 'video_ssl_backend'
    plan = dict(pipeline_dir=str(backend), python='unused-worker-python', experiment_config='unused',
                iterations=3, strategy='Try intermediate targets', agent={'timeout_seconds':1800},
                preinit=True, finalize='manual', hashes={}, runner={'mode':'native'}, campaign=str(tmp_path/'campaign.json'))
    write(plan['campaign'], {'deadline':'2099-01-01T00:00:00Z','report_reserve_seconds':5400,'gpu_allocation':[0]})
    monkeypatch.setattr(suite, 'load_plan', lambda _:plan.copy())
    calls = []
    def worker(self, action, directory, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        calls.append((action,directory.name))
        if action == 'resolve':
            write(directory/'backend.json', {'task':'shape_color_then_scale', 'sources':{'v':{'frame_count':4}},
                  'rows_per_trial':1,'build_seed':1000,'training':{},'evaluation':{},
                  'frame_policy':{'mode':'select','target':'select','max_input_frames':4}})
            return {'status':'resolved'}
        if action == 'audit':
            if not (Path(kwargs['dataset'])/'manifest.jsonl').exists():
                raise ValueError('missing manifest')
            return {'status':'valid'}
        if action == 'train':
            checkpoint = directory/'train'/'last.pt'
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_text('fake checkpoint')
            return {'status':'completed','checkpoint':{'path':str(checkpoint),'sha256':digest(checkpoint)}}
        if action == 'preinit':
            return {'status':'completed','dev':{'mean':.1}}
        if action == 'eval':
            return {'status':'completed','dev':{'mean':.4+int(directory.name[-1])*.1}}
        return {'status':'completed','test':{'mean':.6}}
    monkeypatch.setattr(suite.SuiteController,'worker',worker)
    run = suite.SuiteController.create('unused',tmp_path/'run')
    return run, calls


def fresh(run):
    return suite.SuiteController(run.root)


def candidate(run):
    run = fresh(run)
    iteration = run.active()['iteration']
    code = run.root/'workspace'/'curation'
    if iteration > 1:
        with (code/'curate.py').open('a') as f:
            f.write(f'\n# hypothesis {iteration}\n')
        (code/'hypothesis.md').write_text('Test intermediate frame supervision')
    path = run.root/'workspace'/'dataset'
    path.mkdir(exist_ok=True)
    write(path/'image.json', {'fake':'image placeholder'})
    (path/'manifest.jsonl').write_text(json.dumps({'source_video_id':'v','input_frames':[0],
        'target_frame':3 if iteration==1 else 2,'input_images':['image.json'],'target_image':'image.json'}))
    return path


def complete(run):
    path = candidate(run)
    Session(fresh(run)).dispatch('audit',path=str(path))
    Session(fresh(run)).dispatch('submit',path=str(path))
    Session(fresh(run)).dispatch('finetune')
    Session(fresh(run)).dispatch('eval')


def test_continuous_method_lifecycle_idempotence_reports_and_one_final_test(method):
    run,calls=method
    with pytest.raises(ValueError,match='preinit'):
        run.next()
    run.preinit()
    fresh(run).preinit()
    assert sum(a=='preinit' for a,_ in calls)==1
    first=fresh(run).next()
    assert first['iteration']==1
    assert fresh(run).next()['iteration']==1
    complete(run)
    Session(fresh(run)).dispatch('finetune')
    Session(fresh(run)).dispatch('eval')
    assert sum(a=='train' for a,_ in calls)==1
    assert sum(a=='eval' for a,_ in calls)==1
    fresh(run).next(); complete(run)
    fresh(run).next(); complete(run)
    assert fresh(run).next()['status']=='task_done'
    result=finalize(fresh(run))
    assert result['selected']=='shape_color_then_scale_iter3'
    assert finalize(fresh(run))==result
    assert sum(a=='test' for a,_ in calls)==1
    assert (run.root/'results.tsv').read_text(encoding='utf-8').count('keep')==3
    import csv
    records=list(csv.DictReader((run.root/'results.tsv').read_text(encoding='utf-8').splitlines(),delimiter='\t'))
    assert records[0]['status']=='reference' and records[0]['iteration']=='0'
    assert records[0]['code_sha256']==''
    assert (run.root/'README.md').read_text(encoding='utf-8').startswith('# Video SSL Data Construction Experiment')
    assert 'State: completed' in (run.root/'shape_color_then_scale_iter1'/'REPORT.md').read_text(encoding='utf-8')
    delta=read(run.root/'shape_color_then_scale_iter2'/'selection_diff.json')
    assert delta['frame_changes'][0]['after']['target_frame']==2
    assert (run.root/'README.md').exists()


def test_baseline_builder_is_enforced(method):
    run,_=method
    run.preinit(); fresh(run).next()
    path=candidate(run)
    (run.root/'workspace'/'curation'/'curate.py').write_text('changed baseline')
    with pytest.raises(ValueError,match='unchanged'):
        Session(fresh(run)).dispatch('submit',path=str(path))
    assert fresh(run).active()['state']=='active'


def test_accepted_data_tamper_prevents_training(method):
    run,calls=method
    run.preinit(); fresh(run).next()
    Session(fresh(run)).dispatch('submit',path=str(candidate(run)))
    accepted=fresh(run).target_dir()/'curated'/'final_submission'/'image.json'
    accepted.write_text('tampered')
    with pytest.raises(ValueError,match='changed'):
        Session(fresh(run)).dispatch('finetune')
    assert not any(a=='train' for a,_ in calls)


def test_failed_improvement_does_not_restart_but_can_advance(method,monkeypatch):
    run,calls=method
    run.preinit(); fresh(run).next(); complete(run)
    fresh(run).next()
    Session(fresh(run)).dispatch('submit',path=str(candidate(run)))
    original=suite.SuiteController.worker
    def fail(self,action,*args,**kwargs):
        if action=='train':
            raise RuntimeError('GPU failed')
        return original(self,action,*args,**kwargs)
    monkeypatch.setattr(suite.SuiteController,'worker',fail)
    with pytest.raises(RuntimeError):
        Session(fresh(run)).dispatch('finetune')
    with pytest.raises(ValueError,match='never replayed'):
        Session(fresh(run)).dispatch('finetune')
    assert fresh(run).next()['iteration']==3


def test_interrupted_training_cannot_advance_or_replay(method):
    run,_=method
    run.preinit(); fresh(run).next()
    r=fresh(run); r.active()['state']='training'; r.save()
    with pytest.raises(ValueError,match='inspection'):
        fresh(run).next()
    with pytest.raises(ValueError,match='never replayed'):
        Session(fresh(run)).dispatch('finetune')


def test_final_failure_is_not_retried(method,monkeypatch):
    run,_=method
    run.preinit()
    for _ in range(3):
        fresh(run).next(); complete(run)
    fresh(run).next()
    def fail(*args,**kwargs):
        raise RuntimeError('test interrupted')
    monkeypatch.setattr(suite.SuiteController,'worker',fail)
    with pytest.raises(RuntimeError):
        finalize(fresh(run))
    with pytest.raises(ValueError,match='never replayed'):
        finalize(fresh(run))


def test_whole_run_uses_one_agent_session(method,monkeypatch):
    from video_ssl.runner import agent_runner as runner
    run,_=method
    calls=[]
    monkeypatch.setattr(runner,'require_linux',lambda:None)
    def agent(argv,log,**kwargs):
        calls.append(kwargs['prompt'])
        assert 'next/task/build/audit/submit/finetune/eval' in kwargs['prompt']
        fresh(run).preinit()
        for _ in range(3):
            fresh(run).next(); complete(run)
        fresh(run).next()
    monkeypatch.setattr(runner,'process',agent)
    assert runner.launch(run)['status']=='search_completed'
    assert len(calls)==1
    commands=(run.root/'workspace/COMMANDS.md').read_text(encoding='utf-8')
    assert '--project' in commands and '--frozen --no-sync' in commands
    assert '--no-project --no-config' in commands
    assert fresh(run).state['agent_status']=='completed'


def test_docker_wraps_one_agent_with_explicit_mounts_and_no_secret_values(method):
    from video_ssl.runner.agent_runner import docker_command
    run,_=method
    run.plan['runner']={'mode':'docker','image':'configured:test','read_only_mounts':[str(run.root/'config')],
                        'pass_env':['EXPLICIT_TOKEN_NAME']}
    command=docker_command(run,['codex','exec','--json','-'])
    assert command[:2]==['docker','run']
    assert command[-6:]==['--entrypoint','codex','configured:test','exec','--json','-']
    assert '--gpus' in command and 'EXPLICIT_TOKEN_NAME' in command
    assert any(str(run.root) in item and 'type=bind' in item for item in command)


def test_checkpoint_metadata_and_content_remain_bound_to_accepted_trial(method):
    run,_=method
    run.preinit(); fresh(run).next(); complete(run)
    record=fresh(run).target_dir()/'checkpoint.json'
    original=record.read_bytes()
    checkpoint=read(record)
    checkpoint['sha256']='changed'
    write(record,checkpoint)
    with pytest.raises(ValueError,match='record changed'):
        Session(fresh(run)).dispatch('eval')
    record.write_bytes(original)
    Path(read(record)['path']).write_text('changed checkpoint')
    with pytest.raises(ValueError,match='content changed'):
        Session(fresh(run)).dispatch('eval')
