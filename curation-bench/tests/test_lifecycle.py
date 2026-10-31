"""Upstream lifecycle interfaces with the video backend; no GPU execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import pytest
from test_suite import method, fresh, complete, candidate
from video_ssl.tools.suite import SuiteController
from video_ssl.tools.session import Session
from video_ssl.tools.evaluation import finalize
from video_ssl.core.runtime import read,write,digest


def budget(run, attempts=0, limit=7):
    c=read(run.plan['campaign'])
    c.update(training_budget=limit,training_attempts=[dict(status='completed') for _ in range(attempts)])
    write(run.plan['campaign'],c)


def test_factory_reuses_existing_suite_and_exposes_upstream_api(method,monkeypatch):
    run,calls=method
    monkeypatch.chdir(run.root.parent)
    loaded=SuiteController.from_state_file(run.root/'suite_state.json')
    assert loaded.root==run.root
    same=SuiteController.from_prompt('unused',output_root=run.root)
    assert same.root==run.root
    assert sum(a=='resolve' for a,_ in calls)==1
    assert loaded.get_run_plan()['total_targets']==3
    loaded.preinit(); loaded.begin_next_target()
    assert loaded.get_active_session().summarize_task()['iteration']==1


def test_session_receipt_survives_cli_reconstruction_and_cached_stages(method):
    run,calls=method
    run.preinit(); run.next(); complete(run)
    receipt=read(fresh(run).target_dir()/'session_state.json')
    assert receipt['event_seq']==4 and receipt['target']['state']=='completed'
    assert fresh(run).prepare_resume(mutate=False)['active_target']['state']=='completed'
    session=fresh(run).get_active_session()
    session.dispatch('finetune'); session.dispatch('eval')
    assert sum(a=='train' for a,_ in calls)==1
    assert sum(a=='eval' for a,_ in calls)==1


def test_resume_trained_candidate_only_evaluates(method):
    run,calls=method
    run.preinit(); run.next()
    session=fresh(run).get_active_session()
    session.dispatch('submit',path=str(candidate(run)))
    fresh(run).get_active_session().dispatch('finetune')
    fresh(run).prepare_resume()
    assert fresh(run).active()['state']=='trained'
    fresh(run).get_active_session().dispatch('eval')
    assert sum(a=='train' for a,_ in calls)==1


@pytest.mark.parametrize('state',['training','evaluating','submitting'])
def test_resume_never_resets_uncertain_stage(method,state):
    run,_=method
    run.preinit(); run.next()
    edited=fresh(run); edited.active()['state']=state; edited.save()
    with pytest.raises(ValueError,match='inspection'):
        fresh(run).prepare_resume()


def test_resume_rejects_session_tampering_and_live_agent(method):
    run,_=method
    run.preinit(); run.next()
    receipt=fresh(run).target_dir()/'session_state.json'
    original=receipt.read_bytes(); receipt.write_text('{}')
    with pytest.raises(ValueError,match='Session state changed'):
        fresh(run).prepare_resume()
    receipt.write_bytes(original)
    r=fresh(run); r.state.update(agent_status='running',agent_pid=os.getpid(),agent_host=socket.gethostname());r.save()
    with pytest.raises(ValueError,match='still running'):
        fresh(run).prepare_resume()


def test_discard_restores_best_code_and_keeps_candidate_archive(method,monkeypatch):
    run,_=method
    run.preinit(); run.next(); complete(run)
    baseline=(fresh(run).target_dir()/'code'/'curate.py').read_bytes()
    fresh(run).next()
    original=SuiteController.worker
    def worse(self,action,*args,**kwargs):
        if action=='eval': return dict(status='completed',dev=dict(mean=.1))
        return original(self,action,*args,**kwargs)
    monkeypatch.setattr(SuiteController,'worker',worse)
    complete(run)
    losing=fresh(run).target_dir()
    assert (losing/'code'/'curate.py').read_bytes()!=baseline
    context=fresh(run).next()
    assert context['base_target']=='shape_color_then_scale_iter1'
    assert (run.root/'workspace'/'curation'/'curate.py').read_bytes()==baseline
    assert (losing/'code'/'curate.py').read_bytes()!=baseline
    assert not (run.root/'workspace'/'curation'/'hypothesis.md').exists()
    assert list((run.root/'workspace_history').iterdir())
    complete(run)
    delta=read(fresh(run).target_dir()/'selection_diff.json')
    assert delta['frame_changes'][0]['before']['target_frame']==3
    assert 'shape_color_then_scale_iter1' in (fresh(run).target_dir()/'changes.diff').read_text()


def test_extension_reuses_finished_work_and_reserves_budget(method):
    run,calls=method
    run.preinit()
    for _ in range(3): fresh(run).next();complete(run)
    fresh(run).next()
    budget(run,attempts=3,limit=4)
    existing=(run.root/'shape_color_then_scale_iter1'/'train_result.json').read_bytes()
    extended=fresh(run).extend(4)
    assert extended['total_targets']==4
    assert fresh(run).plan['iterations']==4
    assert fresh(run).next()['iteration']==4
    assert (run.root/'shape_color_then_scale_iter1'/'train_result.json').read_bytes()==existing
    assert sum(a=='train' for a,_ in calls)==3


def test_failed_extension_does_not_change_saved_plan(method):
    run,_=method;budget(run,limit=3)
    original=(run.root/'config'/'plan.json').read_bytes()
    with pytest.raises(ValueError,match='Insufficient shared'):
        fresh(run).extend(4)
    assert (run.root/'config'/'plan.json').read_bytes()==original
    assert len(fresh(run).state['targets'])==3


def test_cannot_extend_after_test(method):
    run,_=method;run.preinit()
    for _ in range(3): fresh(run).next();complete(run)
    fresh(run).next();finalize(fresh(run))
    with pytest.raises(ValueError,match='after final test'):
        fresh(run).extend(4)


def test_resumed_agent_uses_new_log_directory_without_retraining(method,monkeypatch):
    from video_ssl.runner import agent_runner
    run,calls=method
    monkeypatch.setattr(agent_runner,'require_linux',lambda:None)
    invocations=[]
    def agent(argv,log,**kwargs):
        invocations.append(log)
        if len(invocations)==1:
            fresh(run).preinit();fresh(run).next();complete(run)
            raise RuntimeError('Agent transport disconnected')
        for _ in range(2): fresh(run).next();complete(run)
        fresh(run).next()
    monkeypatch.setattr(agent_runner,'process',agent)
    with pytest.raises(RuntimeError,match='disconnected'): agent_runner.launch(run)
    restored=fresh(run);restored.prepare_resume();agent_runner.launch(restored)
    assert invocations[0]!=invocations[1]
    assert sum(a=='train' for a,_ in calls)==3
    assert fresh(run).state['agent_status']=='completed'


def test_multitask_runs_separate_agent_sessions_and_workspaces(method,monkeypatch,tmp_path):
    from video_ssl.runner import agent_runner
    run,calls=method;budget(run)
    plans={}
    for name in ('first','second'):
        p=deepcopy(run.plan);p.update(task_id=name,iterations=2)
        plans[name]=p
    batch=SuiteController.create('unused',tmp_path/'batch',plan=dict(kind='multi_task',tasks=plans,task_order=list(plans),iterations=2,finalize='manual',campaign=run.plan['campaign']))
    monkeypatch.setattr(agent_runner,'require_linux',lambda:None)
    workspaces=[]
    def agent(argv,log,**kwargs):
        child=SuiteController(kwargs['cwd'].parent)
        workspaces.append(kwargs['cwd'])
        assert os.environ['VIDEO_SSL_CURRENT_TASK']==child.plan['task_id']
        assert not (kwargs['cwd']/'curation'/'hypothesis.md').exists()
        child.preinit()
        for _ in range(2): fresh(child).begin_next_target();complete(child)
        fresh(child).begin_next_target()
    monkeypatch.setattr(agent_runner,'process',agent)
    assert agent_runner.launch(batch)['status']=='search_completed'
    assert len(set(workspaces))==2
    assert batch.get_run_plan()['completed']==4
    result=finalize(batch)
    assert set(result['tasks'])=={'first','second'}
    assert sum(a=='test' for a,_ in calls)==2


def test_multitask_extension_counts_other_tasks_pending_budget(method,tmp_path):
    run,_=method;budget(run,limit=4)
    plans={}
    for name in ('first','second'):
        p=deepcopy(run.plan);p.update(task_id=name,iterations=2);plans[name]=p
    batch=SuiteController.create('unused',tmp_path/'batch',plan=dict(kind='multi_task',tasks=plans,task_order=list(plans),iterations=2,finalize='manual',campaign=run.plan['campaign']))
    with pytest.raises(ValueError,match='Insufficient shared'):
        batch.children()[0].extend(3)


def test_host_docker_worker_uses_container_adapter(method,monkeypatch):
    from video_ssl.backends import video_ssl
    run,_=method
    run.plan['runner']=dict(mode='docker',image='configured:test')
    args=[]
    def process(argv,log,**kwargs):
        args.extend(argv);log.parent.mkdir(parents=True,exist_ok=True)
        log.write_text(json.dumps(dict(status='completed',test=dict(mean=.5))))
    monkeypatch.setattr(video_ssl,'process',process)
    result=video_ssl.VideoSSLBackend(run).invoke('test',run.root/'final',checkpoint=run.root/'checkpoint.json')
    assert result['status']=='completed'
    assert args[:2]==['docker','run']
    assert 'VIDEO_SSL_CONTAINER=1' in args


def test_multitask_finalization_cannot_test_an_incomplete_queue(method,tmp_path):
    run,calls=method;budget(run)
    plans={}
    for name in ('first','second'):
        p=deepcopy(run.plan);p.update(task_id=name,iterations=1);plans[name]=p
    batch=SuiteController.create('unused',tmp_path/'batch',plan=dict(kind='multi_task',tasks=plans,task_order=list(plans),iterations=1,finalize='manual',campaign=run.plan['campaign']))
    with pytest.raises(ValueError,match='all task searches'):
        finalize(batch)
    assert not any(action=='test' for action,_ in calls)
    first,second=batch.children()
    first.state['final']='completed';first.save()
    with pytest.raises(ValueError,match='any task final test'):
        second.extend(2)


def test_resume_refuses_operation_lock_without_removing_it(method):
    run,_=method
    lock=run.root/'operation.lock';lock.write_text('inspect owner')
    with pytest.raises(ValueError,match='lock exists'):
        fresh(run).prepare_resume()
    assert lock.read_text()=='inspect owner'


def test_stale_session_objects_cannot_repeat_training_or_final_test(method):
    run,calls=method;run.preinit();run.next()
    fresh(run).get_active_session().dispatch('submit',path=str(candidate(run)))
    stale=fresh(run).get_active_session()
    fresh(run).get_active_session().dispatch('finetune')
    stale.dispatch('finetune')
    assert sum(action=='train' for action,_ in calls)==1
    fresh(run).get_active_session().dispatch('eval')
    for _ in range(2):fresh(run).next();complete(run)
    fresh(run).next()
    stale_suite=fresh(run)
    assert finalize(fresh(run))==finalize(stale_suite)
    assert sum(action=='test' for action,_ in calls)==1


def test_stale_next_does_not_reset_active_strategy(method):
    run,_=method;run.preinit()
    stale=fresh(run)
    first=fresh(run).begin_next_target()
    second=stale.begin_next_target()
    assert first['key']==second['key']
    assert first['strategy_started_at']==second['strategy_started_at']


def test_init_selects_existing_completed_suite_without_hiding_its_status(method,monkeypatch,capsys):
    from video_ssl.cli import main
    run,_=method;run.preinit()
    for _ in range(3):fresh(run).next();complete(run)
    fresh(run).next()
    monkeypatch.chdir(run.root.parent)
    assert main(['init','--prompt','unused','--root',str(run.root)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='search_completed'
    assert Path('.video_ssl_suite').read_text()==str(run.root)
