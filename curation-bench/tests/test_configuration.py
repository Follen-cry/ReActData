"""Configuration boundaries and real CPU initialization of the composed contract."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import yaml

from video_ssl.core.layout import load_plan
from video_ssl.core.runtime import read,write,verify
from video_ssl.tools.suite import SuiteController

REPO=Path(__file__).resolve().parents[1]


def yaml_write(path,value):
    path.write_text(yaml.safe_dump(value,allow_unicode=True,sort_keys=False),encoding='utf-8')


@pytest.fixture
def configured(tmp_path):
    data=tmp_path/'data'; data.mkdir()
    frames=[]
    for i,color in enumerate(['red','green','blue']):
        p=data/f'{i}.png'; Image.new('RGB',(512,512),color).save(p); frames.append(p.name)
    rows=[]
    for i in range(128):
        video=data/f'v{i}.gif'; video.write_bytes(b'CPU fixture; source provenance test only')
        rows.append({'source_video_id':f'v{i:03d}','video':video.name,'f0':frames[0],'final':frames[2],
                     'frame_count':3,'description':'Change a shape','eval_text':'<image>\nGenerate final.', 'frames':{'1':frames[1]}})
    (data/'train_sources.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    for name in ['meta.json','wrapper.py','scorer.py']:
        (tmp_path/name).write_text('{}',encoding='utf-8')
    for name in ['dev.jsonl','test.jsonl']:
        (tmp_path/name).write_text('\n'.join(json.dumps({'id':f'{name}_{i}','image':['data/0.png'],
            'cond_image':'data/0.png','target_image':'data/2.png','conversations':[{'from':'human','value':'<image>\nGenerate final.'},{'from':'gpt','value':'<img>'}]}) for i in range(50)),encoding='utf-8')
    (tmp_path/'library').mkdir(); (tmp_path/'weights').mkdir()
    profile={'python':sys.executable,'dataset_path_map':{'shape_color_train':str(data)},
             'model_path_map':{'internvlu_base':str(tmp_path/'weights')},
             'eval_path_map':{'shape_color_dev':str(tmp_path/'dev.jsonl'),'shape_color_test':str(tmp_path/'test.jsonl')},
             'resource_path_map':{'internvlu_train':str(tmp_path/'library'),'prompt_wrapper':str(tmp_path/'wrapper.py'),
                                  'scorer_cvlib':str(tmp_path/'scorer.py'),'training_meta':str(tmp_path/'meta.json')},
             'gpu_allocation':[0], 'env':{'METHOD_TEST_ENV':'present'}}
    yaml_write(tmp_path/'profile.yaml',profile)
    task=yaml.safe_load((REPO/'tasks/shape_color_video_ssl.yaml').read_text(encoding='utf-8'))
    task['training_config']='../train.yaml'
    (tmp_path/'tasks').mkdir()
    yaml_write(tmp_path/'tasks/shape_color_video_ssl.yaml',task)
    (tmp_path/'train.yaml').write_bytes((REPO/'configs/training/internvlu.yaml').read_bytes())
    yaml_write(tmp_path/'runner.yaml',{'mode':'native','agent':{'executable':sys.executable,'model':'test-model'},'docker':'DOES_NOT_EXIST.yaml'})
    (tmp_path/'protocol.md').write_text('Custom method protocol: optimize input and target frames.',encoding='utf-8')
    write(tmp_path/'campaign.json',{'deadline':(datetime.now(timezone.utc)+timedelta(hours=8)).isoformat(),
          'report_reserve_seconds':5400,'gpu_allocation':[0],'max_concurrent_gpus':4,'training_budget':7,'training_attempts':[]})
    prompt=tmp_path/'run.md'
    prompt.write_text('---\ntasks: [shape_color_video_ssl]\niterations: 3\nprofile: profile.yaml\nrunner: runner.yaml\nprotocol: protocol.md\ncampaign: campaign.json\ntasks_dir: tasks\n---\nTry intermediate targets.\n',encoding='utf-8')
    return prompt


def test_layers_resolve_by_owner_and_native_never_reads_docker(configured):
    plan=load_plan(configured)
    assert plan['task_id']=='shape_color_video_ssl' and plan['iterations']==3
    assert plan['strategy']=='Try intermediate targets.'
    assert plan['agent']['model']=='test-model'
    assert plan['agent']['timeout_seconds']==1800
    assert plan['backend_input']['task_contract']['rows_per_trial']==128
    assert Path(plan['backend_input']['sources']['v000']['video']).is_absolute()
    assert 'docker' not in plan['input_files']
    assert not any('DOES_NOT_EXIST' in path for path in plan['hashes'])
    assert plan['training_input']['train']['max_steps']==8
    assert plan['training_input']['model']['snapshot']==str(configured.parent/'weights')


@pytest.mark.parametrize('case',['profile_agent','profile_recipe','missing_model','gpu_mismatch','active','budget','deadline','same_eval','bad_frame','bad_recipe','unknown_prompt','unknown_runner','train_conflict','eval_count','timeout'])
def test_rejects_cross_layer_errors(configured,case):
    base=configured.parent
    profile=yaml.safe_load((base/'profile.yaml').read_text())
    task=yaml.safe_load((base/'tasks/shape_color_video_ssl.yaml').read_text())
    runner=yaml.safe_load((base/'runner.yaml').read_text())
    ledger=read(base/'campaign.json')
    if case=='profile_agent': profile['agent']={'model':'wrong-layer'}
    elif case=='profile_recipe': profile['rows_per_trial']=10
    elif case=='missing_model': profile['model_path_map']={}
    elif case=='gpu_mismatch': profile['gpu_allocation']=[1]
    elif case=='active': ledger['training_attempts']=[{'status':'running'}]
    elif case=='budget': ledger['training_budget']=2
    elif case=='deadline': ledger['deadline']='2000-01-01T00:00:00Z'
    elif case=='same_eval': profile['eval_path_map']['shape_color_test']=profile['eval_path_map']['shape_color_dev']
    elif case=='bad_frame': task['frame_policy']['max_input_frames']=0
    elif case=='bad_recipe': task['contract']['rows_per_trial']=127
    elif case=='unknown_prompt': configured.write_text(configured.read_text().replace('iterations: 3','iterations: 3\ntyop: true'))
    elif case=='unknown_runner': runner['agent']['timeout_seconds']=1
    elif case=='train_conflict':
        train=yaml.safe_load((base/'train.yaml').read_text())
        train['train']['max_steps']=100
        yaml_write(base/'train.yaml',train)
    elif case=='eval_count': task['contract']['evaluation']['test_samples']=49
    elif case=='timeout': task['contract']['training']['timeout_seconds']=0
    yaml_write(base/'profile.yaml',profile); yaml_write(base/'tasks/shape_color_video_ssl.yaml',task)
    yaml_write(base/'runner.yaml',runner); write(base/'campaign.json',ledger)
    with pytest.raises((ValueError,KeyError)):
        load_plan(configured)


def test_changing_protocol_or_task_is_detected_after_freeze(configured):
    plan=load_plan(configured)
    (configured.parent/'protocol.md').write_text('changed')
    with pytest.raises(ValueError,match='Frozen input changed'):
        verify(plan)


def test_evaluation_images_are_frozen_as_well_as_the_manifest(configured):
    plan=load_plan(configured)
    Image.new('RGB',(512,512),'black').save(configured.parent/'data/2.png')
    with pytest.raises(ValueError,match='Frozen input changed'):
        verify(plan)


def test_docker_is_optional_and_agent_can_exist_only_in_image(configured):
    base=configured.parent
    yaml_write(base/'runner.yaml',{'mode':'docker','agent':{'executable':'not-on-host'},'docker':'container.yaml'})
    profile=yaml.safe_load((base/'profile.yaml').read_text())
    profile['python']='/container-only/python'
    yaml_write(base/'profile.yaml',profile)
    with pytest.raises(ValueError,match='not found'):
        load_plan(configured)
    yaml_write(base/'container.yaml',{'image':'pinned:image','mount_profile_resources':True,'env':{'CUDA_HOME':'/container/cuda'}})
    plan=load_plan(configured)
    assert plan['runner']['image']=='pinned:image'
    assert plan['input_files']['docker']==str(base/'container.yaml')
    assert str(base/'data') in plan['runner']['read_only_mounts']
    assert plan['runner']['container_env']=={'CUDA_HOME':'/container/cuda'}


def test_real_cpu_initialization_uses_task_contract_and_selected_protocol(configured,tmp_path):
    # Exercise the real resolve subprocess with a smaller, consistent recipe; no optimizer steps.
    task_path=configured.parent/'tasks/shape_color_video_ssl.yaml'
    task=yaml.safe_load(task_path.read_text())
    task['contract']['rows_per_trial']=4
    task['contract']['training'].update(max_steps=2,global_batch=2,warmup_steps=1)
    yaml_write(task_path,task)
    run=SuiteController.create(configured,tmp_path/'output')
    resolved=read(run.root/'config/backend.json')
    assert resolved['rows_per_trial']==4
    assert resolved['training']['max_steps']==2
    assert run.state['targets'][0]['key']=='shape_color_video_ssl_iter1'
    assert (run.root/'workspace/METHOD.md').read_text()==(configured.parent/'protocol.md').read_text()
    assert (run.root/'config/inputs/task.yaml').exists()
    assert (run.root/'config/inputs/runner.yaml').exists()
    assert Path(read(run.root/'config/backend.json')['external']['dev_manifest']).name=='dev.resolved.jsonl'
    assert Path(json.loads((run.root/'config/dev.resolved.jsonl').read_text().splitlines()[0])['image'][0]).is_absolute()
    assert not (run.root/'config/inputs/docker.yaml').exists()
    feedback=run.root/'workspace/dev_feedback.jsonl'
    assert len(feedback.read_text(encoding='utf-8').splitlines())==50
    assert read(run.root/'workspace/task.json')['dev_manifest']==str(feedback)
    assert read(run.root/'workspace/task.json')['max_text_chars']==task['max_text_chars']
    verify(run.plan)
    built=subprocess.run(run.build_command(),capture_output=True,text=True,encoding='utf-8')
    assert built.returncode==0,built.stderr
    assert {'uv_pyproject_toml','uv_uv_lock','uv__python-version'} <= run.plan['input_files'].keys()
    audited=run.worker('audit',run.root/'cpu-audit',dataset=run.root/'workspace/dataset')
    assert audited['samples']==4


def test_separate_training_dev_test_timeouts_reach_process_launcher(configured,tmp_path,monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    plan=load_plan(configured)
    config=deepcopy(plan['backend_input']['task_contract'])
    config['campaign']=str(configured.parent/'campaign.json')
    config['training']['timeout_seconds']=11
    config['evaluation'].update(dev_timeout_seconds=22,test_timeout_seconds=33)
    backend=REPO/'vendor/video_ssl_backend'
    monkeypatch.syspath_prepend(str(backend))
    spec=importlib.util.spec_from_file_location('method_training_test',backend/'training.py')
    training=importlib.util.module_from_spec(spec); spec.loader.exec_module(training)
    calls=[]
    monkeypatch.setitem(sys.modules,'campaign',SimpleNamespace(run_process=lambda *args,**kwargs:calls.append(kwargs) or 0))
    trial=tmp_path/'trial'; trial.mkdir()
    for mode,output in [('train','train'),('eval','dev'),('eval','test')]:
        training.launch(trial,config,mode,trial/output)
    assert [c['timeout_seconds'] for c in calls]==[11,22,33]
    assert [c['reserve_seconds'] for c in calls]==[5400,5400,1800]
