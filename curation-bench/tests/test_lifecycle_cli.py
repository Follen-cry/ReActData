"""Real CPU configuration/resolver through the lifecycle CLI; no model loading."""
import json
from pathlib import Path
import pytest
import yaml
from test_configuration import configured
from video_ssl.cli import main
from video_ssl.core.layout import load_plan
from video_ssl.core.runtime import read
from video_ssl.tools.suite import SuiteController


def result(capsys,args):
    assert main(args)==0
    return json.loads(capsys.readouterr().out)


def test_init_reuse_inspect_and_extend_cli(configured,tmp_path,monkeypatch,capsys):
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'suite'
    args=['init','--prompt',str(configured),'--root',str(root)]
    first=result(capsys,args)
    assert first['total_targets']==3
    assert result(capsys,args)['suite_dir']==str(root)
    assert len(list(root.glob('config/backend.json')))==1
    before=(root/'suite_state.json').read_bytes()
    inspect=result(capsys,['resume','--inspect','--suite-dir',str(root)])
    assert inspect['status']=='resumable'
    assert (root/'suite_state.json').read_bytes()==before
    assert result(capsys,['extend','--iterations','4','--suite-dir',str(root)])['plan']['total_targets']==4
    assert read(root/'workspace/task.json')['iterations']==4
    assert result(capsys,['plan','--run-dir',str(root)])['total_targets']==4


def test_explicit_profile_can_supply_missing_frontmatter_profile(configured):
    configured.write_text(configured.read_text().replace('profile: profile.yaml\n',''))
    plan=load_plan(configured,profile_path=str(configured.parent/'profile.yaml'))
    assert Path(plan['input_files']['profile']).name=='profile.yaml'


def add_second_task(configured):
    task=yaml.safe_load((configured.parent/'tasks/shape_color_video_ssl.yaml').read_text())
    task['task_id']='second'
    (configured.parent/'tasks/second.yaml').write_text(yaml.safe_dump(task))
    configured.write_text(configured.read_text().replace('tasks: [shape_color_video_ssl]','tasks: [shape_color_video_ssl, second]'))


def test_real_multitask_plan_and_initialization(configured,tmp_path):
    add_second_task(configured)
    plan=load_plan(configured)
    assert plan['kind']=='multi_task'
    suite=SuiteController.create(configured,tmp_path/'suite')
    assert suite.get_run_plan()['total_targets']==6
    assert len({child.root/'workspace' for child in suite.children()})==2
    assert all((child.root/'config/backend.json').exists() for child in suite.children())


def test_multitask_budget_is_aggregate(configured):
    add_second_task(configured)
    configured.write_text(configured.read_text().replace('iterations: 3','iterations: 4'))
    with pytest.raises(ValueError,match='task queue'): load_plan(configured)


@pytest.mark.parametrize('variant',['instruction','skill'])
def test_guidance_appendix_composes_base_and_freezes_both(configured,tmp_path,variant):
    configured.write_text(configured.read_text().replace('protocol: protocol.md',f'protocol: {variant}'))
    plan=load_plan(configured)
    assert 'protocol_base' in plan['input_files']
    suite=SuiteController.create(configured,tmp_path/'suite')
    protocol=(suite.root/'workspace/METHOD.md').read_text(encoding='utf-8')
    assert 'Research-Agent Operating Protocol' in protocol or 'Video SSL' in protocol
    assert ('Instruction-Guided' if variant=='instruction' else 'Skill-Grounded') in protocol
