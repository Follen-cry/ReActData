import sys,json,importlib.util
from pathlib import Path
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment,agent
from dataset import validate
from storage import read,write
spec=importlib.util.spec_from_file_location("starter",Path(experiment.BASE)/"starter/build_dataset.py")
starter=importlib.util.module_from_spec(spec);spec.loader.exec_module(starter)

def setup(tmp_path,mode="simulation"):
    video=tmp_path/"tiny.gif"
    Image.new("RGB",(8,8),"red").save(video,save_all=True,append_images=[Image.new("RGB",(8,8),"blue")],duration=100)
    sources={}
    for i in range(128 if mode=="real" else 1):
        v=tmp_path/f"v{i}.gif";v.write_bytes(video.read_bytes())
        f0=tmp_path/f"v{i}_f0.png";last=tmp_path/f"v{i}_final.png"
        Image.new("RGB",(512,512),"red").save(f0)
        Image.new("RGB",(512,512),"blue").save(last)
        sources[f"v{i}"]=dict(video=str(v),f0=str(f0),final=str(last),frame_count=2,
                             description="Complete grid",eval_text="<image>\nComplete grid")
    cfg=dict(mode=mode,task="shape_color_then_scale",max_dataset_bytes=5000000,sources=sources)
    if mode=="simulation":cfg.update(simulation_samples=1,simulation_max_trials=2)
    else:
        import datetime
        ledger=tmp_path/'campaign.json'
        write(ledger,dict(deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).isoformat(),training_budget=7,max_concurrent_gpus=4,report_reserve_seconds=5400,training_attempts=[],consumed=0,remaining=7,gpu_allocation=[0]))
        cfg.update(campaign=str(ledger),estimated_trial_seconds=1)
    write(tmp_path/"config.json",cfg)
    root=tmp_path/"runs/e";experiment.init(root,tmp_path/"config.json")
    cfg=read(root/"config.json")
    data=root/"workspace/dataset";starter.build(cfg,data)
    return root,data,cfg

def test_multitrial_snapshot_and_locked_final(tmp_path):
    root,data,c=setup(tmp_path)
    assert len(validate(data,c))==1
    a=experiment.run(root,data,"one");assert a["status"]=="completed" and a["simulated"]
    saved=(root/"trials/t0001/dataset/images/0_f0.png").read_bytes()
    (root/"workspace/build_dataset.py").write_text("changed")
    assert experiment.run(root,data,"one")==a
    b=experiment.run(root,data,"two");assert b["trial_id"]=="t0002"
    with pytest.raises(ValueError):experiment.run(root,data,"three")
    (data/"images/0_f0.png").write_bytes(b"broken")
    assert (root/"trials/t0001/dataset/images/0_f0.png").read_bytes()==saved
    final=experiment.finalize(root,"t0001");assert final["simulated"] and final["test"]["checkpoint"] is None
    assert experiment.finalize(root,"t0001")==final
    with pytest.raises(ValueError):experiment.finalize(root,"t0002")

@pytest.mark.parametrize("change",["duplicate","escape","missing","frame","source","count","image","structure"])
def test_bad_submissions(tmp_path,change):
    root,data,c=setup(tmp_path);rows=[json.loads((data/"manifest.jsonl").read_text())]
    if change=="duplicate":rows*=2
    elif change=="escape":rows[0]["target_image"]="../outside.png"
    elif change=="missing":rows[0]["target_image"]="missing.png"
    elif change=="frame":rows[0]["target_frame"]=0
    elif change=="source":rows[0]["source_video_id"]="test"
    elif change=="count":c["rows_per_trial"]=0
    elif change=="image":(data/"images/0_final.png").write_bytes(b"bad")
    else:rows[0]["input_images"]*=2
    (data/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(Exception):validate(data,c)
    assert read(root/"state.json")["trials"]=={}

def test_real_failure_no_fallback_and_no_retry(tmp_path):
    root,data,c=setup(tmp_path,"real")
    r=experiment.run(root,data,"one");assert r["status"]=="failed" and not r["simulated"]
    assert "REAL_RESOURCE_MISSING" in r["error"]
    assert not list((root/"trials").rglob("*.pt"))
    assert experiment.run(root,data,"one")==r
    with pytest.raises(ValueError):experiment.finalize(root,r["trial_id"])
    c["mode"]="simulation";write(root/"config.json",c)
    with pytest.raises(ValueError):experiment.run(root,data,"two")

def test_tamper(tmp_path):
    root,data,c=setup(tmp_path);experiment.run(root,data,"one")
    (root/"trials/t0001/code/build_dataset.py").write_text("tamper")
    with pytest.raises(ValueError):experiment.finalize(root,"t0001")

def test_lock_and_interrupted_state_do_not_restart(tmp_path):
    root,data,c=setup(tmp_path)
    (root/"operation.lock").touch()
    with pytest.raises(FileExistsError):experiment.run(root,data,"one")
    (root/"operation.lock").unlink()
    s=read(root/"state.json");s["requests"]["one"]="t0001";s["trials"]["t0001"]={"status":"running"}
    write(root/"state.json",s)
    assert experiment.run(root,data,"one")=={"status":"running"}


def test_cli_and_independence(tmp_path):
    import subprocess,shutil
    root,data,c=setup(tmp_path)
    # Copy only new implementation to an unrelated directory; use CLI in a fresh interpreter.
    isolated=tmp_path/"isolated"
    shutil.copytree(experiment.BASE,isolated,ignore=shutil.ignore_patterns("runs","__pycache__",".pytest_cache"))
    script="import sys,runpy;sys.path[:0]="+repr([str(isolated)]+[x for x in sys.path if "pipeline" not in x])+";sys.argv="+repr([str(isolated/"cli.py"),"--root",str(root),"run","--dataset",str(data),"--request-id","cli"])+";runpy.run_path(sys.argv[0],run_name='__main__')"
    proc=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)["simulated"] is True
    for path in isolated.glob("*.py"):
        assert "pipeline-v6" not in path.read_text() and "pipeline/" not in path.read_text()


def test_agent_launch_resume_and_logs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root, _, _ = setup(tmp_path)
    calls = []
    def fake_run(args, **kw):
        calls.append(args)
        assert kw['cwd'] == root / 'workspace'
        assert 'COMMANDS.md' in kw['input']
        assert '--dangerously-bypass-approvals-and-sandbox' in args
        assert kw.get('shell', False) is False
        kw['stdout'].write('{"type":"thread.started","thread_id":"session-123"}\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(agent.subprocess, 'run', fake_run)
    result = agent.launch(root, executable='/opt/codex', model='configured-model')
    assert result['status'] == 'completed' and result['session_id'] == 'session-123'
    assert result['isolation'] == 'protocol_only'
    assert read(Path(result['artifacts']['logs']) / 'launch.json') == result
    assert str(root) in (root / 'workspace/COMMANDS.md').read_text(encoding='utf-8')
    again = agent.launch(root, resume=result['session_id'])
    assert calls[1][:4] == ['codex', 'exec', 'resume', 'session-123']
    assert calls[1][-1] == '-'
    assert again['artifacts']['logs'] != result['artifacts']['logs']
    assert read(root / 'state.json')['trials'] == {}


@pytest.mark.parametrize('missing', [False, True])
def test_agent_failure_is_recorded(tmp_path, monkeypatch, missing):
    from types import SimpleNamespace
    root, _, _ = setup(tmp_path)
    def fail(*args, **kw):
        if missing:
            raise FileNotFoundError('codex executable missing')
        kw['stderr'].write('authentication failed')
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(agent.subprocess, 'run', fail)
    result = agent.launch(root)
    assert result['status'] == 'failed' and result['error']
    assert read(Path(result['artifacts']['logs']) / 'launch.json') == result
    assert read(root / 'state.json')['trials'] == {}


def test_fixed_text_source_integrity_and_exact_count(tmp_path):
    root,data,c=setup(tmp_path)
    manifest=data/'manifest.jsonl'
    row=json.loads(manifest.read_text())
    row['text']='agent changed task description'
    manifest.write_text(json.dumps(row),encoding='utf-8')
    with pytest.raises(ValueError,match='fixed evaluation text'):validate(data,c)
    row['text']=c['sources'][row['source_video_id']]['eval_text']
    manifest.write_text(json.dumps(row)+'\n'+json.dumps({**row,'sample_id':'different'}),encoding='utf-8')
    with pytest.raises(ValueError,match='sample budget'):validate(data,c)
    c['rows_per_trial']=2
    with pytest.raises(ValueError,match='only be used once'):validate(data,c)
    c['rows_per_trial']=1;manifest.write_text(json.dumps(row),encoding='utf-8')
    Image.new('RGB',(512,512),'green').save(c['sources'][row['source_video_id']]['f0'])
    with pytest.raises(ValueError,match='canonical source changed'):validate(data,c)


def test_task_defaults_aliases_and_real_override(tmp_path):
    from task import resolve
    root,data,c=setup(tmp_path)
    assert [c[k] for k in ('model_init_seed','train_seed','build_seed','eval_seed')]==[42,1000,1000,0]
    assert c['training']['max_steps']==8 and c['training']['global_batch']==16
    assert c['training']['warmup_steps']==3 and not c['training']['shuffle']
    raw=read(tmp_path/'config.json')
    raw['sources']['alias']=dict(raw['sources']['v0'])
    write(tmp_path/'alias.json',raw)
    with pytest.raises(ValueError,match='aliases'):resolve(tmp_path/'alias.json')
    raw['sources'].pop('alias');raw['mode']='real'
    write(tmp_path/'bad-real.json',raw)
    with pytest.raises(ValueError,match='simulation overrides'):resolve(tmp_path/'bad-real.json')


def test_baseline_reproducibility_and_manifest_order(tmp_path):
    import training
    root,data,c=setup(tmp_path)
    # Independent source IDs/paths; synthetic copies only for the CPU fixture.
    raw=read(tmp_path/'config.json')
    src=raw['sources']['v0']
    for i in range(1,4):
        v=tmp_path/f'extra{i}.gif';v.write_bytes(Path(src['video']).read_bytes())
        raw['sources'][f'v{i}']={**src,'video':str(v),'eval_text':f'<image>\nTask {i}'}
    raw['simulation_samples']=3
    write(tmp_path/'multi.json',raw)
    second=tmp_path/'runs/multi';experiment.init(second,tmp_path/'multi.json')
    config=read(second/'config.json')
    one=second/'workspace/dataset';two=tmp_path/'repeat'
    starter.build(config,one);starter.build(config,two)
    assert (one/'manifest.jsonl').read_bytes()==(two/'manifest.jsonl').read_bytes()
    rows=validate(one,config)
    assert len({r['source_video_id'] for r in rows})==3
    specs=training.row_specs(one)
    assert [r['conversations'][0]['value'] for r in specs]==[r['text'] for r in rows]
    assert all(r['cond_image']==r['images'][0] for r in specs)
    result=experiment.run(second,one,'baseline')
    receipt=read(Path(result['artifacts']['trial'])/'training_contract.json')
    assert receipt['simulated'] and receipt['seeds']['train_seed']==1000
