"""Actual subprocess smoke test for bundled CPU data operations only."""
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
from video_ssl.core.uv_runtime import worker_command


def test_bundled_worker_resolves_builds_and_audits_without_sibling_backend(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    backend=repo/'vendor'/'video_ssl_backend'
    images=[Image.new('RGB',(512,512),color) for color in ['red','green','blue']]
    video=tmp_path/'video.gif'
    images[0].save(video,save_all=True,append_images=images[1:],duration=100)
    paths=[]
    for i,image in enumerate(images):
        path=tmp_path/f'{i}.png'; image.save(path); paths.append(str(path))
    config={'mode':'simulation','task':'shape_color_then_scale','simulation_samples':1,
            'frame_policy':{'mode':'select','target':'select','max_input_frames':2},
            'sources':{'v':{'video':str(video),'f0':paths[0],'final':paths[2],'frame_count':3,
                           'frames':{'1':paths[1]},'description':'Change a shape','eval_text':'<image>\nGenerate the final result.'}}}
    source=tmp_path/'source.json'; source.write_text(json.dumps(config),encoding='utf-8')
    resolved=tmp_path/'resolved'
    def command(*args):
        return subprocess.run(worker_command(sys.executable,args[0],*args[1:]),cwd=repo,capture_output=True,text=True,encoding='utf-8')
    result=command(backend/'worker.py','resolve','--config',source,'--root',resolved)
    assert result.returncode==0, result.stderr+result.stdout
    data=tmp_path/'dataset'
    built=command(repo/'curation'/'curate.py','--config',resolved/'backend.json','--tools',backend,'--output',data)
    assert built.returncode==0, built.stderr+built.stdout
    fixed=tmp_path/'fixed-baseline'
    baseline=command(repo/'baselines'/'generate_random.py','--config',resolved/'backend.json',
                     '--tools',backend,'--output',fixed)
    assert baseline.returncode==0,baseline.stderr+baseline.stdout
    assert (fixed/'manifest.jsonl').read_bytes()==(data/'manifest.jsonl').read_bytes()
    for image in (data/'images').iterdir():
        assert (fixed/'images'/image.name).read_bytes()==image.read_bytes()
    audited=command(backend/'worker.py','audit','--config',resolved/'backend.json','--root',tmp_path/'audit','--dataset',data)
    assert audited.returncode==0, audited.stderr+audited.stdout
    assert json.loads(audited.stdout)['status']=='valid'
    # Pixel provenance must survive the actual process boundary.
    Image.new('RGB',(512,512),'black').save(data/'images'/'0_final.png')
    failed=command(backend/'worker.py','audit','--config',resolved/'backend.json','--root',tmp_path/'bad-audit','--dataset',data)
    assert failed.returncode==1
    assert json.loads(failed.stdout)['status']=='failed'
