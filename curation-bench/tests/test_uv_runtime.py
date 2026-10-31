"""Real uv process boundaries: project selection and existing worker environments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

from video_ssl.core.uv_runtime import project_command, worker_command

REPO=Path(__file__).resolve().parents[1]


def run(command,cwd,env):
    result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stderr+result.stdout
    return result.stdout


def test_cli_from_unrelated_workspace_uses_prepared_project(tmp_path):
    # A nearby conflicting project must not change the Agent's CLI environment.
    (tmp_path/'pyproject.toml').write_text('[project]\nname="decoy"\nversion="1.0"\nrequires-python="<3.0"\n')
    env={**os.environ,'UV_CACHE_DIR':str(tmp_path/'cache'),'UV_OFFLINE':'1','PYTHONIOENCODING':'utf-8'}
    before=(REPO/'uv.lock').read_bytes()
    output=run(project_command(REPO,'video-ssl','--help'),tmp_path,env)
    assert 'Agent-driven video SSL data construction method' in output
    assert (REPO/'uv.lock').read_bytes()==before
    assert not (tmp_path/'.venv').exists()


def test_worker_preserves_existing_venv_and_packages_without_sync(tmp_path):
    gpu=tmp_path/'gpu-env'
    venv.EnvBuilder(with_pip=False).create(gpu)
    python=gpu/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    env={**os.environ,'UV_CACHE_DIR':str(tmp_path/'cache'),'UV_OFFLINE':'1','PYTHONIOENCODING':'utf-8',
         'VIRTUAL_ENV':str(REPO/'.venv'),'UV_PROJECT_ENVIRONMENT':str(tmp_path/'must-not-create')}
    purelib=Path(run([str(python),'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],tmp_path,env).strip())
    (purelib/'gpu_env_marker.py').write_text('VALUE="original GPU dependency"\n')
    probe=tmp_path/'probe.py'
    probe.write_text('import gpu_env_marker, json, sys\nprint(json.dumps([sys.prefix,gpu_env_marker.VALUE]))\n')
    before=set(purelib.iterdir())
    result=json.loads(run(worker_command(str(python),probe),REPO,env))
    assert Path(result[0]).resolve()==gpu.resolve()
    assert result[1]=='original GPU dependency'
    assert set(purelib.iterdir())-before <= {purelib/'__pycache__'}
    assert not (tmp_path/'must-not-create').exists()
