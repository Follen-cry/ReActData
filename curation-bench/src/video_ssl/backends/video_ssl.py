"""Video SSL adapter for the suite/session tool lifecycle."""
import json
import os
from pathlib import Path
from ..core.runtime import process
from ..core.uv_runtime import project_command, worker_command


class VideoSSLBackend:
    def __init__(self, suite):
        self.suite = suite

    def invoke(self, action, directory, *, dataset=None, checkpoint=None, config=None):
        directory = Path(directory)
        script=Path(self.suite.plan['pipeline_dir']) / 'worker.py'
        args=[action,'--config',str(config or self.suite.root/'config'/'backend.json'),'--root',str(directory)]
        if dataset is not None:
            args += ['--dataset', str(dataset)]
        if checkpoint is not None:
            args += ['--checkpoint', str(checkpoint)]
        repo=Path(__file__).resolve().parents[3]
        container=self.suite.plan['runner'].get('mode')=='docker' and action!='resolve' and os.environ.get('VIDEO_SSL_CONTAINER')!='1'
        argv=project_command(repo,'python',str(script),*args) if action=='resolve' else worker_command(self.suite.plan['python'],script,*args,uv=['uv'] if container else None)
        if container:
            from ..runner.agent_runner import docker_command
            argv=docker_command(self.suite,argv)
        log = directory / f'{action}_worker.log'
        try:
            process(argv, log, env={**os.environ,**self.suite.plan.get('env',{})})
        except RuntimeError as exc:
            lines=log.read_text(encoding='utf-8').strip().splitlines() if log.exists() else []
            raise RuntimeError(f'{action} worker failed: {lines[-1] if lines else exc}; see {log}') from exc
        return json.loads(log.read_text(encoding='utf-8').strip().splitlines()[-1])
