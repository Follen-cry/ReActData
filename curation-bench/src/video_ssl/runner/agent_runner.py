"""One continuous research-agent session, native or containerized."""
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import hashlib
import subprocess

from ..core.runtime import agent_command, process, read, write, verify, require_linux, digest
from ..tools.suite import SuiteController
from ..core.runtime import exclusive
from ..core.uv_runtime import project_command


def docker_command(run, command):
    runner = run.plan['runner']
    repo = Path(__file__).resolve().parents[3]
    image = runner.get('image')
    if not image:
        raise ValueError('runner.image required for docker mode')
    devices = read(run.plan['campaign'])['gpu_allocation']
    name = 'video-ssl-'+hashlib.sha256(str(run.root).encode()).hexdigest()[:16]
    argv = ['docker','run','--name',name,'--rm','-i','--init','--gpus','"device='+','.join(map(str,devices))+'"',
            '--ipc=host','--network=host','--workdir',str(run.root/'workspace'),
            '--env','VIDEO_SSL_RUN='+str(run.root),'--env','VIDEO_SSL_AGENT=1','--env','VIDEO_SSL_CONTAINER=1',
            '--env','HOME=/tmp/video-ssl-home',
            '--env','VIDEO_SSL_CURRENT_TASK='+run.plan.get('task_id','shape_color_then_scale'), '--env',f'PYTHONPATH={repo / "src"}']
    if hasattr(os, 'getuid'):
        argv += ['--user',f'{os.getuid()}:{os.getgid()}']
    mounts={str(repo):(str(repo),True),str(run.root):(str(run.root),False)}
    for path in runner.get('read_only_mounts', []):
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise ValueError(f'Missing container mount: {p}')
        mounts.setdefault(str(p),(str(p),True))
    ledger_parent = Path(run.plan['campaign']).resolve().parent
    if ledger_parent == Path(ledger_parent.anchor) or ledger_parent==repo:
        raise ValueError('Keep campaign in a dedicated directory, not a filesystem root')
    mounts[str(ledger_parent)]=(str(ledger_parent),False)
    for source, destination in runner.get('credential_files', {}).items():
        p = Path(source).expanduser().resolve()
        if not p.is_file():
            raise ValueError('Credential file missing')
        if destination in mounts:
            raise ValueError('Credential destination overlaps a resource mount')
        mounts[destination]=(str(p),True)
    for destination,(source,readonly) in mounts.items():
        argv += ['--mount',f'type=bind,src={source},dst={destination}'+(',readonly' if readonly else '')]
    for name in sorted(set(runner.get('pass_env', []))|set(run.plan.get('env',{}))):
        argv += ['--env', str(name)]  # docker reads host value; do not log secrets
    for name,value in runner.get('container_env',{}).items():
        argv += ['--env',name+'='+value]
    # Environment-specific GPU Python paths must exist inside the configured image/mounts.
    return argv + ['--entrypoint',command[0], image, *command[1:]]


def launch(run):
    if run.plan.get('kind')=='multi_task':
        for index,child in enumerate(run.children()):
            run.state.update(active_task=index,status='running')
            run.save()
            child.prepare_resume()
            try:
                launch(child)
            finally:
                run.report()
        # Do not expose any final-test result while another task is researching.
        for child in run.children():
            if child.plan['finalize']=='auto':
                from ..tools.evaluation import finalize
                finalize(child)
        return run.report()
    if run.state['status']=='search_completed':
        return run.report()
    require_linux()
    verify(run.plan)
    if run.state['status'] not in ('initialized','running'):
        raise ValueError('Cannot launch an already completed or failed method run')
    workspace = run.root/'workspace'
    runner = run.plan['runner']
    mode = runner.get('mode','native')
    if mode not in ('native','docker'):
        raise ValueError('runner.mode must be native or docker')
    repo=Path(__file__).resolve().parents[3]
    # The host uv binary path need not exist inside the container.
    uv=['uv'] if mode=='docker' else None
    prefix=shlex.join(project_command(repo,'video-ssl',uv=uv,python=runner.get('cli_python')))
    build=shlex.join(run.build_command())
    commands=[f'# Commands for {run.plan.get("task_id","shape_color_then_scale")}', '',
              f'Run root: `{run.root}`', f'Workspace: `{workspace}`', '',
              'Start with this workspace AGENTS.md and METHOD.md. Follow the reading scope there; do not use repository history or other protocol variants to infer this experiment.', '',
              'Use these exact uv commands and paths. The CLI uses the prepared locked project; worker commands preserve the profile GPU environment without syncing it.', '',
              f'CLI prefix: `{prefix}`', '', '## Construct the current candidate', '',
              'Run the following after next activates an iteration. The output directory must be absent when rebuilding.',
              '', '```bash', build, '```', '', '## Experiment tools', '',
              'VIDEO_SSL_RUN selects this run automatically. Explicit --run-dir is included below for clarity.', '',
              '| Tool | Exact command | Expected success |', '|---|---|---|']
    tools=[('plan','Inspect the declared queue and saved stages'),('preinit','completed or disabled'),
           ('next','target_activated or task_done'),('task','Current context; inspect state and task identity'),
           ('time-budget','Remaining strategy interval; expired is a boolean'),
           ('audit','valid and samples'),('submit','submitted and submission_path'),
           ('finetune','completed and checkpoint'),('eval','completed and dev'),('report','Updated human report')]
    for name,result in tools:
        args=[name,'--run-dir',str(run.root)]
        if name in ('audit','submit'): args+=['--path',str(workspace/'dataset')]
        commands.append(f'| {name} | `{prefix} {shlex.join(args)}` | {result} |')
    commands += ['', 'To abandon a constructible but unfinishable strategy, use fail --reason with a specific explanation. '
                 'It intentionally returns failed/exit 1; inspect the saved state before next. Never discard an in-flight operation.', '',
                 'Run finetune/eval as blocking foreground commands. A successful finetune is trained, not yet dev-completed. '
                 'Do not use background jobs, direct torchrun or extra submissions.', '',
                 'Do not run finalize. The host owns final test after task_done. See METHOD.md for failure handling, '
                 'data isolation, feedback paths and recording requirements.']
    command_path=workspace/'COMMANDS.md'
    command_text='\n'.join(commands)+'\n'
    if command_path.exists() and command_path.read_text(encoding='utf-8')!=command_text:
        raise ValueError('Frozen COMMANDS changed; host inspection required')
    command_path.write_text(command_text,encoding='utf-8')
    command = agent_command(run.plan['agent'])
    if mode == 'docker':
        command = docker_command(run, command)
    prompt = (f"Read this workspace AGENTS.md, METHOD.md, COMMANDS.md, task.json and NOTES.md. Follow their research reading scope; do not inventory the repository or read history/setup documents. You own {run.plan.get('task_id','shape_color_then_scale')} in this method run. "
              'Complete preinit if enabled, then use next/task/build/audit/submit/finetune/eval repeatedly until next returns task_done. '
              'First iteration is the unchanged random F0-to-final baseline. Later iterations optimize video choice, input/context frames, '
              'target frames and order. Read previous iteration code, predictions and scores before each hypothesis. '
              'Resume saved submitted/trained stages instead of rebuilding them; next restores the best accepted code before a new iteration. '
              'Do not finalize or access test. Maintain NOTES.md. Do not stop for confirmation during the declared iterations.\n'
              +run.plan.get('task_definition',{}).get('goal','')+'\n'+run.plan['strategy'])
    ledger = read(run.plan['campaign'])
    seconds = (datetime.fromisoformat(ledger['deadline'].replace('Z','+00:00'))-datetime.now(timezone.utc)).total_seconds()-ledger['report_reserve_seconds']
    seconds = min(seconds, runner.get('timeout_seconds',seconds))
    if seconds <= 0:
        raise ValueError('Campaign agent window exhausted')
    with exclusive(run.root):
        run.state=SuiteController(run.root).state
        if run.state.get('agent_status') in ('running','interrupted','failed'):
            raise ValueError('Agent has active/interrupted state; inspect before explicit manual continuation')
        session_number=len(run.state.get('agent_sessions', []))+1
        output=run.root/'output'/f'session_{session_number:04d}'
        while output.exists():
            session_number+=1
            output=run.root/'output'/f'session_{session_number:04d}'
        output.mkdir(parents=True,exist_ok=False)
        (output/'initial_prompt.md').write_text(prompt,encoding='utf-8')
        run.state['agent_status']='running'
        run.state.setdefault('agent_sessions', []).append(dict(number=session_number,output=str(output)))
        run.plan['hashes'][str(command_path)]=digest(command_path)
        write(run.root/'config'/'plan.json',run.plan)
        run.save()
    environment = {**run.plan.get('env',{}),'VIDEO_SSL_RUN':str(run.root),'VIDEO_SSL_AGENT':'1','VIDEO_SSL_CURRENT_TASK':run.plan.get('task_id','shape_color_then_scale'),
                   'PYTHONPATH':str(Path(__file__).resolve().parents[2])+os.pathsep+os.environ.get('PYTHONPATH','')}
    previous = {k:os.environ.get(k) for k in environment}
    os.environ.update(environment)
    try:
        def on_start(pid):
            import socket
            owned=SuiteController(run.root)
            owned.state.update(agent_pid=pid,agent_host=socket.gethostname())
            owned.save()
        process(command,output/'run.jsonl',cwd=workspace,prompt=prompt,timeout=seconds,on_start=on_start)
        fresh = SuiteController(run.root)
        verify(fresh.plan)
        if fresh.state['status'] != 'search_completed':
            raise RuntimeError('Agent exited before completing the declared iteration loop')
        fresh.state['agent_status']='completed'
        fresh.save()
        return fresh.report()
    except BaseException as exc:
        fresh = SuiteController(run.root)
        fresh.state.update(agent_status='failed',error=str(exc))
        fresh.save()
        fresh.report()
        raise
    finally:
        if mode == 'docker':
            name = 'video-ssl-'+hashlib.sha256(str(run.root).encode()).hexdigest()[:16]
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name,None)
            else:
                os.environ[name]=value
