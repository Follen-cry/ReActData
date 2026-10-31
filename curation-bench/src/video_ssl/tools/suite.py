"""Suite lifecycle: configuration, target queue and persistent state."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import yaml

from ..core.runtime import load_plan, read, write, digest, verify, process, exclusive, tree_hash
from ..core.target import build_target_queue
from ..core.uv_runtime import project_command, worker_command


class SuiteController:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.plan = read(self.root / 'config' / 'plan.json')
        self.state = read(self.root / 'suite_state.json')


    @classmethod
    def create(cls, prompt, root, *, plan=None):
        plan = load_plan(prompt) if plan is None else plan
        if plan.get("kind") == "multi_task":
            from .multisuite import MultiTaskSuite
            return MultiTaskSuite.create_from_plan(prompt, root, plan)
        if not (Path(plan['pipeline_dir']) / 'worker.py').exists():
            raise ValueError('Bundled vendor/video_ssl_backend/worker.py is missing')
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        if 'backend_input' in plan:
            config_dir=root/'config'
            config_dir.mkdir()
            training=deepcopy(plan['training_input'])
            training['train']['output_dir']=str(root/'outputs')
            training_path=config_dir/'training.resolved.yaml'
            training_path.write_text(yaml.safe_dump(training,allow_unicode=True,sort_keys=False),encoding='utf-8')
            backend_input=deepcopy(plan['backend_input'])
            backend_input['external']['training_config']=str(training_path)
            for split,rows in plan.get('evaluation_inputs',{}).items():
                manifest=config_dir/(split+'.resolved.jsonl')
                manifest.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
                backend_input['external'][split+'_manifest']=str(manifest)
                plan['hashes'][str(manifest)]=digest(manifest)
            input_path=config_dir/'backend.input.json'
            write(input_path,backend_input)
            plan['experiment_config']=str(input_path)
            plan['hashes'].update({str(input_path):digest(input_path),str(training_path):digest(training_path)})
            for role,path in plan['input_files'].items():
                original=Path(path)
                destination=config_dir/'inputs'/(role+original.suffix)
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(original,destination)
            shutil.copy2(plan['campaign'],config_dir/'campaign.initial.json')
        write(root / 'config' / 'plan.json', plan)
        targets = []
        for target in build_target_queue([plan.get('task_id','shape_color_then_scale')], plan['iterations']):
            target.run_dir = root / target.key
            target.task_path = target.run_dir / 'task.resolved.json'
            targets.append(target.to_dict())
        write(root / 'suite_state.json', dict(status='initialized', targets=targets, active=None,
              preinit='pending' if plan['preinit'] else 'disabled', final='pending'))
        run = cls(root)
        try:
            from ..core.protocol import protocol_text, load_protocol, workspace_instructions
            from importlib.resources import files
            method_protocol = Path(str(files('video_ssl').joinpath('protocols/method.md')))
            if not plan.get('protocol_file'):
                run.plan['hashes'][str(method_protocol)] = digest(method_protocol)
            run.worker('resolve', root / 'config', config=plan['experiment_config'])
            config = read(root / 'config' / 'backend.json')
            # Snapshot all resolved settings. The mutable campaign is intentionally excluded.
            run.plan['hashes'][str(root / 'config' / 'backend.json')] = digest(root / 'config' / 'backend.json')
            write(root / 'config' / 'plan.json', run.plan)
            workspace = root / 'workspace'
            (workspace / 'curation').mkdir(parents=True)
            starter = Path(__file__).resolve().parents[3] / 'curation' / 'curate.py'
            if not starter.is_file():
                raise ValueError('Missing curation/curate.py starter')
            shutil.copy2(starter, workspace / 'curation' / 'curate.py')
            run.plan['hashes'][str(starter)] = digest(starter)
            public = {k:config[k] for k in ['task','sources','rows_per_trial','build_seed','training','evaluation','frame_policy']}
            public.update(task_id=plan.get('task_id',config['task']),goal=plan.get('task_definition',{}).get('goal',''))
            public['seeds']={k:config[k] for k in ('model_init_seed','train_seed','build_seed','eval_seed') if k in config}
            public['iterations']=plan['iterations']
            public.update({k:config[k] for k in ('max_dataset_bytes','max_text_chars') if k in config})
            if 'dev' in plan.get('evaluation_inputs',{}):
                feedback=workspace/'dev_feedback.jsonl'
                feedback.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in plan['evaluation_inputs']['dev']),encoding='utf-8')
                public['dev_manifest']=str(feedback)
                run.plan['hashes'][str(feedback)]=digest(feedback)
            write(workspace / 'task.json', public)
            protocol=load_protocol(plan['protocol_file']) if plan.get('protocol_file') else protocol_text()
            (workspace / 'METHOD.md').write_text(protocol, encoding='utf-8')
            (workspace / 'AGENTS.md').write_text(workspace_instructions(plan), encoding='utf-8')
            (workspace / 'NOTES.md').write_text('# Video SSL research notes\n', encoding='utf-8')
            # Baseline bytes are frozen from the repository starter, never inherited from an older run.
            run.plan['baseline_builder_sha256'] = digest(workspace / 'curation' / 'curate.py')
            for p in [workspace / 'METHOD.md', workspace / 'AGENTS.md', workspace / 'task.json']:
                run.plan['hashes'][str(p)] = digest(p)
            write(root / 'config' / 'plan.json', run.plan)
            (workspace / '.video_ssl_suite').write_text(str(root), encoding='utf-8')
            run.save()
            return run
        except BaseException as exc:
            run.state.update(status='initialization_failed', error=str(exc))
            run.save()
            raise


    def save(self):
        write(self.root / 'suite_state.json', self.state)


    def worker(self, action, directory, **kwargs):
        from ..backends.video_ssl import VideoSSLBackend
        return VideoSSLBackend(self).invoke(action, directory, **kwargs)


    def active(self):
        if self.state['active'] is None:
            raise ValueError('No active target; call next first')
        return self.state['targets'][self.state['active']]


    def target_dir(self, target=None):
        return self.root / (target or self.active())['key']


    def next(self):
        with exclusive(self.root):
            self.state=read(self.root/'suite_state.json')
            verify(self.plan)
            if self.state['preinit'] not in ('completed','disabled'):
                raise ValueError('Complete preinit first (or disable it before initialization)')
            if self.state['active'] is not None:
                current = self.active()
                if current['state'] in ('training','evaluating','submitting'):
                    raise ValueError('Interrupted/active operation requires inspection; never restart it automatically')
                if current['state'] not in ('completed','failed'):
                    # Repeated next is idempotent; skipping requires explicit fail.
                    return self.task()
                if current['iteration'] == 1 and current['state'] == 'failed':
                    raise ValueError('Baseline failed; host inspection required')
                self.settle_strategy()
            for i, target in enumerate(self.state['targets']):
                if target['state'] == 'pending':
                    target.update(state='active', strategy_started_at=time.time(), base_target=self.state.get('best_target'))
                    self.state.update(active=i, status='running')
                    self.target_dir(target).mkdir(parents=True, exist_ok=True)
                    write(self.target_dir(target) / 'task.resolved.json', read(self.root / 'workspace' / 'task.json'))
                    hypothesis = self.root / 'workspace' / 'curation' / 'hypothesis.md'
                    if hypothesis.exists():
                        hypothesis.unlink()
                    self.get_active_session().save_state()
                    self.report()
                    return self.task()
            self.state['status'] = 'search_completed'
            self.save()
            self.report()
            return {'status':'task_done', 'run_dir':str(self.root)}


    def build_command(self):
        workspace=self.root/'workspace'
        return worker_command(self.plan['python'],workspace/'curation'/'curate.py',
            '--config',str(workspace/'task.json'),'--tools',self.plan['pipeline_dir'],
            '--output',str(workspace/'dataset'),uv=['uv'] if self.plan['runner'].get('mode')=='docker' else None)


    def task(self):
        target = self.active()
        public=read(self.root/'workspace'/'task.json')
        return dict(target, status='target_activated', run_dir=str(self.target_dir()),
                    spec={k:public[k] for k in ('task','task_id','goal','rows_per_trial','frame_policy','training','evaluation','seeds','iterations','max_dataset_bytes','max_text_chars','dev_manifest') if k in public},
                    task_file=str(self.root / 'workspace' / 'task.json'),
                    default_submission_path=str(self.root / 'workspace' / 'dataset'),
                    builder=str(self.root / 'workspace' / 'curation' / 'curate.py'),
                    tools=self.plan['pipeline_dir'], python=self.plan['python'], build_command=self.build_command(),
                    prior_feedback=[{'key':t['key'], 'dev':t.get('dev'), 'state':t['state'],
                                     'artifacts':str(self.target_dir(t))} for t in self.state['targets'] if t['state'] in ('completed','failed')])


    def time_budget(self):
        target = self.active()
        elapsed = time.time() - target['strategy_started_at']
        limit = self.plan['agent'].get('timeout_seconds', 1800)
        return dict(strategy_timeout_seconds=limit, elapsed_seconds=elapsed,
                    remaining_seconds=max(0, limit-elapsed), expired=elapsed >= limit)


    def fail(self, reason):
        with exclusive(self.root):
            t = self.active()
            if t['state'] in ('training','evaluating','submitting','completed'):
                raise ValueError('Cannot discard an active operation or completed trial')
            session=self.get_active_session()
            t.update(state='failed', error=reason)
            session._log_event('fail',dict(reason=reason),dict(status='failed',error=reason))
            session.save_state()
            self.report()
            return {'status':'failed', 'error':reason}


    def preinit(self):
        from .evaluation import preinit
        return preinit(self)

    def report(self):
        from .reporting import write_report
        return write_report(self)

    # Public lifecycle API retained from upstream SuiteController.
    @classmethod
    def from_prompt(cls, prompt_path='run_prompt.md', profile_path=None, *,
                    output_root=None, new_suite=False):
        pointer = Path('.video_ssl_suite')
        root = Path(output_root).resolve() if output_root else None
        if root is None and not new_suite and pointer.is_file():
            root = Path(pointer.read_text(encoding='utf-8').strip()).resolve()
        if root is not None and root.exists() and not new_suite:
            suite = cls.from_state_file(root/'suite_state.json')
            # Config/source hashes are checked before any state is reused.
            suite.verify_inputs()
            suite.check_declaration(prompt_path, profile_path)
            pointer.write_text(str(suite.root), encoding='utf-8')
            return suite
        if root is None:
            from datetime import datetime, timezone
            root = Path('runs')/datetime.now(timezone.utc).strftime('suite_%Y%m%dT%H%M%S_%fZ')
        plan = load_plan(prompt_path) if profile_path is None else load_plan(prompt_path, profile_path=profile_path)
        suite = cls.create(prompt_path, root, plan=plan)
        pointer.write_text(str(suite.root), encoding='utf-8')
        return suite

    @classmethod
    def from_state_file(cls, state_path):
        root = Path(state_path).resolve().parent
        state = read(state_path)
        if state.get('kind') == 'multi_task':
            from .multisuite import MultiTaskSuite
            return MultiTaskSuite(root)
        return cls(root)

    def verify_inputs(self):
        verify(self.plan)

    def check_declaration(self, prompt_path, profile_path=None):
        inputs = self.plan.get('input_files', {})
        for path, role in ((prompt_path, 'prompt'), (profile_path, 'profile')):
            if path is not None and role=='profile':
                from ..core.layout import reference, REPO
                path=reference(Path(prompt_path).resolve().parent,str(path),REPO/'profiles','.yaml')
            if path is not None and role in inputs and (Path(path).resolve()!=Path(inputs[role]).resolve() or digest(path) != digest(inputs[role])):
                raise ValueError('Declaration differs from saved suite; use the frozen declaration or a new suite')

    def get_run_plan(self):
        return dict(self.state, suite_dir=str(self.root),
                    total_targets=len(self.state['targets']),
                    completed=sum(t['state']=='completed' for t in self.state['targets']),
                    failed=sum(t['state']=='failed' for t in self.state['targets']),
                    pending=sum(t['state']=='pending' for t in self.state['targets']),
                    active_target=self.active() if self.state['active'] is not None else None)

    def mark_target_completed(self, eval_result):
        target=self.active()
        target.update(state='completed',dev=eval_result['dev'],eval_result=eval_result)
        self.save()

    def begin_next_target(self):
        assigned = os.environ.get('VIDEO_SSL_CURRENT_TASK')
        if assigned and assigned != self.plan.get('task_id', 'shape_color_then_scale'):
            raise ValueError('Assigned task differs from suite task')
        return self.next()

    def get_active_session(self, *, validate=True):
        from .session import Session
        return Session(self, validate=validate)

    def prepare_resume(self, *, mutate=True):
        if not mutate:
            return self._prepare_resume(mutate=False)
        if (self.root/'operation.lock').exists():
            raise ValueError('Operation lock exists; host inspection required before resume')
        with exclusive(self.root):
            self.state=read(self.root/'suite_state.json')
            return self._prepare_resume(mutate=True, owned_lock=True)

    def _prepare_resume(self, *, mutate, owned_lock=False):
        self.verify_inputs()
        if self.state['status'] not in ('initialized','running','search_completed'):
            raise ValueError('Suite initialization requires host inspection')
        if not owned_lock and (self.root/'operation.lock').exists():
            raise ValueError('Operation lock exists; host inspection required before resume')
        if self.state['preinit'] not in ('pending','completed','disabled'):
            raise ValueError('Preinit requires host inspection')
        if self.state['final'] not in ('pending','completed'):
            raise ValueError('Final test requires host inspection; never replay it')
        ledger = read(self.plan['campaign'])
        if any(a['status'] in ('preparing','running','interrupted') for a in ledger.get('training_attempts', [])):
            raise ValueError('Campaign operation requires host inspection')
        for target in self.state['targets']:
            if target['state'] in ('training','evaluating','submitting'):
                raise ValueError('Uncertain operation requires host inspection; never replay it')
            if target['iteration']==1 and target['state']=='failed':
                raise ValueError('Baseline failed; host inspection required')
        if self.state.get('agent_status') == 'running':
            import socket
            pid = self.state.get('agent_pid')
            if type(pid) is not int or pid<=0 or self.state.get('agent_host') != socket.gethostname():
                raise ValueError('Agent ownership requires host inspection')
            from ..core.runtime import process_is_alive
            if process_is_alive(pid):
                raise ValueError('Agent is still running; do not launch a second session')
        if self.state['active'] is not None:
            session = self.get_active_session()
            if session.target['state'] in ('submitted','trained','completed'):
                session.verify_submission()
        if mutate and self.state['status']!='search_completed':
            self.state['agent_status'] = 'ready'
            self.save()
        return self.get_run_plan()

    def settle_strategy(self):
        from .session import Session
        candidates = [t for t in self.state['targets'] if t['state']=='completed']
        if not candidates:
            return
        best = max(candidates, key=lambda t:t['dev']['mean'])
        active = self.state['active']
        try:
            self.state['active'] = self.state['targets'].index(best)
            session = Session(self)
            session.verify_submission()
            source = session.directory/'code'
            workspace = self.root/'workspace'/'curation'
            if tree_hash(workspace) != tree_hash(source):
                # Preserve the candidate worktree; accepted archives are never reset.
                import uuid
                token = uuid.uuid4().hex
                staged = self.root/'workspace'/('.restore_'+token)
                shutil.copytree(source, staged)
                backup = self.root/'workspace_history'/token
                backup.parent.mkdir(exist_ok=True)
                workspace.rename(backup)
                staged.rename(workspace)
            self.state['best_target'] = best['key']
            if active is not None:
                current = self.state['targets'][active]
                current['strategy_result'] = 'keep' if current is best else 'discard' if current['state']=='completed' else 'crash'
        finally:
            self.state['active'] = active

    def extend(self, iterations):
        from datetime import datetime, timezone
        from contextlib import ExitStack
        lock_root=Path(self.plan.get('suite_parent', self.root))
        with ExitStack() as locks:
            locks.enter_context(exclusive(lock_root))
            if lock_root!=self.root:
                locks.enter_context(exclusive(self.root))
            self.state=read(self.root/'suite_state.json')
            self.plan=read(self.root/'config'/'plan.json')
            self.verify_inputs()
            if self.state['final'] != 'pending':
                raise ValueError('Cannot extend after final test has started')
            if self.state.get('agent_status') == 'running':
                raise ValueError('Stop/inspect the Agent before extending')
            if any(t['state'] in ('training','evaluating','submitting') for t in self.state['targets']):
                raise ValueError('Uncertain operation requires host inspection')
            if any(t['iteration']==1 and t['state']=='failed' for t in self.state['targets']):
                raise ValueError('Baseline failed; host inspection required')
            old = self.plan['iterations']
            maximum = read(self.root/'config'/'backend.json').get('max_trials', 7)
            if type(iterations) is not int or not old < iterations <= maximum:
                raise ValueError('New total iterations must increase and fit task max_trials')
            additional = iterations-old
            ledger = read(self.plan['campaign'])
            targets = self.state['targets']
            if self.plan.get('suite_parent'):
                parent = SuiteController.from_state_file(Path(self.plan['suite_parent'])/'suite_state.json')
                if any(child.state['final']!='pending' for child in parent.children()):
                    raise ValueError('Cannot extend after any task final test has started')
                targets = parent.get_run_plan()['targets']
                if any(t['state'] in ('training','evaluating','submitting') for t in targets):
                    raise ValueError('Another task operation requires host inspection')
            reserved = sum(t['state'] in ('pending','active','submitted') for t in targets)
            if any(a['status'] in ('preparing','running','interrupted') for a in ledger.get('training_attempts', [])):
                raise ValueError('Campaign operation requires host inspection')
            if len(ledger.get('training_attempts', []))+reserved+additional > ledger['training_budget']:
                raise ValueError('Insufficient shared training attempts for extension')
            remaining = (datetime.fromisoformat(ledger['deadline'].replace('Z','+00:00'))-datetime.now(timezone.utc)).total_seconds()
            if remaining < self.plan.get('task_definition',{}).get('estimated_trial_seconds',1)*additional+ledger['report_reserve_seconds']:
                raise ValueError('Insufficient campaign time for extension')
            receipt = self.root/'config'/'extensions'/f'{iterations}.json'
            write(receipt, dict(previous_iterations=old, iterations=iterations, time=datetime.now(timezone.utc).isoformat()))
            self.plan['hashes'][str(receipt)] = digest(receipt)
            self.plan['iterations'] = iterations
            public_path = self.root/'workspace'/'task.json'
            public = read(public_path); public['iterations'] = iterations
            write(public_path, public)
            self.plan['hashes'][str(public_path)] = digest(public_path)
            for iteration in range(old+1, iterations+1):
                from ..core.target import Target
                target = Target(self.plan.get('task_id','shape_color_then_scale'), iteration)
                target.run_dir = self.root/target.key
                target.task_path = target.run_dir/'task.resolved.json'
                self.state['targets'].append(target.to_dict())
            self.state.update(status='running', agent_status='ready')
            write(self.root/'config'/'plan.json', self.plan)
            self.save()
            self.report()
            return self.get_run_plan()
