"""Persistent per-target Session, adapted from upstream curation-bench."""
import json
import time
from ..core.runtime import read, write, digest, verify, exclusive
from .submission import SubmissionTools
from .training import TrainingTools
from .evaluation import EvaluationTools


class Session(SubmissionTools, TrainingTools, EvaluationTools):
    fields = ('state','code_sha256','checkpoint_record_sha256','dev','error','eval_result')

    def __init__(self, run, *, validate=True):
        self.run = run
        self.target = run.active()
        self.directory = run.target_dir()
        self.events_path = self.directory/'events.jsonl'
        self.state_path = self.directory/'session_state.json'
        self.event_seq = 0
        if validate:
            self.load_state()

    def summarize_task(self):
        return self.run.task()

    def get_time_budget(self):
        return self.run.time_budget()

    def load_state(self):
        if not self.state_path.exists():
            if self.target.get('session_state_sha256'):
                raise ValueError('Session state missing; host inspection required')
            return
        if digest(self.state_path) != self.target.get('session_state_sha256'):
            raise ValueError('Session state changed; host inspection required')
        state = read(self.state_path)
        # In-flight state is rejected by the operation guards, never reset.
        if self.target['state'] not in ('training','evaluating','submitting'):
            if state['target'] != {k:self.target.get(k) for k in self.fields}:
                raise ValueError('Session/suite state disagree; host inspection required')
        self.event_seq = state['event_seq']

    def save_state(self):
        write(self.state_path, dict(event_seq=self.event_seq, target_key=self.target['key'],
                                   target={k:self.target.get(k) for k in self.fields}))
        self.target['session_state_sha256'] = digest(self.state_path)
        self.run.save()

    def _log_event(self, tool_name, args, result):
        self.event_seq += 1
        with self.events_path.open('a',encoding='utf-8') as file:
            file.write(json.dumps(dict(seq=self.event_seq,time=time.time(),tool=tool_name,args=args,result=result),ensure_ascii=False)+'\n')

    def dispatch(self, tool, **args):
        with exclusive(self.run.root):
            expected=self.target['key']
            self.run.state=read(self.run.root/'suite_state.json')
            self.target=self.run.active()
            if self.target['key']!=expected:
                raise ValueError('Active target changed; inspect the latest task context')
            self.load_state()
            verify(self.run.plan)
            try:
                result = getattr(self, '_tool_' + tool)(**args)
                return result
            except BaseException as exc:
                if self.target['state'] in ('training','evaluating','submitting'):
                    self.target.update(state='failed', error=str(exc))
                result = {'status':'error', 'error':str(exc)}
                raise
            finally:
                self._log_event(tool,args,locals().get('result',{'status':'interrupted'}))
                self.save_state()
                self.run.report()
