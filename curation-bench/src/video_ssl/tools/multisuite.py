"""Upstream-style task queue: isolated task workspaces and one Agent per task."""
from copy import deepcopy
from pathlib import Path
from ..core.runtime import read, write, exclusive
from .suite import SuiteController


class MultiTaskSuite(SuiteController):
    @classmethod
    def create_from_plan(cls, prompt, root, plan):
        root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=False)
        write(root/'config'/'plan.json',plan)
        write(root/'suite_state.json',dict(kind='multi_task',status='initialized',active_task=0,targets=[],final='pending'))
        suite=cls(root)
        for task_id in plan['task_order']:
            child_plan=deepcopy(plan['tasks'][task_id]); child_plan['suite_parent']=str(root)
            SuiteController.create(prompt,root/'tasks'/task_id,plan=child_plan)
        suite.report()
        return suite

    def children(self):
        return [SuiteController(self.root/'tasks'/name) for name in self.plan['task_order']]

    def selected(self):
        assigned=__import__('os').environ.get('VIDEO_SSL_CURRENT_TASK')
        if assigned:
            if assigned not in self.plan['task_order']:
                raise ValueError('Assigned task is not in this suite')
            return SuiteController(self.root/'tasks'/assigned)
        return self.children()[min(self.state['active_task'],len(self.plan['task_order'])-1)]

    def verify_inputs(self):
        for child in self.children(): child.verify_inputs()

    def check_declaration(self, prompt_path, profile_path=None):
        for child in self.children(): child.check_declaration(prompt_path,profile_path)

    def get_run_plan(self):
        children=self.children()
        targets=[dict(t,suite_dir=str(child.root)) for child in children for t in child.state['targets']]
        self.state['targets']=targets
        if all(c.state['status']=='search_completed' for c in children): self.state['status']='search_completed'
        elif any(c.state['status']!='initialized' for c in children): self.state['status']='running'
        if all(c.state['final']=='completed' for c in children): self.state['final']='completed'
        return dict(self.state,suite_dir=str(self.root),total_targets=len(targets),
                    completed=sum(t['state']=='completed' for t in targets),
                    tasks={name:child.get_run_plan() for name,child in zip(self.plan['task_order'],children)})

    def report(self):
        result=self.get_run_plan(); self.save()
        rows=['# Video SSL Multi-Task Suite','','| Task | Search | Final test | Report |','|---|---|---|---|']
        for name,child in zip(self.plan['task_order'],self.children()):
            child.report()
            rows.append(f"| {name} | {child.state['status']} | {child.state['final']} | [Report](tasks/{name}/README.md) |")
        (self.root/'README.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
        write(self.root/'suite_result.json',result)
        return dict(status=result['status'],report=str(self.root/'README.md'))

    def prepare_resume(self, *, mutate=True):
        for child in self.children(): child.prepare_resume(mutate=False)
        if mutate:
            for child in self.children(): child.prepare_resume(mutate=True)
        return self.get_run_plan()

    def next(self):
        with exclusive(self.root):
            if self.state['active_task']>=len(self.plan['task_order']):
                return dict(status='task_done',run_dir=str(self.root))
            child=self.selected()
            result=child.begin_next_target()
            if result['status']=='task_done' and not __import__('os').environ.get('VIDEO_SSL_CURRENT_TASK'):
                self.state['active_task']+=1
                if self.state['active_task']<len(self.plan['task_order']):
                    child=self.selected()
                    if child.state['preinit']=='pending':
                        result=dict(status='preinit_required',task_id=child.plan['task_id'])
                    else: result=child.begin_next_target()
            self.report()
            return result

    def begin_next_target(self): return self.next()
    def get_active_session(self, **kwargs): return self.selected().get_active_session(**kwargs)
    def task(self): return self.selected().task()
    def time_budget(self): return self.selected().time_budget()
    def preinit(self): return self.selected().preinit()
    def fail(self, reason): return self.selected().fail(reason)
    def extend(self, iterations):
        raise ValueError('Extend an individual task with --suite-dir SUITE/tasks/TASK; shared budgets still apply')
