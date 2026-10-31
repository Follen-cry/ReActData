"""Reference evaluation, per-round dev and host-owned final test."""
import math
from ..core.runtime import read, write, verify, digest, exclusive


class EvaluationTools:
    def _tool_eval(self):
        if self.target['state'] == 'completed':
            self.verify_submission()
            return read(self.directory/'eval_result.json')
        if self.target['state'] != 'trained':
            raise ValueError('Training must complete before dev; interrupted evaluation is not replayed')
        self.verify_submission()
        self.target['state'] = 'evaluating'
        self.run.save()
        result = self.run.worker('eval', self.directory, checkpoint=self.directory/'checkpoint.json')
        verify(self.run.plan)
        self.verify_submission()
        score = result.get('dev', {}).get('mean')
        if result.get('status') != 'completed' or type(score) not in (int,float) or not math.isfinite(score):
            raise ValueError('Dev must return a finite measured mean')
        write(self.directory/'eval_result.json', result)
        self.run.mark_target_completed(result)
        return result



def preinit(suite):
    with exclusive(suite.root):
        suite.state=read(suite.root/'suite_state.json')
        verify(suite.plan)
        phase = suite.state['preinit']
        if phase in ('completed','disabled'):
            return suite.state.get('preinit_result', {'status':phase})
        if phase != 'pending':
            raise ValueError('Preinit was interrupted/failed; no automatic replay')
        suite.state['preinit'] = 'running'
        suite.save()
        try:
            result = suite.worker('preinit', suite.root / 'preinit')
            verify(suite.plan)
            suite.state.update(preinit='completed', preinit_result=result)
            return result
        except BaseException as exc:
            suite.state.update(preinit='failed', error=str(exc))
            raise
        finally:
            suite.save()
            suite.report()



def finalize(run):
    if run.plan.get('kind')=='multi_task':
        if run.get_run_plan()['status']!='search_completed':
            raise ValueError('Finish all task searches before final test')
        result={child.plan['task_id']:finalize(child) for child in run.children()}
        run.report()
        return dict(status='completed',tasks=result)
    from .session import Session
    with exclusive(run.root):
        run.state=read(run.root/'suite_state.json')
        verify(run.plan)
        if run.state['final'] == 'completed':
            return read(run.root/'final'/'result.json')
        if run.state['status'] != 'search_completed' or run.state['final'] != 'pending':
            raise ValueError('Finish search before test; interrupted/failed test is never replayed')
        completed = [t for t in run.state['targets'] if t['state']=='completed']
        if not completed:
            raise ValueError('No completed candidate')
        selected = max(completed, key=lambda t:t['dev']['mean'])
        run.state['active'] = run.state['targets'].index(selected)
        session = Session(run)
        session.verify_submission()
        run.state.update(final='running', selected=selected['key'])
        run.save()
        try:
            result = run.worker('test', run.root/'final', checkpoint=session.directory/'checkpoint.json')
            verify(run.plan)
            session.verify_submission()
            if result.get('status') != 'completed':
                raise ValueError('Test did not complete')
            result['selected'] = selected['key']
            write(run.root/'final'/'result.json', result)
            run.state['final'] = 'completed'
            return result
        except BaseException as exc:
            run.state.update(final='failed', error=str(exc))
            raise
        finally:
            run.save()
            run.report()
