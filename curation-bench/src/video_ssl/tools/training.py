"""Training stage orchestration over the configured backend."""
from ..core.runtime import read, write, verify, digest


class TrainingTools:
    def _tool_finetune(self):
        if self.target['state'] in ('trained','completed'):
            self.verify_submission()
            result = read(self.directory / 'train_result.json')
            if digest(result['checkpoint']['path']) != result['checkpoint']['sha256']:
                raise ValueError('Checkpoint changed')
            return result
        if self.target['state'] != 'submitted':
            raise ValueError('Submit once before training; interrupted/failed training is never replayed')
        self.verify_submission()
        self.target['state'] = 'training'
        self.run.save()
        result = self.run.worker('train', self.directory, dataset=self.submission)
        verify(self.run.plan)
        self.verify_submission()
        if result.get('status') != 'completed':
            raise RuntimeError('Worker did not complete training')
        write(self.directory/'train_result.json', result)
        write(self.directory/'checkpoint.json', result['checkpoint'])
        self.target.update(state='trained', checkpoint_record_sha256=digest(self.directory/'checkpoint.json'))
        return result
