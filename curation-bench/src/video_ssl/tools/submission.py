"""Submission validation, baseline checks and immutable archives."""
import hashlib
import json
from pathlib import Path
import random
import shutil
from ..core.runtime import read, write, verify, digest, tree_hash


class SubmissionTools:
    def _tool_audit(self, path):
        if self.target['state'] != 'active':
            raise ValueError('Audit requires active strategy stage')
        return self.run.worker('audit', self.directory, dataset=Path(path).resolve())


    def _tool_submit(self, path):
        if self.target['state'] == 'submitted':
            self.verify_submission()
            return {'status':'submitted', 'submission_path':str(self.submission)}
        if self.target['state'] != 'active':
            raise ValueError('Each iteration accepts exactly one submission')
        if self.run.time_budget()['expired']:
            raise TimeoutError('Strategy timeout reached')
        source = Path(path).resolve()
        workspace = self.run.root / 'workspace'
        code = workspace / 'curation'
        if self.target['iteration'] == 1:
            if digest(code / 'curate.py') != self.run.plan['baseline_builder_sha256']:
                raise ValueError('First iteration must use the unchanged random baseline builder')
            config = read(self.run.root / 'config' / 'backend.json')
            rows = [json.loads(s) for s in (source / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
            rng = random.Random(f"{config['build_seed']}:subset")
            expected = rng.sample(sorted(config['sources']), config['rows_per_trial'])
            rng.shuffle(expected)
            if [r['source_video_id'] for r in rows] != expected or any(r['input_frames'] != [0] or r['target_frame'] != config['sources'][r['source_video_id']]['frame_count']-1 for r in rows):
                raise ValueError('Baseline data must use the frozen random selection/order and F0-to-final')
        else:
            hypothesis = code / 'hypothesis.md'
            if not hypothesis.exists() or not hypothesis.read_text(encoding='utf-8').strip():
                raise ValueError('Write a fresh curation/hypothesis.md before submitting')
        code_hash = tree_hash(code)
        if sum((code / name).stat().st_size for name in code_hash) > 5_000_000:
            raise ValueError('Code snapshot exceeds 5 MB')
        data_hash = tree_hash(source)  # reject symlinks before copying
        limit = read(self.run.root/'config'/'backend.json').get('max_dataset_bytes',500_000_000)
        if sum((source / name).stat().st_size for name in data_hash) > limit:
            raise ValueError('Full submitted directory exceeds data byte budget')
        self.run.worker('audit', self.directory, dataset=source)
        self.target['state'] = 'submitting'
        self.run.save()
        shutil.copytree(code, self.directory / 'code', ignore=shutil.ignore_patterns('__pycache__','.git'))
        shutil.copytree(source, self.submission)
        self.run.worker('audit', self.directory, dataset=self.submission)
        seal = {'code':tree_hash(self.directory/'code'), 'data':tree_hash(self.submission)}
        write(self.directory / 'submission.json', seal)
        self.target.update(state='submitted', code_sha256=hashlib.sha256(json.dumps(seal['code'],sort_keys=True).encode()).hexdigest())
        return {'status':'submitted', 'submission_path':str(self.submission)}


    @property
    def submission(self):
        return self.directory / 'curated' / 'final_submission'


    def verify_submission(self):
        seal = read(self.directory / 'submission.json')
        if seal != {'code':tree_hash(self.directory/'code'), 'data':tree_hash(self.submission)}:
            raise ValueError('Accepted code/data changed')
        if self.target.get('checkpoint_record_sha256'):
            record = self.directory/'checkpoint.json'
            if digest(record) != self.target['checkpoint_record_sha256']:
                raise ValueError('Checkpoint record changed')
            checkpoint = read(record)
            if digest(checkpoint['path']) != checkpoint['sha256']:
                raise ValueError('Checkpoint content changed')
