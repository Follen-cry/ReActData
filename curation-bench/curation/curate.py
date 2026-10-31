"""Baseline: seeded sampling without replacement, followed by seeded ordering.
Baseline uses F0 -> final; improvements follow task.json frame_policy.
Use frame_policy.source_frame and training_text for frame-aware candidates.
"""
import argparse,json,random,sys
from pathlib import Path


def build(config, output):
    from video import source_endpoints
    out = Path(output)
    (out / 'images').mkdir(parents=True, exist_ok=False)
    rng = random.Random(f"{config['build_seed']}:subset")
    selected = rng.sample(sorted(config['sources']), config['rows_per_trial'])
    rng.shuffle(selected)
    rows = []
    for i, source in enumerate(selected):
        src = config['sources'][source]
        first, last, count = source_endpoints(src)
        a, b = f'images/{i}_f0.png', f'images/{i}_final.png'
        first.save(out / a); last.save(out / b)
        rows.append(dict(sample_id=str(i), source_video_id=source, input_images=[a], target_image=b,
                         text=src['eval_text'], input_frames=[0], target_frame=count-1))
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--tools',required=True)
    a=ap.parse_args(); sys.path.insert(0,str(Path(a.tools).resolve()))
    build(json.loads(Path(a.config).read_text(encoding='utf-8')),a.output)
