"""Baseline: seeded sampling without replacement, followed by seeded ordering.
Agent may change selection/order only. Source text and endpoints are fixed.
"""
import argparse,json,random,sys,re
from pathlib import Path


def build(config, output):
    from video import source_endpoints
    out = Path(output)
    (out / 'images').mkdir(parents=True, exist_ok=False)
    rng = random.Random(f"{config['build_seed']}:subset")
    pool = sorted(config['sources'])
    rng.shuffle(pool)
    sizes = {'small': 0, 'medium': 1, 'large': 2, 'extra_large': 3}
    def target_size(source):
        return sizes[re.search(r'then change size to (\w+)\.', config['sources'][source]['description'])[1]]
    buckets = {size: [source for source in pool if target_size(source) == size][:32]
               for size in range(4)}
    selected = []
    for i in range(32):
        order = list(range(4))
        rng.shuffle(order)
        selected.extend(buckets[size][i] for size in order)
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
