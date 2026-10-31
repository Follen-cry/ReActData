"""Single-task v1 scorer adapter; exceptions fail evaluation rather than become zeros."""
import json
from pathlib import Path
import cv2
import analogy_grid as G
from storage import write

def item(f0,cand,gt):
    im=cv2.imread(str(cand)) if Path(cand).is_file() else None
    if im is None or im.max()<8 or im.std()<1:
        return dict(score=0.,valid=False,sub={k:0. for k in G.SUBS},categories=['invalid'])
    sub,dbg=G.score_images('shape_color_then_scale',G.load(f0),G.load(im),G.load(gt))
    return dict(score=G.overall(sub),valid=True,sub={k:float(v) for k,v in sub.items()},categories=G.categories(sub,dbg))
def score(rows,out):
    items=[dict(id=r['id'],**item(r['image'][0],out/'images'/f"{r['id']}.png",r['target_image'])) for r in rows]
    (out/'per_item.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
    result=dict(mean=sum(r['score'] for r in items)/len(items),n=len(items),n_invalid=sum(not r['valid'] for r in items),per_item=str(out/'per_item.jsonl'))
    write(out/'metrics.json',result);return result
