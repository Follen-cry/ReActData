"""Self-contained submissions verified against frozen sources and frame policies."""
import json
from pathlib import Path
from PIL import Image, ImageChops
from frame_policy import policy, check_frames, source_frame, training_text
from storage import digest

def local(root,name):
    if not isinstance(name,str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("relative contained image path required")
    p=root/name
    if any(x.is_symlink() for x in [p,*p.parents]): raise ValueError("symlink forbidden")
    if not p.resolve().is_relative_to(root.resolve()): raise ValueError("path escape")
    return p

def validate(root,config):
    root=Path(root).resolve()
    manifest=local(root,"manifest.jsonl")
    if manifest.stat().st_size>config["max_dataset_bytes"]: raise ValueError("manifest byte budget")
    rows=[json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows)!=config["rows_per_trial"]: raise ValueError("sample budget")
    ids=set(); sources=set(); total=manifest.stat().st_size
    p = policy(config)
    for r in rows:
        sid=r["sample_id"]
        if not isinstance(sid,str) or not sid or sid in ids: raise ValueError("duplicate/invalid sample_id")
        ids.add(sid)
        source=r["source_video_id"]
        if source not in config["sources"]: raise ValueError("source not allowed")
        if source in sources: raise ValueError("video may only be used once")
        sources.add(source)
        src=config["sources"][source]
        for key in ("f0", "final"):
            if digest(src[key])!=config["source_hashes"][source][key]: raise ValueError("canonical source changed")
        check_frames(src, r['input_frames'], r['target_frame'], config)
        if p['mode'] == 'select':
            expected_hashes = config['source_hashes'][source]
            if digest(src['video']) != expected_hashes['video']: raise ValueError('source video changed')
            for index in r['input_frames'] + [r['target_frame']]:
                if str(index) in src.get('frames', {}):
                    if digest(src['frames'][str(index)]) != expected_hashes['frames'][str(index)]:
                        raise ValueError('canonical frame changed')
        if r['text'] != training_text(src, r['input_frames'], r['target_frame']):
            raise ValueError('fixed frame-conditioned training text required')
        if not isinstance(r["text"],str) or not 1<=len(r["text"])<=config["max_text_chars"]: raise ValueError("text budget")
        if not isinstance(r['input_images'], list) or len(r['input_images']) != len(r['input_frames']):
            raise ValueError('one image per declared input frame required')
        if p['mode'] == 'select' and r['text'].count('<image>') != len(r['input_frames']):
            raise ValueError('image token count must equal input frame count')
        for name,index in zip(r["input_images"]+[r["target_image"]],r['input_frames']+[r['target_frame']]):
            expected = source_frame(src, index)
            path=local(root,name); total+=path.stat().st_size
            if total>config["max_dataset_bytes"]: raise ValueError("dataset byte budget")
            with Image.open(path) as image:
                image.load(); actual=image.convert("RGB")
                if actual.size!=expected.size or ImageChops.difference(actual,expected).getbbox():
                    raise ValueError("image does not match declared source frame")
    return rows
