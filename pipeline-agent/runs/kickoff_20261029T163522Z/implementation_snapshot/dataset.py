"""Self-contained image submissions, verified against configured source endpoints."""
import json
from pathlib import Path
from PIL import Image, ImageChops
from video import source_endpoints
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
    ids=set(); sources=set(); total=manifest.stat().st_size; cache={}
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
        if r["text"]!=src["eval_text"]: raise ValueError("fixed evaluation text required")
        if source not in cache: cache[source]=source_endpoints(src)
        first,last,n=cache[source]
        if r["input_frames"]!=[0] or r["target_frame"]!=n-1: raise ValueError("only F0 -> final supported")
        if not isinstance(r["text"],str) or not 1<=len(r["text"])<=config["max_text_chars"]: raise ValueError("text budget")
        if not isinstance(r["input_images"],list) or len(r["input_images"])!=1: raise ValueError("one ordered F0 input required")
        for name,expected in zip(r["input_images"]+[r["target_image"]],[first,last]):
            path=local(root,name); total+=path.stat().st_size
            if total>config["max_dataset_bytes"]: raise ValueError("dataset byte budget")
            with Image.open(path) as image:
                image.load(); actual=image.convert("RGB")
                if actual.size!=expected.size or ImageChops.difference(actual,expected).getbbox():
                    raise ValueError("image does not match declared source frame")
    return rows
