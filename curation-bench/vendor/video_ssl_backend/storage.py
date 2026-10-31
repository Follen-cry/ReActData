import json, hashlib, os
from pathlib import Path
from contextlib import contextmanager

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    os.replace(tmp,p)
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory(root):
    return {str(p.relative_to(root)):digest(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}
@contextmanager
def lock(root):
    p=Path(root)/"operation.lock"
    fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    try: yield
    finally: p.unlink()
