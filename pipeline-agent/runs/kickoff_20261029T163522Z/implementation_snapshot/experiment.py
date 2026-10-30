import json,re,shutil
from pathlib import Path
from storage import read,write,inventory,lock,digest
from dataset import validate,local
import training
import campaign

BASE=Path(__file__).resolve().parent

def init(root,config_path):
    root=Path(root)
    from task import resolve
    c=resolve(config_path)
    root.mkdir(parents=True,exist_ok=False)
    (root/"workspace").mkdir(); (root/"agent_logs").mkdir()
    shutil.copy2(BASE/"starter/build_dataset.py",root/"workspace/build_dataset.py")
    shutil.copytree(BASE/"protocols",root/"workspace/protocols")
    write(root/"config.json",c);write(root/"workspace/task.json",c)
    write(root/"state.json",{"mode":c["mode"],"trials":{},"requests":{},"selection":None,"config_sha256":digest(root/"config.json")})
    return {"status":"initialized","simulated":c["mode"]=="simulation","artifacts":{"root":str(root)}}

def run(root,dataset,key):
    root=Path(root)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",key): raise ValueError("invalid request ID")
    with lock(root):
        s=read(root/"state.json");c=read(root/"config.json")
        if key in s["requests"]: return s["trials"][s["requests"][key]]
        if digest(root/"config.json")!=s["config_sha256"]: raise ValueError("configuration is immutable")
        if s["selection"]: raise ValueError("selection already locked")
        if len(s["trials"])>=c["max_trials"]: raise ValueError("trial budget exhausted")
        if s["mode"]!=c["mode"]: raise ValueError("mode cannot change")
        rows=validate(dataset,c)
        tid=f"t{len(s['trials'])+1:04d}";t=root/"trials"/tid
        if c["mode"] == "real": campaign.register(c["campaign"],root,tid,key,c["estimated_trial_seconds"])
        record={"trial_id":tid,"role":"baseline" if not s["trials"] else "improvement","status":"preparing","simulated":c["mode"]=="simulation","error":None,"artifacts":{"trial":str(t)}}
        s["requests"][key]=tid;s["trials"][tid]=record;write(root/"state.json",s)
        try:
            t.mkdir(parents=True)
            code_bytes=0
            for p in (root/"workspace").rglob("*"):
                if p.is_symlink(): raise ValueError("workspace symlink")
                if p.is_file() and "dataset" not in p.relative_to(root/"workspace").parts:
                    code_bytes+=p.stat().st_size
                    if code_bytes>5000000: raise ValueError("code archive exceeds 5 MB")
            # Archive all code/notes, excluding generated datasets by convention.
            shutil.copytree(root/"workspace",t/"code",ignore=shutil.ignore_patterns("dataset","__pycache__"))
            write(t/"config.json",c)
            dest=t/"dataset";dest.mkdir()
            write_manifest=[]
            for r in rows:
                for name in r["input_images"]+[r["target_image"]]:
                    target=local(dest,name);target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(local(Path(dataset).resolve(),name),target)
                write_manifest.append(r)
            (dest/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
            validate(dest,c) # detect workspace races while copying
            write(t/"integrity.json",{"dataset":inventory(dest),"code":inventory(t/"code"),"config":inventory(t)["config.json"]})
            record["status"]="running";write(root/"state.json",s)
            if c["mode"] == "real": campaign.update(c["campaign"],root,key,"running")
            record["result"]=training.train_and_dev(t,c)
            record["status"]="completed"
        except Exception as e:
            record.update(status="failed",error=str(e))
        if c["mode"] == "real": campaign.update(c["campaign"],root,key,record["status"])
        write(root/"state.json",s);write(t/"result.json",record)
        return record

def finalize(root,tid):
    root=Path(root)
    with lock(root):
        s=read(root/"state.json");c=read(root/"config.json")
        if s["mode"]!=c["mode"]: raise ValueError("mode changed")
        if s["selection"]:
            if s["selection"]!=tid: raise ValueError("selection locked")
            if (root/"final.json").exists(): return read(root/"final.json")
            raise RuntimeError("finalization interrupted: manual inspection required; no automatic test retry")
        r=s["trials"][tid]
        if r["status"]!="completed" or r["simulated"]!=(c["mode"]=="simulation"): raise ValueError("invalid completed trial")
        t=root/"trials"/tid;h=read(t/"integrity.json")
        from storage import digest
        if inventory(t/"dataset")!=h["dataset"] or inventory(t/"code")!=h["code"] or digest(t/"config.json")!=h["config"]: raise ValueError("trial modified")
        if read(t/"config.json")!=c: raise ValueError("experiment configuration changed")
        s["selection"]=tid;write(root/"state.json",s)
        try:
            result={"status":"finalized","trial_id":tid,"simulated":r["simulated"],
                    "dev":r["result"]["dev"],"checkpoint":r["result"]["checkpoint"],
                    "integrity":h,"selection_policy":c["selection"],
                    "seeds":{k:c[k] for k in ("model_init_seed","train_seed","build_seed","eval_seed")},
                    "test":training.test(t,c)}
        except Exception as e:
            result={"status":"failed","trial_id":tid,"error":str(e)}
        write(root/"final.json",result)
        (root/"report.md").write_text("# Experiment result\n\n"+json.dumps(result,indent=2)+"\n",encoding="utf-8")
        return result
