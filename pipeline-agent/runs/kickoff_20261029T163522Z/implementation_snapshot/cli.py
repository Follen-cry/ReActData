import argparse,json
from pathlib import Path
import experiment,agent
from dataset import validate
from storage import read

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True,help="runs/<experiment_id>")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("init").add_argument("--config",required=True)
    sub.add_parser("task")
    sub.add_parser("validate").add_argument("--dataset",required=True)
    r=sub.add_parser("run");r.add_argument("--dataset",required=True);r.add_argument("--request-id",required=True)
    sub.add_parser("status").add_argument("trial",nargs="?")
    ag=sub.add_parser("agent")
    ag.add_argument("--resume",help="Explicit Codex session ID")
    ag.add_argument("--executable",default="codex",help="Codex executable path")
    ag.add_argument("--model",help="Optional override; otherwise use Codex configuration")
    sub.add_parser("finalize").add_argument("--trial",required=True)
    a=p.parse_args();root=Path(a.root).resolve()
    try:
        if a.command=="init": result=experiment.init(root,a.config)
        elif a.command=="task": result={"status":"ok","task":read(root/"workspace/task.json")}
        elif a.command=="validate": result={"status":"valid","samples":len(validate(a.dataset,read(root/"config.json")))}
        elif a.command=="run": result=experiment.run(root,a.dataset,a.request_id)
        elif a.command=="finalize": result=experiment.finalize(root,a.trial)
        elif a.command=="agent": result=agent.launch(root,a.resume,a.executable,a.model)
        else:
            state=read(root/"state.json");result=state["trials"][a.trial] if a.trial else {"status":"ok","state":state,"remaining_trials":read(root/"config.json")["max_trials"]-len(state["trials"])}
    except Exception as e: result={"status":"error","error":str(e)}
    print(json.dumps({"trial_id":None,"error":None,"artifacts":{},**result},ensure_ascii=True,indent=2))
    return 1 if result.get("status") in ("error","failed") else 0
if __name__=="__main__": raise SystemExit(main())
