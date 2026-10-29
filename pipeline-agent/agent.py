"""Direct Codex execution; Markdown rules are agreements, not isolation."""
import json
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from storage import read, write


def command(resume=None, executable="codex", model=None):
    args = [executable, "exec"]
    if resume:
        args += ["resume", resume]
    args += ["--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json"]
    if model:
        args += ["--model", model]
    return args + ["-"]


def launch(root, resume=None, executable="codex", model=None):
    root = Path(root).resolve()
    read(root / "state.json")
    workspace = root / "workspace"
    base = Path(__file__).resolve().parent
    # Refresh protocols for experiments initialized by an earlier version.
    protocol = (base / "protocols/agent.md").read_text(encoding="utf-8")
    (workspace / "protocols/agent.md").write_text(protocol, encoding="utf-8")
    cli = shlex.join([sys.executable, str(base / "cli.py"), "--root", str(root)])
    build = shlex.join([sys.executable, str(workspace / "build_dataset.py"),
                       "--config", str(workspace / "task.json"), "--tools", str(base),
                       "--output", str(workspace / "dataset")])
    (workspace / "COMMANDS.md").write_text(
        f"# Commands (Linux shell)\n\nCLI prefix: `{cli}`\n\nBuild dataset: `{build}`\n\n"
        f"Example: `{cli} run --dataset {shlex.quote(str(workspace / 'dataset'))} --request-id baseline`\n\n"
        "Append task, validate --dataset PATH, or status [TRIAL_ID] to the CLI prefix.\n",
        encoding="utf-8")
    logdir = root / "agent_logs" / uuid.uuid4().hex
    logdir.mkdir(parents=True)
    args = command(resume, executable, model)
    prompt = ("Read protocols/agent.md, COMMANDS.md, task.json and existing NOTES.md. "
              "Continue this experiment using its CLI; maintain your own notes. "
              "When finished, write selection.txt with a completed trial ID. Do not finalize.")
    record = {"status": "running", "isolation": "protocol_only", "resume": resume,
              "command": args, "artifacts": {"logs": str(logdir)}}
    write(logdir / "launch.json", record)
    try:
        with (logdir / "events.jsonl").open("w", encoding="utf-8") as stdout, \
                (logdir / "stderr.log").open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(args, cwd=workspace, input=prompt, text=True,
                                  encoding="utf-8", stdout=stdout, stderr=stderr)
        record.update(status="completed" if proc.returncode == 0 else "failed",
                      returncode=proc.returncode,
                      error=None if proc.returncode == 0 else "Codex failed; see stderr.log and events.jsonl")
    except OSError as exc:
        record.update(status="failed", error=str(exc))
    except KeyboardInterrupt:
        record.update(status="failed", error="Agent interrupted; inspect logs before resuming")
        write(logdir / "launch.json", record)
        raise
    # Use an explicit session ID for resume; never select a global latest session.
    for line in (logdir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started":
            record["session_id"] = event.get("thread_id")
    write(logdir / "launch.json", record)
    return record
