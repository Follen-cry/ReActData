"""Selected research protocol and optional upstream-style guidance appendices."""
from importlib.resources import files
from pathlib import Path


def protocol_text():
    return files('video_ssl').joinpath('protocols/method.md').read_text(encoding='utf-8')


def load_protocol(path):
    path=Path(path).resolve()
    directory=Path(__file__).resolve().parents[3]/'protocols'
    if path.parent==directory and path.name in ('instruction.md','skill.md'):
        return (directory/'method.md').read_text(encoding='utf-8')+'\n\n'+path.read_text(encoding='utf-8')
    return path.read_text(encoding='utf-8')


def workspace_instructions(plan):
    """Keep research discovery inside the assigned task, not the repository history."""
    task=plan.get('task_id','shape_color_then_scale')
    skills=Path(__file__).resolve().parents[3]/'skills'
    return f"""# Research workspace: {task}

Start with this workspace's METHOD.md, COMMANDS.md, task.json and NOTES.md.
METHOD.md is the selected, frozen protocol; do not substitute a repository protocol.
Use the exact COMMANDS.md CLI/build commands and continue the saved task stage.

Read the declared task.json training resources, dev_feedback.jsonl, and this task's
prior code, predictions, metrics and reports returned by the task CLI. Read backend
helper files only as needed to construct a valid candidate. Inspect a framework file
only to diagnose a specific CLI error; do not inventory the repository to discover
the workflow. Framework fixes and missing dependencies are host responsibilities.

Do not browse docs/history, legacy archives, tests, Git history, environment/cache
contents, other protocol variants or other tasks/suites. Repository setup and
migration documents are operator/developer material, not research instructions.
Only when METHOD.md explicitly selects skill guidance may you read the read-only
research library at {skills}. A skill card never overrides the frozen task contract.

Edit only this workspace's curation/ program and research notes, including NOTES.md.
Preserve accepted submissions and checkpoints. Do not access test feedback or run
finalize, resume, extend, or another Agent/suite. Continue the authorized rounds
until task_done or a protocol-defined host blocker.
"""
