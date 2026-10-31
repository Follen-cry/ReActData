from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


def resolve_uv_command() -> list[str]:
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path]

    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]

    raise RuntimeError(
        "uv is required for method CLI, construction and worker execution. "
        "Install uv and prepare this project with `uv sync --locked`."
    )


def project_command(project: str | Path, *args: str, uv: list[str] | None = None,
                    python: str | None = None) -> list[str]:
    """Use the prepared locked CLI project, never synchronize during research."""
    command=[*(uv if uv is not None else resolve_uv_command()),'run','--frozen','--no-sync',
             '--no-python-downloads','--project',str(Path(project).resolve())]
    if python:
        command+=['--python',str(python)]
    return command+['--',*map(str,args)]


def worker_command(python: str, script: str | Path, *args: str,
                   uv: list[str] | None = None) -> list[str]:
    """Wrap the exact existing GPU interpreter, without syncing CLI dependencies.

    The absolute Python is both the requested interpreter and the command. This
    preserves its site-packages even when a different CLI venv is active.
    """
    # --no-project already disables project synchronization; --no-sync here
    # would be redundant and makes uv emit a warning on every worker call.
    return [*(uv if uv is not None else resolve_uv_command()),'run','--no-project','--no-config',
            '--no-python-downloads','--python',str(python),'--',str(python),str(script),*map(str,args)]
