"""Compatibility script; implementation: support/portable_flow.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "support/portable_flow.py"), run_name="__main__")
