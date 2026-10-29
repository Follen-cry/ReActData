"""Compatibility script; implementation: integration/test_sandbox.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "integration/test_sandbox.py"), run_name="__main__")
