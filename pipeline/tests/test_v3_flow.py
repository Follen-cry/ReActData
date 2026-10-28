"""Compatibility script; implementation: integration/test_v3_flow.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "integration/test_v3_flow.py"), run_name="__main__")
