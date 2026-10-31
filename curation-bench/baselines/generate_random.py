"""Generate the fixed seeded random baseline using the repository starter."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'curation'/'curate.py'),run_name='__main__')
