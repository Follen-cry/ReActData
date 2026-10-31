"""Load the configured cvlib file rather than a historical server location."""
import importlib.util
from pathlib import Path
import sys


def load_cvlib(path):
    path=Path(path).resolve()
    if not path.is_file():
        raise ValueError('Configured scorer_cvlib file is missing')
    spec=importlib.util.spec_from_file_location('cvlib',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules['cvlib']=module
    sys.path.insert(0,str(path.parent))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop('cvlib',None)
        raise
    finally:
        sys.path.remove(str(path.parent))
    return module
