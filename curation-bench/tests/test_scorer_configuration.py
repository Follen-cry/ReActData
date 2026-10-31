import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def test_configured_scorer_file_wins_over_an_existing_module(tmp_path,monkeypatch):
    backend=Path(__file__).resolve().parents[1]/'vendor/video_ssl_backend'
    spec=importlib.util.spec_from_file_location('method_scorer_loader',backend/'scorer_loader.py')
    loader=importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    configured=tmp_path/'named_scorer.py'
    configured.write_text("SOURCE = 'configured resource'\n",encoding='utf-8')
    monkeypatch.setitem(sys.modules,'cvlib',SimpleNamespace(SOURCE='wrong import'))
    loaded=loader.load_cvlib(configured)
    assert loaded.SOURCE=='configured resource'
    assert sys.modules['cvlib'] is loaded
