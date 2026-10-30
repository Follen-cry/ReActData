import datetime, json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import campaign
from storage import write, read

def test_budget_and_reserve(tmp_path):
    p=tmp_path/'campaign.json'
    write(p,dict(deadline=(campaign.utc()+datetime.timedelta(hours=8)).isoformat(),training_budget=1,max_concurrent_gpus=4,report_reserve_seconds=5400,training_attempts=[],consumed=0,remaining=1,gpu_allocation=[0]))
    campaign.register(p,tmp_path/'one','t0001','a',1)
    with pytest.raises(ValueError): campaign.register(p,tmp_path/'two','t0001','b',1)
    campaign.update(p,tmp_path/'one','a','failed')
    assert read(p)['remaining']==0
    with pytest.raises(ValueError): campaign.register(p,tmp_path/'two','t0001','b',1)
    c=read(p);c.update(training_budget=7,training_attempts=[],deadline=campaign.utc().isoformat());write(p,c)
    with pytest.raises(ValueError): campaign.register(p,tmp_path/'two','t0001','c',1)

def test_timeout_process_group(tmp_path):
    p=tmp_path/'campaign.json'
    write(p,dict(deadline=(campaign.utc()+datetime.timedelta(seconds=1)).isoformat(),max_concurrent_gpus=4,gpu_allocation=[0]))
    with pytest.raises(Exception): campaign.run_process(p,[sys.executable,'-c','import time; time.sleep(20)'],tmp_path/'log')

def test_timeout_kills_descendants(tmp_path):
    import subprocess, time
    p=tmp_path/'campaign.json';marker=tmp_path/'survived'
    write(p,dict(deadline=(campaign.utc()+datetime.timedelta(seconds=1)).isoformat(),max_concurrent_gpus=4,gpu_allocation=[0]))
    child="import time,pathlib;time.sleep(2);pathlib.Path("+repr(str(marker))+").touch()"
    parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);time.sleep(20)"
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.run_process(p,[sys.executable,'-c',parent],tmp_path/'log')
    time.sleep(1.5)
    assert not marker.exists()
