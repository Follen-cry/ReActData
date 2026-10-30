"""Host-only selection; run once after the search Agent has exited."""
import pathlib,json,sys,hashlib,datetime,subprocess,shlex
p=pathlib.Path(__file__).resolve().parent;e=p/'experiment';sys.path.insert(0,str(p.parents[1]))
from storage import read,write,digest
import campaign
launches=list((e/'agent_logs').glob('*/launch.json'))
assert launches and all(read(f)['status']=='completed' for f in launches),'Agent must exit successfully before test'
s=read(e/'state.json');assert s['selection'] is None,'already selected; inspect before any finalization'
assert not any(r['status'] in ('preparing','running') for r in s['trials'].values())
valid=[r for r in s['trials'].values() if r['status']=='completed' and not r['simulated']]
chosen=sorted(valid,key=lambda r:(-r['result']['dev']['mean'],r['role']!='baseline',r['trial_id']))[0]
tid=chosen['trial_id'];assert (e/'workspace/selection.txt').read_text().strip()==tid,'Agent/host selection disagreement'
for name,h in read(p/'freeze.json')['sha256'].items(): assert digest(name)==h, 'frozen resource changed: '+name
ck=chosen['result'];assert digest(ck['checkpoint'])==ck['checkpoint_sha256']
assert not (e/'trials'/tid/'test').exists(),'test artifacts already exist; no blind retry'
write(p/'selection.json',dict(selected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),trial_id=tid,dev=ck['dev'],checkpoint=ck['checkpoint'],checkpoint_sha256=ck['checkpoint_sha256'],policy='highest completed real dev.mean; ties baseline then earliest; no test feedback',candidate_scores={r['trial_id']:r['result']['dev']['mean'] for r in valid}))
with campaign.ledger(p/'campaign.json') as c:c['status']='finalizing'
args=[sys.executable,str(p.parents[1]/'cli.py'),'--root',str(e),'finalize','--trial',tid]
with (p/'commands.md').open('a') as f:f.write('\n## Host finalization command\n\n```sh\n'+shlex.join(args)+'\n```\n')
with (p/'finalize_cli.log').open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
print('selected',tid,'finalize returncode',r.returncode,flush=True)
raise SystemExit(r.returncode)
