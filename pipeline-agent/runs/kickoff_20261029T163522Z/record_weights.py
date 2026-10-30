import pathlib,hashlib,json,time
out=pathlib.Path(__file__).resolve().parent
root=pathlib.Path('/scratch/network/ssd2/junlin/models/InternVL-U').resolve(); records={}; start=time.monotonic()
for p in sorted(root.rglob('*')):
 if p.is_file() and p.suffix in ('.json','.bin','.safetensors','.txt','.model'):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
  records[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
(out/'model_weights_inventory.json').write_text(json.dumps({'root':str(root),'files':records,'seconds':time.monotonic()-start},indent=2)+'\n')
print(len(records),'files hashed')
