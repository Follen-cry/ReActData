import cv2, hashlib, json, pathlib, collections
p=pathlib.Path(__file__).resolve().parent
rows=[json.loads(l) for s in ('train','dev','test') for l in (p/f'{s}_sources.jsonl').read_text().splitlines()]
seen={}; collisions=[]; discrepancies=[]; params={}; param_cross=[]
for r in rows:
 for key in ('f0','final'):
  im=cv2.imread(r[key]); h=hashlib.sha256(im.tobytes()).hexdigest()
  if h in seen: collisions.append([seen[h], [r['id'],key]])
  else: seen[h]=[r['id'],key]
 cap=cv2.VideoCapture(r['video']); n=0
 while True:
  ok,_=cap.read()
  if not ok: break
  n+=1
 cap.release()
 if n!=r['frame_count']: discrepancies.append([r['id'],r['frame_count'],n])
 meta=json.loads((pathlib.Path(r['source_dir'])/'metadata.json').read_text())
 h=hashlib.sha256(json.dumps(meta.get('parameters',{}),sort_keys=True).encode()).hexdigest()
 if h in params and params[h][0]!=r['split']: param_cross.append([params[h],[r['split'],r['id']]])
 else: params.setdefault(h,[r['split'],r['id']])
a=json.loads((p/'data_audit.json').read_text());a.update(pixel_endpoint_collisions=collisions,decoded_frame_count_discrepancies=discrepancies,cross_split_identical_parameters=param_cross,full_decode_count=len(rows),note='SHA256 raw video bytes and decoded normalized endpoint pixels checked for duplicates; all frames decoded for counts; exact generator parameter JSON checked across splits. Video copies with changed encoding and different endpoints are not detected; prior historical exposure unknown.')
(p/'data_audit.json').write_text(json.dumps(a,indent=2)+'\n'); print(json.dumps({k:a[k] for k in ('counts','full_decode_count','pixel_endpoint_collisions','decoded_frame_count_discrepancies','cross_split_identical_parameters')}))
