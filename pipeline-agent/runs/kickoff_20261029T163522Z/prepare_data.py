"""Prepare a deterministic independent raw-data candidate split, with content audit."""
import cv2, glob, hashlib, importlib.util, json, random, pathlib, datetime
ROOT=pathlib.Path(__file__).resolve().parent
RAW='/scratch/network/ssd/junlin/vbvr_next_frame/raw/shape_color_then_scale/chunk_*/*/*'
WRAP='/scratch/network/ssd2/junlin/ssl_mllm/VBVR-DataGeneration/target_pred/vbvr_target_pred_prompt_wrap.py'
spec=importlib.util.spec_from_file_location('promptwrap',WRAP); w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
paths=sorted(glob.glob(RAW)); random.Random(20261029).shuffle(paths)
seen=set(); rows=[]; rejected=[]
for d in paths:
 if len(rows)>=260: break
 p=pathlib.Path(d); vid=p/'ground_truth.mp4'; f0=p/'first_frame.png'; fin=p/'final_frame.png'
 if not all(x.is_file() for x in (vid,f0,fin,p/'prompt.txt')): rejected.append([d,'missing']); continue
 vhash=sha(vid); a=sha(f0); b=sha(fin)
 if len({vhash,a,b})<3 or any(h in seen for h in (vhash,a,b)): rejected.append([d,'content duplicate']); continue
 cap=cv2.VideoCapture(str(vid)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok,frame=cap.read(); cap.release()
 if n<2 or not ok: rejected.append([d,'video unreadable']); continue
 prompt=(p/'prompt.txt').read_text().strip(); text=w.wrap_prompt('shape_color_then_scale',prompt)
 if w.find_residual_video_language(text.split('Task: ',1)[1]): rejected.append([d,'residual video language']); continue
 i=len(rows); split='train' if i<160 else 'dev' if i<210 else 'test'; sid=f'{split}_{i:03d}'
 out=ROOT/'data'/sid;out.mkdir(parents=True,exist_ok=False)
 for src,name in [(f0,'f0.png'),(fin,'final.png')]:
  im=cv2.imread(str(src));
  if im is None: raise ValueError(str(src))
  if im.shape[:2]!=(512,512): im=cv2.resize(im,(512,512),interpolation=cv2.INTER_AREA)
  if not cv2.imwrite(str(out/name),im): raise IOError(name)
 row=dict(id=sid,split=split,source_dir=d,video=str(vid),video_sha256=vhash,raw_f0_sha256=a,raw_final_sha256=b,f0=str(out/'f0.png'),final=str(out/'final.png'),f0_sha256=sha(out/'f0.png'),final_sha256=sha(out/'final.png'),frame_count=n,description=prompt,eval_text='<image>\n'+text)
 seen.update((vhash,a,b)); rows.append(row)
for split in ('train','dev','test'):
 with open(ROOT/f'{split}_sources.jsonl','w') as f:
  for r in rows:
   if r['split']==split: f.write(json.dumps(r,ensure_ascii=False)+'\n')
summary={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':20261029,'candidate_count':len(paths),'counts':{s:sum(r['split']==s for r in rows) for s in ('train','dev','test')},'reject_count':len(rejected),'rejections':rejected,'unique_raw_video_and_endpoints':True,'prompt_wrapper_sha256':sha(WRAP),'opencv':cv2.__version__,'note':'Raw video byte hashes and raw endpoint file hashes are unique across all selected rows; decoded frame content and parameter semantic duplicates are not checked. Frame count is container metadata and first-frame decoding, not a full frame decode. Historical exposure unknown.'}
(ROOT/'data_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,ensure_ascii=False))
