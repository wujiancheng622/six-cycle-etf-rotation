from pathlib import Path
import json,gzip,base64,hashlib
H=Path(__file__).resolve().parent
d=json.loads((H/'R13_compact.json').read_text(encoding='utf8'))
keys=['freeze','verification','coverage','primary','diagnostics','baselines','support','event_summary','weak_order_months','source_comparison','scope']
z={k:d[k] for k in keys}
z['segments']=[];seen=set()
for x in d['segments']:
    if x['evidence']=='official' and x['segment'] in ['2014-2019','2020-2022','2023-2025','2026']:
        key=(x['hypothesis'],x['segment'])
        if key not in seen:z['segments'].append(x);seen.add(key)
z['source_directory']=str(H)
raw=json.dumps(z,ensure_ascii=False,separators=(',',':')).encode('utf8')
(H/'delivery_summary.json').write_bytes(raw)
print('SHA256',hashlib.sha256(raw).hexdigest(),'BYTES',len(raw),flush=True)
print(base64.b64encode(gzip.compress(raw,mtime=0)).decode(),flush=True)
