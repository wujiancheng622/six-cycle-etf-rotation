"""Correct a demonstrable extraction error before fitting, retaining raw originals."""
from pathlib import Path
import json,re,html
H=Path(__file__).resolve().parent;O=H/'input/official';p=O/'all_releases.json'
raw=p.read_text(encoding='utf8');rows=json.loads(raw);fixes=[]
for row in rows:
    if row['kind']!='pmi' or not row.get('file'):continue
    text=html.unescape(re.sub('<[^>]+>',' ',(O/row['file']).read_text(encoding='utf8')));text=re.sub(r'\s+','',text)
    m=re.search(r'生产指数和新订单指数(?:分别)?为([\d.]+)%和([\d.]+)%',text)
    if m and row.get('raw_row','')=='新订单指数为'+m[1]+'%':
        fixes.append(dict(source=row['source'],month=row['month'],old=row['value'],new=float(m[2]),evidence=m[0],reason='combined sentence lists production first and orders second; substring regex incorrectly took production'))
        row.update(value=float(m[2]),raw_row=m[0],extraction_correction=True)
(O/'all_releases_before_joint_sentence_fix.json').write_text(raw,encoding='utf8')
p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
(H/'extraction_fix_log.json').write_text(json.dumps(dict(before_any_hypothesis_fit=True,fixes=fixes),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(fixes,ensure_ascii=False))
