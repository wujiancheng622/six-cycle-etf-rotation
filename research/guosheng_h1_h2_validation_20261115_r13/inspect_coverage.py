from pathlib import Path
import json,re
import pandas as pd
H=Path(__file__).resolve().parent;O=H/'input/official'
data=json.loads((O/'all_releases.json').read_text(encoding='utf8'))
for kind in ['ppi','pmi']:
    valid={r['month'] for r in data if r['kind']==kind and r['status']=='parsed'}
    missing=[str(x) for x in pd.period_range('2011-01','2026-10',freq='M') if str(x) not in valid]
    bad=[r for r in data if r['kind']==kind and r['status']!='parsed']
    print('MISSING',kind,missing);print('BAD_SAMPLE',json.dumps(bad[:3],ensure_ascii=False))
    for r in bad[:1]:
        if r.get('file'):
            t=(O/r['file']).read_text(encoding='utf8')
            print('DATES',re.findall(r'.{0,40}20\d\d[/-]\d\d[/-]\d\d.{0,50}',t)[:5]);print('HEAD',t[:800])
