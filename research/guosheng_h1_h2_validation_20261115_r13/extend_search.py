"""Bounded search of NBS's public website search, then read matching original releases."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.request import Request,urlopen
from urllib.parse import urlencode
import json,re,hashlib,time
import fetch_official as f
O=f.O;Q=O/'search';Q.mkdir(exist_ok=True)
def query(kind,page):
    term='工业生产者出厂价格' if kind=='ppi' else '中国采购经理指数'
    p=Q/f'{kind}_{page:03d}.json'
    try:
        if p.exists():data=json.loads(p.read_text(encoding='utf8'))
        else:
            params={'siteCode':'bm36000002','qt':term,'page':page,'pageSize':20,'tab':'','sort':'date'}
            req=Request('https://api.so-gov.cn/query/s',data=urlencode(params).encode(),headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/x-www-form-urlencoded','Origin':'https://www.stats.gov.cn','Referer':'https://www.stats.gov.cn/search/s'})
            data=json.loads(urlopen(req,timeout=15).read().decode('utf8'));p.write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')
        return [dict(title=x['data'].get('titleO',''),url=x['data'].get('url','')) for x in data.get('resultDocs',[])]
    except Exception as e:print('SEARCH_ERROR',kind,page,str(e),flush=True);return []
items={}
with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(query,kind,page) for kind in ['ppi','pmi'] for page in range(1,71)]
    for n,p in enumerate(as_completed(futures),1):
        for x in p.result():
            title=x['title']
            if re.match(r'20\d\d年\d{1,2}月',title) and ('工业生产者出厂价格' in title or ('采购经理指数' in title and ('运行情况' in title or '制造业采购经理' in title))):items[x['url']]=x
        if n%20==0:print('SEARCH_DONE',n,'MATCHING',len(items),flush=True)
existing=json.loads((O/'parsed_releases.json').read_text(encoding='utf8'));seen={r['source'] for r in existing}
additional=[v for k,v in items.items() if k not in seen]
with ThreadPoolExecutor(max_workers=5) as pool:
    for n,p in enumerate(as_completed([pool.submit(f.parse,v) for v in additional]),1):
        r=p.result()
        if r:existing.append(r)
        if n%40==0:print('EXTRA_DOCS',n,flush=True)
(O/'all_releases.json').write_text(json.dumps(existing,ensure_ascii=False,indent=2),encoding='utf8')
for kind in ['ppi','pmi']:
    z=[r for r in existing if r['kind']==kind and r['status']=='parsed'];months={r['month'] for r in z}
    print('COVERAGE',kind,len(z),len(months),min(months,default='none'),max(months,default='none'),flush=True)
