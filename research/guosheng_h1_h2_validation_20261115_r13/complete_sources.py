"""Resolve documented older HTML layouts and alternate official release titles."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.request import Request,urlopen
from urllib.parse import urlencode
import json,re,hashlib,html,ast
H=Path(__file__).resolve().parent;O=H/'input/official';Q=O/'search'
# Load only pure parser helpers, not their original bulk-fetch entry point.
tree=ast.parse((H/'fetch_official.py').read_text(encoding='utf8'))
namespace=dict(Path=Path,json=json,re=re,hashlib=hashlib,html=html,O=O,urlopen=urlopen,Request=Request)
import time
namespace['time']=time
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['clean','fetch','parse']],type_ignores=[]),'cached_official_parsers','exec'),namespace)
parse=namespace['parse'];fetch=namespace['fetch']
def query(term,page):
    p=Q/f'extra_{hashlib.sha256(term.encode()).hexdigest()[:8]}_{page:03d}.json'
    try:
        if p.exists():o=json.loads(p.read_text(encoding='utf8'))
        else:
            params={'siteCode':'bm36000002','qt':term,'page':page,'pageSize':20,'tab':'','sort':'date'}
            req=Request('https://api.so-gov.cn/query/s',data=urlencode(params).encode(),headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/x-www-form-urlencoded','Referer':'https://www.stats.gov.cn/search/s'})
            o=json.loads(urlopen(req,timeout=15).read().decode('utf8'));p.write_text(json.dumps(o,ensure_ascii=False),encoding='utf8')
        return o.get('resultDocs',[])
    except Exception as e:print('SEARCH_ERROR',term,page,str(e),flush=True);return []
docs=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for fut in as_completed([pool.submit(query,term,page) for term in ['工业生产者价格变动情况','制造业采购经理指数'] for page in range(1,36)]):docs.extend(fut.result())
items={}
for p in Q.glob('*.json'):
    try:docs.extend(json.loads(p.read_text(encoding='utf8')).get('resultDocs',[]))
    except Exception:pass
publication_dates={}
for doc in docs:
    d=doc.get('data',{});title=d.get('titleO','');url=d.get('url','');date=d.get('docDate','');publication_dates[url]=date
    if not re.match(r'(20\d\d年)?\d{1,2}月',title):continue
    if not (('工业生产者' in title and '价格' in title) or ('制造业采购经理指数' in title) or ('采购经理指数运行情况' in title)):continue
    if not re.match(r'20\d\d年',title):
        if not date:continue
        month=int(re.match(r'(\d{1,2})月',title)[1]);year=int(date[:4])-(month>int(date[5:7]));title=f'{year}年'+title
    items[url]=dict(title=title,url=url)
items['https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1898696.html']=dict(title='2015年1月份工业生产者价格变动情况',url='https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1898696.html')
old=json.loads((O/'all_releases.json').read_text(encoding='utf8'));known={r['source'] for r in old}
new=[x for u,x in items.items() if u not in known]
print('ADDITIONAL_RELEASES',len(new),flush=True)
with ThreadPoolExecutor(max_workers=5) as pool:
    for fut in as_completed([pool.submit(parse,x) for x in new]):
        r=fut.result()
        if r:old.append(r)
for r in old:
    if r.get('value') is None or r.get('published_date') or not r.get('file'):continue
    t=(O/r['file']).read_text(encoding='utf8');flat=html.unescape(re.sub('<[^>]+>',' ',t))
    m=re.search(r'成文日期\s*(20\d\d)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日',flat)
    if m:
        date=f'{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}'
        if publication_dates.get(r['source'])==date:
            r.update(published_date=date,status='parsed',date_basis='official_document_date_agrees_with_official_search_docDate')
(O/'all_releases_before_layout_completion.json').write_text((O/'all_releases.json').read_text(encoding='utf8'),encoding='utf8')
(O/'all_releases.json').write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf8')
for kind in ['ppi','pmi']:
    z={r['month'] for r in old if r['kind']==kind and r['status']=='parsed'}
    print('FINAL_COVERAGE',kind,len(z),min(z,default='none'),max(z,default='none'),flush=True)
