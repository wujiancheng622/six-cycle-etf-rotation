"""Archive official release pages without inspecting strategy outcomes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from urllib.request import urlopen,Request
from urllib.parse import urljoin
import json,re,hashlib,html,time
H=Path(__file__).resolve().parent;O=H/'input/official';O.mkdir(parents=True,exist_ok=True)
BASE='https://www.stats.gov.cn/sj/zxfb/'
def clean(x):return re.sub(r'\s+','',html.unescape(re.sub('<[^>]+>',' ',x)))
def fetch(url):
    name=hashlib.sha256(url.encode()).hexdigest()[:24]+'.html';p=O/name
    if p.exists():raw=p.read_bytes()
    else:
        for attempt in range(2):
            try:
                raw=urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=15).read();p.write_bytes(raw);break
            except Exception:
                if attempt:raise
                time.sleep(.5)
    return raw.decode('utf8',errors='replace'),name,hashlib.sha256(raw).hexdigest()
def listing(i):
    url=BASE+('index.html' if i==0 else f'index_{i}.html')
    try:
        t,name,sha=fetch(url);found=[]
        for href,title in re.findall(r'<a[^>]*href=[\"\x27]([^\"\x27]+)[\"\x27][^>]*>(.*?)</a>',t,re.S):
            title=clean(title)
            if ('工业生产者' in title and ('出厂价格' in title or '价格' in title)) or ('采购经理指数' in title and ('运行情况' in title or '制造业采购经理' in title)):
                if re.search(r'20\d{2}年\d{1,2}月',title):found.append(dict(url=urljoin(url,href),title=title))
        return dict(index=i,status='ok',file=name,sha=sha,items=found)
    except Exception as e:return dict(index=i,status='error',error=str(e))
root=fetch(BASE)[0];m=re.search(r'createPageHTML\((\d+),',root);count=int(m.group(1)) if m else 1
lists=[]
with ThreadPoolExecutor(max_workers=5) as pool:
    for f in as_completed([pool.submit(listing,i) for i in range(count)]):
        x=f.result();lists.append(x)
        if len(lists)%10==0:print('LIST',len(lists),'/',count,flush=True)
items={x['url']:x for l in lists if l['status']=='ok' for x in l['items']}
(O/'index_manifest.json').write_text(json.dumps(dict(fetched_at=datetime.now(timezone.utc).isoformat(),lists=lists,urls=list(items.values())),ensure_ascii=False,indent=2),encoding='utf8')
def parse(item):
    url=item['url'];title=item['title'];kind='ppi' if '工业生产者' in title else 'pmi'
    mm=re.search(r'(20\d{2})年(\d{1,2})月',title);period=f'{mm[1]}-{int(mm[2]):02d}'
    if not '2010-01'<=period<='2026-10':return None
    try:
        t,file,sha=fetch(url)
        metas={a.lower():b for a,b in re.findall(r'<meta[^>]*name=[\"\x27]([^\"\x27]+)[\"\x27][^>]*content=[\"\x27]([^\"\x27]*)',t,re.I)}
        dates=[metas.get(k,'') for k in ['pubdate','firstpublishedtime','publishtime']]
        for pat in [r'发布时间[：:]?\s*([^<]{10,35})',r'日期[：:]?\s*([^<]{10,35})']:
            dates.extend(re.findall(pat,t))
        date=None
        for v in dates:
            z=re.search(r'(20\d\d)[-/](\d{1,2})[-/](\d{1,2})(?:\s+(\d\d:\d\d(?::\d\d)?))?',v)
            if z:date=f'{z[1]}-{int(z[2]):02d}-{int(z[3]):02d}';break
        if date is None:
            section=t[t.find('detail-title-des'):t.find('detail-title-des')+1800]
            visible=html.unescape(re.sub('<[^>]+>',' ',section))
            z=re.search(r'(20\d\d)[-/](\d{1,2})[-/](\d{1,2})',visible)
            if z:date=f'{z[1]}-{int(z[2]):02d}-{int(z[3]):02d}'
        rows=[]
        for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>',t,re.S|re.I):
            cells=[clean(c) for c in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',tr,re.S|re.I)]
            if cells:rows.append(cells)
        val=None;raw_row=None
        if kind=='ppi':
            for r in rows:
                if r and re.fullmatch(r'[\u3000\s]*加工(?:工业)?',r[0]):
                    nums=[float(x.replace('%','').replace('－','-').replace('−','-')) for x in r[1:] if re.fullmatch(r'[\-－−+]?\d+(?:\.\d+)?%?',x)]
                    if len(nums)>=2:val=nums[1];raw_row=r;break
        else:
            body=clean(t)
            m=re.search(r'新订单指数为([0-9.]+)%',body)
            if m:val=float(m[1]);raw_row=m[0]
            else:
                # Earlier release format: first manufacturing PMI table columns PMI/production/new orders.
                for r in rows:
                    if len(r)>=4 and r[0] in [str(int(mm[2]))+'月',str(mm[1])+'年'+str(int(mm[2]))+'月']:
                        try:
                            candidate=float(r[3]);pmi=float(r[1]);production=float(r[2])
                            if 20<candidate<80 and 20<pmi<80 and 20<production<80:val=candidate;raw_row=r;break
                        except ValueError:pass
        return dict(kind=kind,month=period,source=url,title=title,published_date=date,value=val,raw_row=raw_row,file=file,sha256=sha,status='parsed' if val is not None and date else 'needs_parse')
    except Exception as e:return dict(kind=kind,month=period,source=url,title=title,status='failed',error=str(e))
records=[]
with ThreadPoolExecutor(max_workers=5) as pool:
    for f in as_completed([pool.submit(parse,v) for v in items.values()]):
        x=f.result()
        if x:records.append(x)
        if len(records)%50==0:print('DOCS',len(records),flush=True)
(O/'parsed_releases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
for kind in ['ppi','pmi']:
    z=[r for r in records if r['kind']==kind];ok=[r for r in z if r['status']=='parsed']
    print(kind,'found',len(z),'parsed',len(ok),'first',min([r['month'] for r in z],default='none'),'last',max([r['month'] for r in z],default='none'),flush=True)
    print('SAMPLES',json.dumps(z[:2],ensure_ascii=False),flush=True)
