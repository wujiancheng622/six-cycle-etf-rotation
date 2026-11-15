from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,urllib.parse,re,json
P=Path(__file__).resolve().parent; D=P/'nbs';D.mkdir(exist_ok=True)
base='https://www.stats.gov.cn/sj/zxfb/'
def get(url):
    name=url.split('/')[-1] or 'index.html';path=D/name
    if path.exists():return path.read_text(encoding='utf-8')
    try:
        s=urllib.request.urlopen(url,timeout=20).read().decode('utf-8');path.write_text(s,encoding='utf-8');return s
    except Exception:return ''
urls=[base+('index.html' if i==0 else f'index_{i}.html') for i in range(67)]
links={}
with ThreadPoolExecutor(max_workers=6) as ex:
    for s in ex.map(get,urls):
        for href,title in re.findall(r'href="([^"]+)"[^>]*title=[\'\"]([^\'\"]+)',s):
            if re.match(r'20\d\d年\d+月份工业生产者',title): links[urllib.parse.urljoin(base,href)]=title
print('PPI announcements',len(links),flush=True)
rows=[]
with ThreadPoolExecutor(max_workers=6) as ex:
    for (url,title),s in zip(links.items(),ex.map(get,links)):
        year,month=map(int,re.match(r'(20\d\d)年(\d+)月',title).groups())
        day=re.search(r'/t(\d{8})_',url)
        if day:rows.append(dict(month=f'{year:04d}-{month:02d}',released=day[1],url=url,title=title,downloaded=bool(s)))
(P/'ppi_release_catalog.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('coverage',min([x['month'] for x in rows],default=''),max([x['month'] for x in rows],default=''),flush=True)
