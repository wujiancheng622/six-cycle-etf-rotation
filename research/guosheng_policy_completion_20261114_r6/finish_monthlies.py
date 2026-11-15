"""Finish archived public monthly notices using Windows TLS without disabling verification."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,subprocess,re,time
HERE=Path(__file__).resolve().parent;OUT=HERE/'sources'
class Text(HTMLParser):
    def __init__(self):super().__init__();self.text=[]
    def handle_data(self,d):self.text.append(d)
def one(item):
    name,url=item;p=OUT/(name+'.html');record=dict(name=name,url=url)
    try:
        if not p.exists():
            r=subprocess.run(['curl.exe','--silent','--show-error','--fail','--max-time','12',url],capture_output=True,check=True)
            p.write_bytes(r.stdout)
        raw=p.read_bytes();text=raw.decode('utf-8',errors='replace');parser=Text();parser.feed(text)
        s='\n'.join(x.strip() for x in parser.text if x.strip());(OUT/(name+'.txt')).write_text(s,encoding='utf-8')
        record.update(status=200,sha256=hashlib.sha256(raw).hexdigest(),file=p.name)
        # Retain compact content excerpt for analyst review, not as parsed events.
        where=s.find('文章来源');end=s.find('打印本页',where+30)
        nextend=s.find('打印本页',end+5)
        body=s[max(0,where-100):nextend if nextend>0 else where+2000]
        record['body']=body
    except Exception as e:record['error']=repr(e)
    return record
if __name__=='__main__':
    candidates=json.loads((HERE/'monthly_candidates.json').read_text(encoding='utf-8'))
    rows=list(ThreadPoolExecutor(6).map(one,candidates.items()))
    (HERE/'monthly_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('MONTHLY',len(rows),'OK',sum(r.get('status')==200 for r in rows),flush=True)
    # Print rate statements for years with gaps, preserving publication metadata.
    for r in rows:
        body=r.get('body','')
        if ('利率' in body) and any(y in body[:200] for y in ['2014','2015','2016','2017','2018','2019','2020','2025','2026']):
            print(r['url'],body.replace('\n',' ')[:1800],flush=True)
