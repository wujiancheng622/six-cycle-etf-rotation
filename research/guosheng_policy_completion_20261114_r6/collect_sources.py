"""Archive public rate sources only; never modify the parent research inputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import urljoin
import csv, hashlib, json, re, time
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'sources'
OUT.mkdir(parents=True, exist_ok=True)

class Parser(HTMLParser):
    def __init__(self):
        super().__init__(); self.text=[]; self.links=[]; self.href=None; self.label=[]
    def handle_starttag(self, tag, attrs):
        if tag=='a': self.href=dict(attrs).get('href'); self.label=[]
    def handle_data(self, s):
        self.text.append(s)
        if self.href is not None:self.label.append(s)
    def handle_endtag(self, tag):
        if tag=='a' and self.href is not None:
            self.links.append((''.join(self.label).strip(),self.href));self.href=None

SEEDS = {
 'pbc_rates':'https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125838/125885/125896/index.html',
 'pbc_deposit_history':'https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125838/125888/2968982/index.html',
 'pbc_loan_history':'https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125838/125888/index.html',
 'pbc_slf':'https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125437/125443/index.html',
 'pbc_mlf':'https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125437/125446/index.html',
 'pbc_rrr':'https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125434/125798/index.html',
 'icbc_index':'https://www.icbc.com.cn/column/1438058326469787819.html',
 'boc_lpr':'https://www.boc.cn/fimarkets/lilv/fd32/201310/t20131031_2591219.html',
 'tushare_lpr_docs':'https://tushare.pro/document/2?doc_id=151',
}

def fetch(item):
    name,url=item; row={'name':name,'url':url,'fetched_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    try:
        r=requests.get(url,timeout=20,headers={'User-Agent':'Mozilla/5.0'})
        row.update(status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
        if r.status_code!=200: return row
        suffix='.pdf' if r.content.startswith(b'%PDF-') else '.html'
        p=OUT/(name+suffix);p.write_bytes(r.content);row['file']=p.name
        if suffix=='.html':
            enc='utf-8' if b'utf-8' in r.content[:5000].lower() else r.apparent_encoding
            text=r.content.decode(enc or 'utf-8',errors='replace');parser=Parser();parser.feed(text)
            (OUT/(name+'.txt')).write_text('\n'.join(x.strip() for x in parser.text if x.strip()),encoding='utf-8')
            row['links']=[(t,urljoin(r.url,h)) for t,h in parser.links if t]
    except Exception as e:row['error']=type(e).__name__+': '+str(e)
    return row

if __name__=='__main__':
    rows=list(ThreadPoolExecutor(4).map(fetch,SEEDS.items()))
    more={}
    for r in rows:
        for title,url in r.get('links',[]):
            if (('常备借贷便利利率表' in title) or ('存贷款基准利率' in title) or
                ('超额' in title) or ('贷款基准利率（2015年10月24日' in title) or
                title in ['常备借贷便利工作信息','常备借贷便利操作表','中期借贷便利工作信息','中期借贷便利操作表']):
                more['linked_'+hashlib.sha256(url.encode()).hexdigest()[:12]]=url
    rows+=list(ThreadPoolExecutor(4).map(fetch,more.items()))
    (HERE/'source_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows:print(r['name'],r.get('status'),r.get('bytes'),r.get('error',''),r['url'],flush=True)
