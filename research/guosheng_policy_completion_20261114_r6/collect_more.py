"""Follow ordinary public website pagination and public ICBC rate endpoints."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
import hashlib,json,re,sys
import requests
from collect_sources import fetch,OUT,HERE,Parser

def main():
    html=(OUT/'icbc_index.html').read_text(encoding='utf-8')
    scripts=re.findall(r'<script[^>]+src=[\"\x27]([^\"\x27]+)',html)
    print('SCRIPTS',scripts,flush=True)
    i=html.find('ajaxPrefilter');print('PUBLIC_HEADERS',html[i:i+1600],flush=True)
    candidates=[urljoin('https://www.icbc.com.cn/',s) for s in scripts if any(x in s.lower() for x in ['config','common','init'])]
    rows=[]
    for u in candidates:
        r=fetch(('icbc_js_'+hashlib.sha256(u.encode()).hexdigest()[:12],u));rows.append(r)
        if 'file' in r:
            text=(OUT/r['file']).read_text(errors='replace')
            for match in re.finditer('papi',text):print('CONFIG',u,text[max(0,match.start()-300):match.start()+600],flush=True)
    seeds={}
    for kind,part,num,last in [('slf','125443/125857','17096',7),('mlf','125446/125873','17099',8)]:
        for n in range(2,last+1):
            seeds[f'{kind}_page_{n}']=f'https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125437/{part}/{num}-{n}.html'
    pages=list(ThreadPoolExecutor(4).map(fetch,seeds.items()));rows+=pages
    candidates={}
    prior=json.loads((HERE/'source_receipts.json').read_text(encoding='utf-8'))
    for r in pages+prior:
        if r['name'] not in ['linked_ce8b14f8d8cd','linked_f43a35543fe2'] and not r['name'].startswith(('slf_page','mlf_page')):continue
        for title,url in r.get('links',[]):
            if ('常备借贷便利开展情况' in title or '中期借贷便利开展情况' in title) and re.search('20(?:1[3-9]|2[0-6])',title):
                # Preserve all monthly publications: unchanged months test completeness too.
                candidates['monthly_'+hashlib.sha256(url.encode()).hexdigest()[:12]]=url
    (HERE/'monthly_candidates.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2),encoding='utf-8')
    print('MONTHLY_CANDIDATES',len(candidates),flush=True)
    monthly=list(ThreadPoolExecutor(6).map(fetch,candidates.items()));rows+=monthly
    (HERE/'more_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINISHED',len(rows),'OK',sum(r.get('status')==200 for r in rows),flush=True)
    for r in monthly:
        if r.get('status')!=200:print('FAIL',r['name'],r.get('error'),r.get('status'),flush=True)

if __name__=='__main__':main()
