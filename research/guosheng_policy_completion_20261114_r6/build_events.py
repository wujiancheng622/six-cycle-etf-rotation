"""Build inspectable source upgrades and event additions; preserve uncertainty labels."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,parse_qs
import csv,hashlib,json,re,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;S=HERE/'sources'
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'guosheng_parameter_robustness_20261113')]
import study_inputs as si

class Tables(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.row=[];self.cell=None
    def handle_starttag(self,t,a):
        if t=='tr':self.row=[]
        if t in ['td','th']:self.cell=[]
    def handle_data(self,d):
        if self.cell is not None:self.cell.append(d)
    def handle_endtag(self,t):
        if t in ['td','th'] and self.cell is not None:self.row.append(''.join(self.cell).strip());self.cell=None
        if t=='tr' and self.row:self.rows.append(self.row)

records=[];levels=[]
def event(cat,d,v,source,evidence,status,known=None,announced='',before='',after='',tenor='category',evidence_date=''):
    records.append(dict(category=cat,effective_date=d,known_date=known or d,usable_date=max(d,known or d),vote=int(v),source=source,evidence=evidence,dataset='r6_source_upgrade',evidence_status=status,announced_at=announced,rate_before=before,rate_after=after,tenor=tenor,evidence_publication_date=evidence_date))

def main():
    # Official historical tables validate levels, not necessarily original publication time.
    for cat,filename,url in [
      ('loan','linked_c04fe251d1bf.html','https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125838/125888/2968985/index.html'),
      ('deposit','pbc_deposit_history.html','https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125838/125888/2968982/index.html')]:
        parser=Tables();parser.feed((S/filename).read_text(encoding='utf-8'))
        prior=None;seen={}
        for row in parser.rows:
            dates=[re.search(r'(\d{4})\.(\d{2})\.(\d{2})',c) for c in row]
            if not dates or dates[0] is None:continue
            d='-'.join(dates[0].groups())
            nums=[]
            for c in row[1:]:
                m=re.search(r'^\s*(\d+(?:\.\d+)?)\s*$',c)
                if m:nums.append(float(m.group(1)))
            if not nums:continue
            # Compare equal-length fixed maturities only. 2014 loan bucket consolidation is
            # explicitly a cut in the official announcement, not a same-bucket numeric diff.
            vals=nums[1:] if cat=='deposit' else nums
            if d in seen:
                assert seen[d]==vals,(cat,d,'conflicting duplicate table row')
                continue
            seen[d]=vals
            levels.append(dict(category=cat,date=d,source=url,values=json.dumps(vals)))
            if d>='2010-01-01':
                if cat=='loan' and d=='2014-11-22':vote=1
                else:
                    assert prior is not None
                    changes=np.asarray(vals[:min(len(vals),len(prior))])-np.asarray(prior[:min(len(vals),len(prior))])
                    nz=changes[np.abs(changes)>1e-9]
                    assert len(nz) and (np.all(nz>0) or np.all(nz<0)),(cat,d,changes)
                    vote=-np.sign(nz[0])
                event(cat,d,vote,url,'PBC official historical benchmark rate table','official_historical_levels_date_not_first_release',before=json.dumps(prior),after=json.dumps(vals),evidence_date='2015-10-24')
            prior=vals
    # Exact public ICBC website API, six fixed-deposit maturities, not deposit ceiling.
    receipt=json.loads((HERE/'icbc_receipts.json').read_text(encoding='utf-8'))
    fields=['zczqThreeMonths','zczqHalfYear','zczqOneYear','zczqTwoYears','zczqThreeYears','zczqFiveYears']
    bank=[]
    for r in receipt[1:]:
        if r.get('response',{}).get('code')!=0:continue
        d=parse_qs(urlparse(r['url']).query)['date'][0]
        vals=[float(r['response']['data'][f]) for f in fields]
        bank.append((d,vals,r['url']))
        levels.append(dict(category='icbc_deposit',date=d,source=r['url'],values=json.dumps(vals)))
    bank.sort()
    for i,(d,vals,u) in enumerate(bank):
        if i==0:continue # no invented pre-sample change; PBC separately provides 2010 first event
        old=bank[i-1][1];changes=np.asarray(vals)-np.asarray(old);nz=changes[np.abs(changes)>1e-9]
        if not len(nz):continue
        assert np.all(nz>0) or np.all(nz<0),(d,changes)
        event('deposit',d,-np.sign(nz[0]),u,'ICBC official posted fixed-deposit rates; all six maturities checked','official_historical_levels_date_not_first_release',before=json.dumps(old),after=json.dumps(vals),tenor='3m/6m/1y/2y/3y/5y')
    # Original cache is a data-vendor historical snapshot. Do not label it a primary vintage.
    lpr=si.PROVIDER.dated('shibor_lpr',None,date_column='date')['1y'].astype(float).loc[:'20190819'].round(3)
    change=lpr.diff();subset=change.loc[change.notna()&change.ne(0)]
    for d,delta in subset.items():
        day=pd.to_datetime(d).date().isoformat()
        event('loan',day,-np.sign(delta),'Tushare frozen shibor_lpr cache; https://tushare.pro/document/2?doc_id=151','Pre-August-2019 LPR changes formerly excluded','vendor_snapshot_historical_not_primary',before=float(lpr.shift().loc[d]),after=float(lpr.loc[d]),tenor='old LPR 1y')
    event('reserve_interest','2020-04-07',1,'https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125434/125798/b97818c9c9644078a3daba8aee0aec92/index.html','Excess reserve interest 0.72 to 0.35, original announcement','original_announcement',known='2020-04-03',announced='2020-04-03T16:57:32+08:00',before=.72,after=.35,tenor='excess reserves',evidence_date='2020-04-03')
    receipts=json.loads((HERE/'source_receipts.json').read_text(encoding='utf-8'))
    for r in receipts:
        if '/125896/' not in r['url'] or '/index.html' not in r['url'] or not r['name'].startswith('linked_'):continue
        text=(S/(r['name']+'.txt')).read_text(encoding='utf-8')
        compact=re.sub(r'\s+','',text)
        d=re.search(r'自(20\d{2})年(\d+)月(\d+)日起执行',compact)
        if not d:d=re.search(r'常备借贷便利利率表[（(](20\d{2})年(\d+)月(\d+)日',compact)
        if not d:continue
        date=f'{d[1]}-{int(d[2]):02d}-{int(d[3]):02d}'
        pub=re.search(r'(20\d{2}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})',text)
        dt=pub[1] if pub else date
        known=(pd.Timestamp(dt)+pd.Timedelta(days=1)).date().isoformat() if pub and pub[2]>'15:00:00' else dt
        event('slf',date,1,r['url'],'PBC original SLF rate table; post-15:00 announcements enter the next close-information date','original_rate_table',known=known,announced=(dt+'T'+pub[2]+'+08:00') if pub else '',evidence_date=dt)
    extra=HERE/'manual_verified_events.csv'
    if extra.exists():records.extend(pd.read_csv(extra,keep_default_na=False).to_dict('records'))
    result=pd.DataFrame(records).sort_values(['category','effective_date','evidence'])
    # Duplicate same-date category evidence is preserved for provenance; scoring dedups it.
    result.to_csv(HERE/'additional_verified_events.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(levels).to_csv(HERE/'verified_rate_levels.csv',index=False,encoding='utf-8-sig')
    print('NEW_ROWS',len(result),'ICBC_SNAPSHOTS',len(bank),'OLD_LPR_CHANGES',len(subset),flush=True)
    print(result.groupby(['category','evidence_status']).size().to_string(),flush=True)

if __name__=='__main__':main()
