"""Independent disk-NAV metrics and raw-return/order cash reconstruction.

This does not regenerate macro signals or author orders. No parent imports.
"""
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def main():
    tag=sys.argv[1] if len(sys.argv)>1 else 'v2'
    out=HERE/tag;tab=pd.read_csv(out/'comparison.csv');checks=[]
    maxerr=0.;n=0;account_days=0
    for d in (out/'accounts').iterdir():
        l=pd.read_parquet(d/'ledger.parquet');m=json.loads((d/'meta.json').read_text());account_days+=len(l)
        capital=m['capital']
        for row in tab[tab.runid.eq(d.name)].itertuples():
            curve=l.nav.loc[row.start:row.end]
            before=l.nav.loc[l.index<row.start]
            base=before.iloc[-1] if len(before) else capital
            firstday=before.index[-1] if len(before) else m['base_date']
            nav=pd.concat([pd.Series([base],index=[firstday]),curve])
            daily=nav.pct_change().iloc[1:]
            vals={'cagr':np.expm1(np.log(nav.iloc[-1]/base)*365.25/(pd.Timestamp(nav.index[-1])-pd.Timestamp(nav.index[0])).days),
                  'total':nav.iloc[-1]/base-1,'vol':daily.std(ddof=1)*np.sqrt(252),
                  'mdd':-(nav/nav.cummax()-1).min()}
            for key,v in vals.items():maxerr=max(maxerr,abs(v-getattr(row,key)));n+=1
        if not d.name.startswith('500000_2014-01-02_') or m['case'] not in ['V12','previous_documented','expanded_policy_only','expanded_policy_only_strictFCF']:continue
        bm=m['case']=='V12'
        path=ROOT/'guosheng_six_regime_long_20261112/data'/('returns_v12.parquet' if bm else 'returns_gs.parquet')
        ret=pd.read_parquet(path)
        tr=pd.read_parquet(d/'trades.parquet');w=pd.read_parquet(d/'weights.parquet')
        pos=pd.Series(0.,index=ret.columns);cash=float(capital);naverr=feeerr=weighterr=0.
        grouped={date:g for date,g in tr.groupby('date')}
        bp=float(d.name.split('_')[2].replace('bp',''))
        for date in l.index:
            cash_before=cash;pos=pos*(1+ret.loc[date]);cost=0.;buyneed=0.
            for t in grouped.get(date,pd.DataFrame()).itertuples():
                fee=max(5.,t.notional*.00015)+t.notional*bp/10000
                feeerr=max(feeerr,abs(fee-t.cost));cost+=fee
                if t.side=='sell':
                    assert t.notional<=pos[t.asset]+1e-6
                    pos[t.asset]-=t.notional;cash+=t.notional-fee
                else:
                    pos[t.asset]+=t.notional;cash-=t.notional+fee;buyneed+=t.notional+fee
            assert buyneed<=cash_before+1e-6,('Same-day sales funding buys',d.name,date)
            nav=pos.sum()+cash;naverr=max(naverr,abs(nav-l.at[date,'nav']))
            assert cash>=-1e-5
            weighterr=max(weighterr,float((pos/nav-w.loc[date,pos.index]).abs().max()))
        assert naverr<1e-4 and feeerr<1e-5 and weighterr<1e-9
        checks.append(dict(runid=d.name,days=len(l),trades=len(tr),nav_error=naverr,fee_error=feeerr,weight_error=weighterr))
    assert maxerr<1e-10
    summary=dict(accounts=tab.runid.nunique(),account_days=account_days,metric_checks=n,max_metric_difference=maxerr,
                 raw_return_order_cash_checks=checks,not_done='Author source data and order-generation reconstruction; actual opening fills')
    (out/'independent_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame(checks).to_csv(out/'independent_cash_checks.csv',index=False)
    events=pd.read_csv(out/'events_used.csv')
    monthly=json.loads((HERE/'monthly_receipts.json').read_text(encoding='utf-8'))
    catalog=[]
    for r in monthly:
        raw=r.get('body','');body=raw.split('关闭窗口',1)[-1].split('打印本页')[0].strip()
        pub=re.search(r'20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',raw)
        title=(HERE/'sources'/r['file']).read_text(encoding='utf-8',errors='replace')
        title=re.search(r'<title>(.*?)</title>',title,re.S)
        catalog.append(dict(category='slf' if '/125443/' in r['url'] else 'mlf',title=title[1] if title else '',publication=pub[0] if pub else '',contains_numeric_rate=bool(re.search(r'\d(?:\.\d+)?\s*[%％]',body)),url=r['url'],sha256=r['sha256'],body=body))
    cat=pd.DataFrame(catalog);cat.to_csv(HERE/'monthly_notice_catalog.csv',index=False,encoding='utf-8-sig')
    counts=events.groupby('category').agg(raw_rows=('vote','size'),first_model_date=('usable_date','min'),last_model_date=('usable_date','max'))
    counts['distinct_category_dates']=events.drop_duplicates(['category','effective_date']).groupby('category').size()
    counts.to_csv(out/'event_coverage.csv',encoding='utf-8-sig')
    print(json.dumps(summary,indent=2),flush=True)
    print(counts.to_string(),flush=True)
    print('MONTHLY_NOTICE_COVERAGE',cat.groupby('category').agg(count=('title','size'),rate_disclosed=('contains_numeric_rate','sum')).to_string(),flush=True)
    selected=tab[(tab.capital==500000)&(tab.account_start=='2014-01-02')&tab.window.isin(['paper','full'])]
    print(selected[['case','bp','window','cagr','vol','mdd','mean_cash']].to_string(index=False),flush=True)

if __name__=='__main__':main()
