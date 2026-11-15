"""Build the preregistered monthly grid from frozen targets; no trading code is called."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parent;I=H/'input';O=H/'results';O.mkdir(exist_ok=True)
P=json.loads((H/'protocol.json').read_text(encoding='utf8'))
KEYS=['growth','gold','metals','fcf','bond','dividend']
def shift_month(m,n):return str(pd.Period(m,freq='M')+n)
def transform(values,lag=3):
    a=values.rolling(12,min_periods=12).mean();u=a/a.shift(12)-1;z=u-u.shift(lag)
    return z,np.where(z.notna(),np.where(z>0,1.,-1.),np.nan)
def main():
    receipt=json.loads((H/'freeze_receipt.json').read_text(encoding='utf8'))
    assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in receipt['hashes'].items())
    files=['month_detail.csv','asset_month_detail.csv']
    monthly=pd.read_csv(R/'guosheng_timing_attribution_20261115/results'/files[0])
    assets=pd.read_csv(R/'guosheng_timing_attribution_20261115/results'/files[1])
    raw_pmi=pd.read_csv(R/'guosheng_original_reconstruction_20261114/results/pmi_dates_used.csv').set_index('month')
    raw_pmi.index=raw_pmi.index.astype(str)
    raw_pmi['available']=pd.to_datetime(raw_pmi.available)
    for col in ['manufacturing','nonmanufacturing','private']:
        delta,vote=transform(raw_pmi[col]);raw_pmi[col+'_vote']=vote
        np.testing.assert_allclose(delta,raw_pmi[col+'_change'],atol=1e-10,rtol=0,equal_nan=True)
    # Only frozen asset histories determine labels and 21-session controls.
    gs_hist=pd.read_parquet(R/'guosheng_six_regime_long_20261112/data/returns_gs.parquet')
    cal=pd.DatetimeIndex(pd.to_datetime(gs_hist.index))
    def delay_date(date,n):
        d=pd.Timestamp(date)
        if not n or d<cal[0]:return d
        j=cal.searchsorted(d,side='right')+n-1
        return cal[j] if j<len(cal) else pd.Timestamp.max.normalize()
    vendor={}
    for kind in ['ppi','pmi']:
        f=pd.read_parquet(I/f'cn_{kind}.parquet').copy();f['month']=pd.to_datetime(f.month,format='%Y%m').dt.strftime('%Y-%m');f=f.set_index('month').sort_index();vendor[kind]=f
        assert f.index.is_unique and list(f.index)==list(pd.period_range(f.index[0],f.index[-1],freq='M').astype(str))
    archives=I/'official/all_releases.json'
    records=json.loads((archives if archives.exists() else I/'official/parsed_releases.json').read_text(encoding='utf8'))
    source_tables={};conflicts=[]
    for kind in ['ppi','pmi']:
        collection=[]
        for month in sorted({x['month'] for x in records if x['kind']==kind}):
            rr=[x for x in records if x['kind']==kind and x['month']==month and x.get('published_date')]
            valid=[x for x in rr if x.get('value') is not None and x.get('status')=='parsed']
            if not rr:continue
            # Exact values from an official monthly-release page; conflicting copies are not selected by results.
            distinct={x['value'] for x in valid}
            if len(distinct)>1:
                conflicts.append(dict(kind=kind,month=month,values=sorted(distinct),sources=[r['source'] for r in valid]));val=np.nan
            else:val=next(iter(distinct)) if distinct else np.nan
            best=sorted(valid or rr,key=lambda r:(r['published_date'],0 if '/sj/zxfb/' in r['source'] else 1,r['source']))[0]
            collection.append(dict(month=month,value=val,released=best['published_date'],source=best['source'],sha256=best.get('sha256',''),copies=len(rr),conflict=len(distinct)>1))
        f=pd.DataFrame(collection).set_index('month').sort_index();f['released']=pd.to_datetime(f.released)
        source_tables[kind]=f;f.to_csv(O/f'official_{kind}_monthly.csv',encoding='utf-8-sig')
    (O/'official_conflicts.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2),encoding='utf8')
    discrepancies=[]
    for kind,field in [('ppi','ppi_mp_p_yoy'),('pmi','pmi010500')]:
        a=source_tables[kind];b=vendor[kind]
        for m in a.index.intersection(b.index):
            discrepancies.append(dict(kind=kind,month=m,official=a.at[m,'value'],vendor=b.at[m,field],difference=b.at[m,field]-a.at[m,'value'],source=a.at[m,'source']))
    pd.DataFrame(discrepancies).to_csv(O/'official_vendor_comparison.csv',index=False,encoding='utf-8-sig')
    # State continuation age: observability only, not a new predictor or altered state rule.
    states=pd.read_csv(R/'guosheng_remaining_review_20261114/results/state_match_days.csv').set_index('date')
    since=0;last=''
    for date,row in states.iterrows():
        if row.direct_match:since=0;last=date
        else:since+=1
        states.at[date,'carried_sessions']=since;states.at[date,'last_direct_date']=last
    credit=pd.read_csv(R/'guosheng_macro_auxiliary_20261115/decision_date_credit.csv').set_index('date')
    grid=[]
    pm=source_tables['pmi'];pp=source_tables['ppi']
    for run in ['base','quote']:
        mm=monthly.loc[monthly.run.eq(run)].sort_values('date');aa=assets.loc[assets.run.eq(run)]
        for row in mm.itertuples():
            a=aa.loc[aa.date.eq(row.date)].set_index('asset');assert set(a.index)==set(KEYS)
            selected=list(a.index[a.selected]);excluded=[k for k in KEYS if k not in selected]
            assert row.known<row.date<row.end
            record=dict(run=run,date=row.date,end=row.end,known=row.known,year=int(row.date[:4]),phase=int(row.phase),
                money=float(a.money.iloc[0]),credit=float(a.credit.iloc[0]),growth=float(a.growth.iloc[0]),carried=1-int(row.direct),basket=row.basket,
                y1=a.at['metals','forward']-a.at['bond','forward'],y2=a.loc[selected,'forward'].mean()-a.loc[excluded,'forward'].mean(),
                price1=a.at['metals','prev_21']-a.at['bond','prev_21'],price2=a.loc[selected,'prev_21'].mean()-a.loc[excluded,'prev_21'].mean())
            assert abs(record['y2']-row.gap)<1e-12
            orig=raw_pmi.loc[raw_pmi.available.le(row.known)].iloc[-1];m=str(orig.name);record['original_pmi_month']=m
            record['original_pmi_available']=str(orig.available.date())
            for k in ['manufacturing','nonmanufacturing','private']:record[k+'_vote']=float(orig[k+'_vote'])
            assert np.sign(sum(record[k+'_vote'] for k in ['manufacturing','nonmanufacturing','private']))==record['growth']
            if row.known in states.index:record.update(carried_sessions=states.at[row.known,'carried_sessions'],last_direct_date=states.at[row.known,'last_direct_date'])
            if row.date in credit.index:
                for k in ['household_delta','enterprise_delta','credit_change','directions_disagree']:record[k]=credit.at[row.date,k]
            for lag_days in [0,1,3]:
                # Latest publication is chosen by release facts, never by which values are easy to verify.
                av=pp.loc[pp.released.map(lambda d:delay_date(d,lag_days)).le(row.known)]
                m1=str(av.index[-1]) if len(av) else None
                # A missing archive month may not be silently replaced by an older publication.
                expected=shift_month(row.known[:7],-1)
                if m1 and m1<expected:m1=None
                record[f'ppi_month_lag{lag_days}']=m1
                for delta_lag in [2,3,4]:
                    for scope in ['official','vendor']:
                        tag=f'x1_{scope}_d{delta_lag}_lag{lag_days}';value=np.nan
                        if m1:
                            prev=shift_month(m1,-delta_lag)
                            series=pp.value if scope=='official' else vendor['ppi'].ppi_mp_p_yoy
                            if m1 in series.index and prev in series.index:
                                if scope=='vendor' or (prev in pp.index and delay_date(pp.at[prev,'released'],lag_days)<=pd.Timestamp(row.known)):
                                    value=series[m1]-series[prev]
                        record[tag]=value
                for timing in ['sync','latest']:
                    m2=m
                    if timing=='latest':
                        knownpm=pm.loc[pm.released.map(lambda d:delay_date(d,lag_days)).le(row.known)]
                        candidates=list(knownpm.index)
                        # The legacy composite proves at least that reference month's official PMI
                        # was known. Missing original-order archive is a quality gap, not a reason
                        # to pretend an older archived month was the latest information.
                        orig_order_date=pm.at[m,'released'] if m in pm.index else orig.available
                        if delay_date(orig_order_date,lag_days)<=pd.Timestamp(row.known):candidates.append(m)
                        m2=max(candidates) if candidates else m
                    record[f'orders_{timing}_month_lag{lag_days}']=m2
                    for delta_lag in [2,3,4]:
                        needed=list(pd.period_range(shift_month(m2,-(23+delta_lag)),m2,freq='M').astype(str))
                        for scope in ['official','vendor']:
                            tag=f'x2_{scope}_{timing}_d{delta_lag}_lag{lag_days}';value=np.nan
                            if scope=='official':
                                ok=all(z in pm.index and pd.notna(pm.at[z,'value']) and delay_date(pm.at[z,'released'],lag_days)<=pd.Timestamp(row.known) for z in needed)
                                series=pm.value.reindex(needed)
                            else:
                                # Exploratory values, official dates when known, otherwise the frozen composite date.
                                date=pm.at[m2,'released'] if m2 in pm.index else raw_pmi.at[m2,'available']
                                ok=delay_date(date,lag_days)<=pd.Timestamp(row.known)
                                series=vendor['pmi'].pmi010500.reindex(needed)
                            if ok and series.notna().all():
                                delta,vote=transform(series,delta_lag);value=float(vote[-1]<0)
                                assert np.isfinite(vote[-1])
                            record[tag]=value
            grid.append(record)
    f=pd.DataFrame(grid);f.to_csv(O/'monthly_auxiliary_grid.csv',index=False,encoding='utf-8-sig')
    # Never change any source or original account.
    assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in receipt['hashes'].items())
    coverage=[]
    for scope in ['official','vendor']:
        for h,x in [('H1',f'x1_{scope}_d3_lag0'),('H2',f'x2_{scope}_sync_d3_lag0')]:
            for run in ['base','quote']:
                subset=f.loc[f.run.eq(run)];eligible=subset if h=='H1' else subset.loc[subset.growth>0];z=eligible.loc[eligible[x].notna()]
                coverage.append(dict(hypothesis=h,evidence=scope,run=run,all_intervals=len(subset),eligible=len(eligible),available=len(z),missing=len(eligible)-len(z),start=z.date.min() if len(z) else None,end=z.end.max() if len(z) else None,
                    negative_order_count=int(z[x].eq(1).sum()) if h=='H2' else None,positive_order_count=int(z[x].eq(0).sum()) if h=='H2' else None))
    pd.DataFrame(coverage).to_csv(O/'coverage.csv',index=False,encoding='utf-8-sig')
    print('COVERAGE',json.dumps(coverage,ensure_ascii=False),flush=True)
    print('PRIMARY_SOURCE_MONTHS',{k:len(v) for k,v in source_tables.items()},'DISAGREEMENTS',sum(abs(x['difference'])>1e-10 for x in discrepancies),'SOURCE_CONFLICTS',len(conflicts),flush=True)
    (O/'preparation_receipt.json').write_text(json.dumps(dict(original_sources_unchanged=True,known_legacy_controls='frozen existing controls; full original macro PIT not independently rebuilt',official_scope='official monthly-release value chain, not full latest-vintage archive',label_reproduction_max_error=float((f.loc[f.run.eq('base'),'y2'].to_numpy()-monthly.loc[monthly.run.eq('base')].sort_values('date').gap.to_numpy()).max())),indent=2),encoding='utf8')
if __name__=='__main__':main()
