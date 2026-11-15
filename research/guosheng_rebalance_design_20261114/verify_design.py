"""Focused implementation checks and independent saved-account metrics."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rebalance_study as r


def micro_checks():
    dates=pd.bdate_range('2019-01-01','2020-03-31').strftime('%Y-%m-%d')
    t=np.arange(len(dates))
    frame=pd.DataFrame({k:.01*np.sin((j+1.11)*t)+.004*np.cos((j+2.47)*t) for j,k in enumerate(r.s.KEYS)},index=dates)
    state=pd.DataFrame(index=pd.to_datetime(dates))
    state['stage']=1
    state.loc['2020-01-10':,'stage']=2
    state.loc['2020-02-13':,'stage']=5
    state.loc['2020-02-17':,'stage']=6
    state['credit_published']='2018-01-01'
    state['growth_available']=pd.Timestamp('2018-01-01')
    receipts={}
    for name in ('MM','WW','DD','DM','WM','MD','DN'):
        p=r.TwoClock(frame,state,r.SPECS[name])
        for i,day in enumerate(dates[dates>='2020-01-02']):
            known=dates[dates.get_loc(day)-1]
            p.targets(day,known,i,500000)
        rec=pd.DataFrame(p.receipts).set_index('date')
        receipts[name]=rec
    assert '2020-01-13' not in receipts['MM'].index
    assert receipts['DM'].at['2020-01-13','requested']=='/'.join(r.s.ASSETS[2])
    assert receipts['WM'].at['2020-01-13','requested']=='/'.join(r.s.ASSETS[2])
    assert receipts['MD'].at['2020-01-13','requested']=='/'.join(r.s.ASSETS[1])
    assert '2020-02-18' not in receipts['DM'].index  # 5 -> 6 is the same basket
    assert '2020-02-18' not in receipts['DN'].index
    for spec in r.SPECS.values():
        a=r.TwoClock(frame,state,spec)
        altered=frame.copy()
        altered.loc['2020-01-13':]=.25
        modified=state.copy()
        modified.loc['2020-01-13':,'stage']=4
        b=r.TwoClock(altered,modified,spec)
        for i,day in enumerate(dates[(dates>='2020-01-02')&(dates<='2020-01-13')]):
            known=dates[dates.get_loc(day)-1]
            x=a.targets(day,known,i,500000)
            y=b.targets(day,known,i,500000)
            for before,after in zip(x,y):
                if before is None:assert after is None
                else:np.testing.assert_array_equal(before,after)
    np.testing.assert_allclose(r.s.risk_parity(np.diag([4.,9.])),[.6,.4],atol=1e-10)
    return dict(clock_semantics=True,same_basket_phase_change_not_triggered=True,
                future_suffix_invariance_architectures=len(r.SPECS),erc_hand_calculation=True)


def books():
    table=pd.read_csv(r.OUT/'comparison.csv')
    plan=json.loads((r.HERE/'plan.json').read_text())
    assert set(table.case)=={x['id'] for x in plan}
    assert not table.duplicated(['case','period']).any()
    rows=[]
    nmetrics=0
    signal_cache={}
    for c in plan:
        path=r.OUT/'accounts'/c['id']
        meta=json.loads((path/'meta.json').read_text())
        ledger,weights,pnl,trades,signals=[pd.read_parquet(path/f'{k}.parquet') for k in ('ledger','weights','pnl','trades','signals')]
        previous=ledger.nav.shift(1,fill_value=meta['capital'])
        err=float((previous+pnl.sum(axis=1)-ledger.cost-ledger.nav).abs().max())
        assert err<1e-6 and ledger.cash.min()>=-1e-7
        np.testing.assert_allclose(weights.sum(axis=1),1.,atol=1e-9)
        assert weights.min().min()>=-1e-9
        assert signals.known.lt(signals.date).all() and signals.last_return.le(signals.known).all()
        assert trades.known.lt(trades.date).all()
        if c['role']=='main' and c['bp']==9:
            signal_cache[(c['spec']['name'],c['mode'],c['start'])]=signals
        expected=np.zeros(len(trades)) if c['gross'] else np.maximum(5.,trades.notional*.00015)+trades.notional*c['bp']/10000
        np.testing.assert_allclose(trades.cost,expected,atol=1e-8)
        np.testing.assert_allclose(trades.groupby('date').cost.sum().reindex(ledger.index,fill_value=0),ledger.cost,atol=1e-8)
        if 'shares' in trades:
            assert (trades.loc[trades.side.eq('buy'),'shares']%100==0).all()
            assert (trades.shares<=.01*trades.adv+1e-8).all()
        for m in table.loc[table.case.eq(c['id'])].itertuples():
            l=ledger.loc[m.start:m.end]
            pos=ledger.index.get_loc(l.index[0])
            value=meta['capital'] if pos==0 else ledger.nav.iloc[pos-1]
            date=meta['base_date'] if pos==0 else ledger.index[pos-1]
            nav=np.r_[value,l.nav]
            ret=np.diff(nav)/nav[:-1]
            y=(pd.Timestamp(l.index[-1])-pd.Timestamp(date)).days/365.25
            computed=[(nav[-1]/nav[0])**(1/y)-1,ret.std(ddof=1)*np.sqrt(252),np.max(1-nav/np.maximum.accumulate(nav)),nav[-1]/nav[0]-1,l.cost.sum(),(l.cash/l.nav).mean()]
            np.testing.assert_allclose(computed,[m.cagr,m.vol,m.mdd,m.total,m.fees,m.mean_cash],rtol=1e-12,atol=1e-8)
            nmetrics+=len(computed)
        rows.append(dict(case=c['id'],days=len(ledger),orders=len(trades),nav_error=err,minimum_cash=float(ledger.cash.min())))
    pd.DataFrame(rows).to_csv(r.OUT/'account_validation.csv',index=False)
    signal_pairs=0
    for (name,mode,start),signals in signal_cache.items():
        if name in ('DD_band','DM_band'):
            pd.testing.assert_frame_equal(signals,signal_cache[(name.split('_')[0],mode,start)])
            signal_pairs+=1
    return dict(accounts=len(rows),metric_checks=nmetrics,max_nav_error=max(x['nav_error'] for x in rows),
                buffered_signals_exactly_equal_pairs=signal_pairs)


def main():
    receipt={'micro':micro_checks()}
    if (r.OUT/'comparison.csv').exists() and len(pd.read_csv(r.OUT/'comparison.csv').case.unique())==len(r.plans()):
        receipt['books']=books()
    paths=[*r.HERE.glob('*.py'),r.HERE/'protocol.json',r.HERE/'plan.json']
    for p in r.HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    receipt['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (r.OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='sha256'},indent=2),flush=True)


if __name__=='__main__':main()
