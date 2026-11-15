"""Independent metrics, fee checks, and allocator invariants for this review."""

import ast
import hashlib
import json

import numpy as np
import pandas as pd

import components as c


def main():
    table=pd.read_csv(c.OUT/'comparison.csv')
    plan=json.loads((c.HERE/'plan.json').read_text())
    assert table.id.nunique()==len(plan)==56
    assert not table.duplicated(['id','period']).any()
    checks=[]
    metrics=0
    groups={}
    for case in plan:
        folder=c.OUT/'accounts'/case['id']
        meta=json.loads((folder/'meta.json').read_text())
        ledger,weights,pnl,signals,trades=[pd.read_parquet(folder/f'{k}.parquet') for k in ('ledger','weights','pnl','signals','trades')]
        old=ledger.nav.shift(1,fill_value=meta['capital'])
        err=float((old+pnl.sum(axis=1)-ledger.cost-ledger.nav).abs().max())
        assert err<1e-6 and ledger.cash.min()>=-1e-7
        np.testing.assert_allclose(weights.sum(axis=1),1.,atol=1e-9)
        assert weights.min().min()>=-1e-9
        assert signals.known.lt(signals.date).all() and signals.last_return.le(signals.known).all()
        assert trades.known.lt(trades.date).all()
        expected=np.maximum(5.,trades.notional*.00015)+trades.notional*(case['bp']+case['buy_extra']*trades.side.eq('buy'))/10000
        np.testing.assert_allclose(trades.cost,expected,atol=1e-8)
        np.testing.assert_allclose(trades.groupby('date').cost.sum().reindex(ledger.index,fill_value=0),ledger.cost,atol=1e-8)
        if 'shares' in trades:
            assert (trades.loc[trades.side.eq('buy'),'shares']%100==0).all()
            assert (trades.shares<=.01*trades.adv+1e-8).all()
        for row in table.loc[table.id.eq(case['id'])].itertuples():
            l=ledger.loc[row.start:row.end]
            pos=ledger.index.get_loc(l.index[0])
            base=meta['capital'] if pos==0 else ledger.nav.iloc[pos-1]
            anchor=meta['base_date'] if pos==0 else ledger.index[pos-1]
            nav=np.r_[base,l.nav.to_numpy()]
            ret=np.diff(nav)/nav[:-1]
            years=(pd.Timestamp(l.index[-1])-pd.Timestamp(anchor)).days/365.25
            actual=[(nav[-1]/base)**(1/years)-1,np.std(ret,ddof=1)*np.sqrt(252),np.max(1-nav/np.maximum.accumulate(nav)),
                    nav[-1]/base-1,float(l.cost.sum()),float((l.cash/l.nav).mean())]
            np.testing.assert_allclose(actual,[row.cagr,row.vol,row.mdd,row.total,row.fees,row.mean_cash],rtol=1e-12,atol=1e-8)
            metrics+=6
        if case['role']!='cash_execution':groups[(case['mode'],case['start'],case['bp'],case['frame'],case['rule'])]=signals
        checks.append(dict(id=case['id'],days=len(ledger),orders=len(trades),nav_error=err))
    pair_errors=[]
    for key,signals in groups.items():
        if key[-1] not in ('inverse_vol','equal'):continue
        reference=groups[key[:-1]+('erc',)]
        pd.testing.assert_frame_equal(signals[['date','known','phase','requested','selected']],reference[['date','known','phase','requested','selected']])
        if key[-1]!='inverse_vol':continue
        for a,b in zip(signals.itertuples(),reference.itertuples()):
            assets=a.selected.split('/') if a.selected else []
            if len(assets)==2:
                error=max(abs(getattr(a,k)-getattr(b,k)) for k in assets)
                assert error<1e-8
                pair_errors.append(error)
    pd.DataFrame(checks).to_csv(c.OUT/'account_validation.csv',index=False)
    result=dict(accounts=len(plan),metrics=metrics,max_nav_error=max(x['nav_error'] for x in checks),
                equivalent_two_asset_targets=len(pair_errors),max_two_asset_weight_difference=max(pair_errors),
                signal_basket_dates_unchanged_for_allocation_rules=True,
                disclaimer='Computational self-check on archived data, not independent untouched data or broker execution certification')
    paths=[*c.HERE.glob('*.py'),c.HERE/'protocol.json',c.HERE/'plan.json',c.s.HERE/'study.py']
    for p in c.HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    result['sha256']={str(p.relative_to(c.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (c.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))


if __name__=='__main__':main()
