"""Fixed allocation alternatives and cash-execution economics."""

from contextlib import contextmanager
from pathlib import Path
import hashlib
import inspect
import json
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'guosheng_final_decision_20261114'))
import study as s

OUT=HERE/'results'
OUT.mkdir(exist_ok=True)
CFG=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))


@contextmanager
def allocation(rule):
    original=s.risk_parity
    if rule=='inverse_vol':
        def inverse(cov):
            w=1/np.sqrt(np.diag(cov))
            return w/w.sum()
        s.risk_parity=inverse
    try:yield
    finally:s.risk_parity=original


def quoted(reuse,buy_extra):
    source=inspect.getsource(s.ENGINE.actual)
    if reuse:
        assert source.count('min(cash_before,book.cash)')==1
        source=source.replace('min(cash_before,book.cash)','book.cash')
    assert source.count("    calendar = market.panels['close'].index")==1
    source=source.replace("    calendar = market.panels['close'].index",
                          "    buy_cfg = dict(cfg, slippage_rate=cfg['slippage_rate']+extra_buy/10000)\n    calendar = market.panels['close'].index")
    source=source.replace('charge = fee(value,cfg)','charge = fee(value,buy_cfg if direction>0 else cfg)')
    needle='buys = affordable(buys,opening,'+('book.cash' if reuse else 'min(cash_before,book.cash)')+',cfg)'
    assert source.count(needle)==1
    source=source.replace(needle,needle[:-4]+'buy_cfg)')
    env=dict(s.ENGINE.__dict__)
    env['extra_buy']=buy_extra
    exec(compile(source,str(HERE/'quote_cost_adapter.py'),'exec'),env)
    return env['actual']


def run(data,rule,mode,start,bp=9,frame_name='base',reuse=False,buy_extra=0):
    frame=data.views.hist
    allowed=None
    fallback=''
    caps=None
    if frame_name in ('old_dividend','old_value'):
        frame=data.full
        fallback=frame_name
        caps=data.caps
    elif frame_name=='listed':allowed=data.views.first_actual_return
    elif frame_name=='normalized':frame=data.normalized
    case=s.Case(rule,equal=rule=='equal',fallback=fallback,permanent=bool(fallback))
    factory=lambda n,r,p:s.Policy(r,r,data.state,case,allowed)
    with s.capital(CFG['cost']['capital']),allocation(rule),s.prior.apply_policy(factory,tuple(frame.columns)):
        if mode=='actual':
            q,adv,up,down=data.views.gs_quotes
            return quoted(reuse,buy_extra)(rule,q,adv,up,down,None,start,bp/10000)
        return s.economic(frame,factory(None,frame,None),start,bp/10000,caps,reuse)


def plans():
    cases=[]
    for rule in CFG['weight_rules']:
        for mode,starts in CFG['windows'].items():
            for start in starts:
                for bp in CFG['cost']['slip_bp']:
                    cases.append(dict(rule=rule,mode=mode,start=start,bp=bp,frame='base',reuse=False,buy_extra=0,role='allocation'))
        for frame in CFG['transfer']:
            for bp in CFG['cost']['slip_bp']:
                cases.append(dict(rule=rule,mode='amount',start='2014-01-02',bp=bp,frame=frame,reuse=False,buy_extra=0,role='transfer'))
    for start in CFG['windows']['actual']:
        for extra in CFG['execution_controls']['extra_buy_slip_bp']:
            cases.append(dict(rule='erc',mode='actual',start=start,bp=9,frame='base',reuse=True,buy_extra=extra,role='cash_execution'))
    for start in CFG['windows']['amount']:
        cases.append(dict(rule='erc',mode='amount',start=start,bp=9,frame='base',reuse=True,buy_extra=0,role='cash_execution'))
    assert len(cases)==CFG['max_accounts']
    return [dict(id='c'+hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12],**c) for c in cases]


def main():
    digest=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    stamp=HERE/'protocol.sha256'
    if stamp.exists():assert stamp.read_text().strip()==digest
    else:stamp.write_text(digest+'\n')
    plan=plans()
    planfile=HERE/'plan.json'
    if planfile.exists():assert json.loads(planfile.read_text())==plan
    else:planfile.write_text(json.dumps(plan,indent=2)+'\n')
    data=s.load()
    nroot=ROOT/'guosheng_return_sources_20261114/results'
    frame=pd.read_parquet(nroot/'GS_nav.parquet')
    adj=pd.read_csv(nroot/'mean_adjustments.csv')
    for item in adj.loc[adj.strategy.eq('GS') & adj.scenario.eq('broad_pre2020') & adj.start.eq('2014-01-02')].itertuples():
        frame[item.asset]-=item.simple_daily_reduction
    data.normalized=frame
    s.OUT=OUT
    rows=[]
    for i,c in enumerate(plan):
        a=run(data,c['rule'],c['mode'],c['start'],c['bp'],c['frame'],c['reuse'],c['buy_extra'])
        expected=np.maximum(5.,a.trades.notional*.00015)+a.trades.notional*(c['bp']+c['buy_extra']*a.trades.side.eq('buy'))/10000
        np.testing.assert_allclose(a.trades.cost,expected,atol=1e-8)
        assert a.ledger.cash.min()>=-1e-7
        assert a.signals.known.lt(a.signals.date).all() and a.signals.last_return.le(a.signals.known).all()
        if c['rule']=='erc' and c['frame']=='base' and not c['reuse'] and c['bp']==9:
            old=(f"monthly_{c['start']}_9bp" if c['mode']=='amount' else f"actual_monthly_{c['start']}_9bp_cash0")
            old=s.ROOT/'guosheng_final_decision_20261114/results/accounts'/old/'ledger.parquet'
            pd.testing.assert_frame_equal(a.ledger,pd.read_parquet(old),check_exact=False,atol=1e-7,rtol=1e-12)
        s.record(a,c['id'],[],{},save=True)
        for period,(lo,hi) in s.PERIODS.items():
            if period=='paper':continue
            m=s.metrics(a,lo,hi)
            if m:
                ledger=a.ledger.loc[m['start']:m['end']]
                ret=ledger.nav/a.ledger.nav.shift(1,fill_value=s.CAPITAL).reindex(ledger.index)-1
                info=dict(c)
                info['account_start']=info.pop('start')
                rows.append(dict(**info,period=period,**m,calmar=m['cagr']/m['mdd'],sharpe_rf0=ret.mean()*252/m['vol']))
        if (i+1)%8==0 or i+1==len(plan):
            s.write(pd.DataFrame(rows),'comparison.csv')
            print(f'{i+1}/{len(plan)}',flush=True)


if __name__=='__main__':main()
