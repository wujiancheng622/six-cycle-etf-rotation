"""Frozen two-clock ablation and bounded rebalance mechanism comparisons."""

from contextlib import contextmanager
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from types import SimpleNamespace
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

CFG=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Spec:
    name:str
    basket:str
    risk:str
    window:int=63
    band:float=0.
    month_offset:int=0
    week_offset:int=0


SPECS={b+r:Spec(b+r,b,r) for b in ('M','W','D') for r in ('M','W','D')}
SPECS['DN']=Spec('DN','D','N')
SPECS['DD_band']=Spec('DD_band','D','D',band=.05)
SPECS['DM_band']=Spec('DM_band','D','M',band=.05)


def clock_dates(calendar,clock,offset=0):
    if clock=='N':return set()
    if clock=='D':return set(calendar)
    dates=pd.DatetimeIndex(calendar)
    groups=dates.to_period('M' if clock=='M' else 'W-SUN')
    frame=pd.DataFrame({'date':calendar,'group':groups})
    return set(frame.groupby('group',sort=False).date.agg(lambda x:x.iloc[min(offset,len(x)-1)]))


class TwoClock(s.Policy):
    def __init__(self,returns,state,spec,allowed=None,fallback=''):
        super().__init__(returns,returns,state,s.Case(spec.name,risk_days=spec.window,
                         fallback=fallback,permanent=bool(fallback)),allowed)
        self.spec=spec
        self.basket_dates=clock_dates(returns.index,spec.basket,spec.month_offset if spec.basket=='M' else spec.week_offset)
        self.risk_dates=clock_dates(returns.index,spec.risk,spec.month_offset if spec.risk=='M' else spec.week_offset)
        self.scheduled={}
        self.active=None
        self.accepted_phase=None
        self.initial_month=None

    def targets(self,day,known,i,nav):
        if self.initial_month is None:self.initial_month=day[:7]
        observed=int(self.state.at[pd.Timestamp(known),'stage'])
        desired=tuple(s.ASSETS[observed])
        basket_due=day in self.basket_dates and not(self.spec.basket=='M' and day[:7]==self.initial_month)
        risk_due=day in self.risk_dates and not(self.spec.risk=='M' and day[:7]==self.initial_month)
        initial=self.active is None
        switched=(initial or basket_due) and desired!=self.active
        if initial or basket_due:
            self.active=desired
            self.accepted_phase=observed
        if initial or switched or risk_due:
            # Delegate exactly the existing eligibility, ERC and receipt logic.
            # A risk-only refresh must use the last accepted basket, not today's
            # unaccepted macro suggestion.
            self.mapping={k:self.active for k in range(1,7)}
            self.pending=None
            result=super().targets(day,known,i,nav)
            receipt=self.receipts[-1]
            receipt.update(trigger='initial' if initial else 'basket_and_risk' if switched and risk_due else 'basket' if switched else 'risk',
                           accepted_phase=self.accepted_phase,observed_phase=observed)
            return result
        if i==self.last+1:return None,self.pending
        return None,None


_risk_original=s.risk_parity
_risk_cache={}
def memo_risk(matrix):
    key=(matrix.shape,matrix.tobytes())
    if key not in _risk_cache:_risk_cache[key]=_risk_original(matrix)
    return _risk_cache[key].copy()


@contextmanager
def fee_context(gross):
    old=s.ENGINE.amount_fee
    if gross:s.ENGINE.amount_fee=lambda values,slip:np.zeros_like(values,dtype=float)
    try:yield
    finally:s.ENGINE.amount_fee=old


def amount_function(band):
    source=inspect.getsource(s.economic)
    assert source.count('.05 * values')==1
    source=source.replace('.05 * values','trade_band * values')
    env=dict(s.__dict__)
    env['trade_band']=band
    exec(compile(source,str(HERE/'amount_adapter.py'),'exec'),env)
    return env['economic']


def actual_function(band,reuse,gross):
    source=inspect.getsource(s.ENGINE.actual)
    needle="relative_buffer=.05 if name=='V12' else 0."
    assert source.count(needle)==1
    source=source.replace(needle,'relative_buffer=trade_band')
    if reuse:
        assert source.count('min(cash_before,book.cash)')==1
        source=source.replace('min(cash_before,book.cash)','book.cash')
    if gross:
        source=source.replace('commission_rate=.00015','commission_rate=0., minimum_commission=0.')
    env=dict(s.ENGINE.__dict__)
    env['trade_band']=band
    exec(compile(source,str(HERE/'quote_adapter.py'),'exec'),env)
    return env['actual']


def run(data,spec,mode,start,bp=9,gross=False,frame_name='base',lag=0,reuse=False):
    returns=data.views.hist
    allowed=None
    fallback=''
    caps=None
    if frame_name in ('old_dividend','old_value'):
        fallback=frame_name
        returns=data.full
        caps=data.caps
    elif frame_name=='listed':allowed=data.views.first_actual_return
    elif frame_name=='normalized':returns=data.normalized
    state=data.states[lag]
    factory=lambda n,frame,p:TwoClock(frame,state,spec,allowed,fallback)
    with s.capital(s.CAPITAL),fee_context(gross),s.prior.apply_policy(factory,tuple(returns.columns)):
        if mode=='actual':
            q,adv,up,down=data.views.gs_quotes
            return actual_function(spec.band,reuse,gross)(spec.name,q,adv,up,down,None,start,0 if gross else bp/10000)
        return amount_function(spec.band)(returns,factory(None,returns,None),start,0 if gross else bp/10000,
                                          caps,reuse,spec.band>0)


def plans():
    cases=[]
    def add(spec,mode,start,bp=9,gross=False,frame='base',lag=0,reuse=False,role='main'):
        payload=dict(spec=asdict(spec),mode=mode,start=start,bp=bp,gross=gross,frame=frame,lag=lag,reuse=reuse,role=role)
        ident=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:12]
        cases.append(dict(id=f'{spec.name}_{ident}',**payload))
    for name in CFG['primary']:
        for mode,starts in CFG['primary_windows'].items():
            for start in starts:
                for bp in CFG['cost']['slip_bp']:add(SPECS[name],mode,start,bp)
    for name in CFG['secondary']:
        for mode,start in (('amount','2014-01-02'),('actual','2025-11-06')):add(SPECS[name],mode,start,role='factorial')
    for spec in SPECS.values():
        for mode,start in (('amount','2014-01-02'),('actual','2025-11-06')):add(spec,mode,start,bp=0,gross=True,role='gross')
    for name in CFG['primary']:
        for frame in ('old_dividend','old_value','listed','normalized'):add(SPECS[name],'amount','2014-01-02',frame=frame,role='transfer')
    for name in CFG['primary']:
        for mode,start in (('amount','2014-01-02'),('actual','2025-11-06')):
            for window in (56,70):add(replace(SPECS[name],window=window),mode,start,role='risk_neighbor')
    for name in ('MM','DM','WM'):
        for mode,start in (('amount','2014-01-02'),('actual','2025-11-06')):
            for offset in (1,2):add(replace(SPECS[name],month_offset=offset),mode,start,role='month_neighbor')
            for lag in (1,3):add(SPECS[name],mode,start,lag=lag,role='release_delay')
    for name in ('WW','WM'):
        for mode,start in (('amount','2014-01-02'),('actual','2025-11-06')):
            for offset in (1,2):add(replace(SPECS[name],week_offset=offset),mode,start,role='week_neighbor')
    for name in ('DD_band','DM_band'):
        for mode,start in (('amount','2014-01-02'),('actual','2025-11-06')):
            for band in (.03,.07):add(replace(SPECS[name],band=band),mode,start,role='band_neighbor')
    for name in CFG['primary']:add(SPECS[name],'actual','2025-11-06',reuse=True,role='same_day_cash')
    assert len(cases)<=CFG['max_accounts'] and len({c['id'] for c in cases})==len(cases)
    return cases


def costs_and_metrics(a,case):
    rows=[]
    spans={k:v for k,v in s.PERIODS.items() if k!='paper'}
    for period,(lo,hi) in spans.items():
        m=s.metrics(a,lo,hi)
        if m is None:continue
        l=a.ledger.loc[m['start']:m['end']]
        previous=a.ledger.nav.shift(1,fill_value=s.CAPITAL).reindex(l.index)
        daily=l.nav/previous-1
        trades=a.trades.loc[a.trades.date.between(l.index[0],l.index[-1])]
        sig=a.signals.loc[a.signals.date.between(l.index[0],l.index[-1])]
        comm=np.zeros(len(trades)) if case['gross'] else np.maximum(5.,trades.notional*.00015)
        slip=np.zeros(len(trades)) if case['gross'] else trades.notional*case['bp']/10000
        np.testing.assert_allclose(trades.cost,comm+slip,atol=1e-8)
        weight=trades.date.map(1/previous)
        vol=m['vol']
        downside=float(np.sqrt(np.mean(np.minimum(daily,0)**2))*np.sqrt(252))
        rows.append(dict(case=case['id'],name=case['spec']['name'],role=case['role'],mode=case['mode'],account_start=case['start'],
                         frame=case['frame'],bp=case['bp'],gross=case['gross'],lag=case['lag'],reuse=case['reuse'],
                         window=case['spec']['window'],band=case['spec']['band'],month_offset=case['spec']['month_offset'],
                         week_offset=case['spec']['week_offset'],period=period,**m,
                         sharpe_rf0=float(daily.mean()*252/vol),sortino_rf0=float(daily.mean()*252/downside),
                         calmar=m['cagr']/m['mdd'],commission=float(np.sum(comm)),slippage=float(np.sum(slip)),
                         annual_commission=float(np.sum(comm*weight)/m['years']),annual_slippage=float(np.sum(slip*weight)/m['years']),
                         orders=len(trades),order_days=int(l.trade_count.gt(0).sum()),
                         initial_targets=int(sig.trigger.eq('initial').sum()),basket_targets=int(sig.trigger.str.contains('basket').sum()),
                         risk_only_targets=int(sig.trigger.eq('risk').sum())))
    return rows


def main():
    plan=plans()
    protocol_hash=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    stamp=HERE/'protocol.sha256'
    if stamp.exists():assert stamp.read_text().strip()==protocol_hash
    else:stamp.write_text(protocol_hash+'\n')
    plan_path=HERE/'plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    print(f'Frozen plan: {len(plan)} accounts, {len(SPECS)} architectures',flush=True)
    data=s.load()
    data.states={0:data.state,1:s.macro(data,'all_macro_extra_calendar_lag',1),3:s.macro(data,'all_macro_extra_calendar_lag',3)}
    nroot=ROOT/'guosheng_return_sources_20261114/results'
    norm=pd.read_parquet(nroot/'GS_nav.parquet')
    adjust=pd.read_csv(nroot/'mean_adjustments.csv')
    for row in adjust.loc[adjust.strategy.eq('GS') & adjust.scenario.eq('broad_pre2020') & adjust.start.eq('2014-01-02')].itertuples():
        norm[row.asset]-=row.simple_daily_reduction
    data.normalized=norm
    s.OUT=OUT
    s.risk_parity=memo_risk
    rows=[]
    reproduction=[]
    legacy={'MM':'monthly','WW':'weekly','DD':'daily','DM':'month_and_switch','DW':'week_and_switch'}
    for i,case in enumerate(plan):
        spec=Spec(**case['spec'])
        folder=OUT/'accounts'/case['id']
        if (folder/'meta.json').exists():
            meta=json.loads((folder/'meta.json').read_text())
            a=SimpleNamespace(**{k:pd.read_parquet(folder/f'{k}.parquet') for k in ('ledger','weights','pnl','signals','trades')},base_date=meta['base_date'])
        else:
            a=run(data,spec,case['mode'],case['start'],case['bp'],case['gross'],case['frame'],case['lag'],case['reuse'])
            assert a.ledger.cash.min()>=-1e-7
            assert a.signals.known.lt(a.signals.date).all() and a.signals.last_return.le(a.signals.known).all()
            s.record(a,case['id'],[],{},save=True)
        if case['role'] in ('main','factorial') and spec.name in legacy and case['bp']==9:
            old=ROOT/'guosheng_frequency_20261114/results/accounts'/f"{case['mode']}_{case['start']}_{legacy[spec.name]}_9bp_0d_cash0"/'ledger.parquet'
            if old.exists():
                prior=pd.read_parquet(old)
                pd.testing.assert_frame_equal(a.ledger,prior,check_exact=False,atol=1e-6,rtol=1e-12)
                reproduction.append(dict(case=case['id'],old=str(old),max_nav_error=float((a.ledger.nav-prior.nav).abs().max())))
        rows.extend(costs_and_metrics(a,case))
        if (i+1)%10==0 or i+1==len(plan):
            s.write(pd.DataFrame(rows),'comparison.csv')
            print(f'{i+1}/{len(plan)} {spec.name} {case["mode"]} {case["role"]}',flush=True)
    s.write(pd.DataFrame(reproduction),'legacy_reproduction.csv')


if __name__=='__main__':main()
