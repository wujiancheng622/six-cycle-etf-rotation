"""Replay frozen weekly targets in the existing research amount and ETF engines."""
import json
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

from build_signals import HERE, ROOT, RAW, OUT, CFG, END, KEYS, panel, receipt, save, freeze, sessions

# Keep the frozen parent project modules on their expected import path.
sys.dont_write_bytecode = True
for folder in ('guosheng_original_reconstruction_20261114','guosheng_parameter_robustness_20261113',
               'guosheng_fcf_cutoff_20261113','guosheng_six_regime_long_20261112'):
    sys.path.insert(0,str(ROOT / folder))
from run_original import exchange_metals, PublishedIndexRotation
from replay_long import engine, apply_policy
from early_inputs import load_v12, Snapshots
from market import market_inputs as original_market
from run_study import KEYS as OLD_KEYS, Rotation, measures

CAPITAL = float(CFG['trade_cost']['capital_cny'])
CARRIERS = CFG['actual_etf_mapping']
FUNDS = {k: tuple(CARRIERS[k]) for k in KEYS}
FUND_KEYS = tuple(c for k in KEYS for c in FUNDS[k])
ARCHIVE = ROOT/'guosheng_six_regime_long_20261112/data'
PRIOR_RULES = ROOT/'guosheng_original_reconstruction_20261114/protocol.json'


def iso(date):
    return pd.Timestamp(date).strftime('%Y-%m-%d')


def dated(frame):
    z=frame.set_index('trade_date').sort_index()
    z.index = pd.to_datetime(z.index.astype(str)).strftime('%Y-%m-%d')
    assert z.index.is_unique
    return z


def actual_market(inputs):
    cal=pd.Index(sessions(inputs).strftime('%Y-%m-%d'))
    quotes, bars, returns, audit, distributions = {}, {}, {}, {}, []
    for code in FUND_KEYS:
        raw = dated(panel(RAW / f'fund_daily_{code}.parquet',inputs))
        adj = dated(panel(RAW / f'fund_adj_{code}.parquet',inputs)).adj_factor
        band = dated(panel(RAW / f'etf_limit_{code}.parquet',inputs))[['up_limit','down_limit']]
        identity = panel(RAW / f'fund_basic_{code}.parquet',inputs).iloc[0]
        first = iso(identity.list_date)
        assert raw.index[0] == max(cal[cal>='2019-01-01'][0],first) and raw.index[-1] == END
        assert raw.index.equals(adj.index)
        assert (raw[['open','close','pre_close']] > 0).all().all() and adj.gt(0).all()
        div = panel(RAW / f'fund_div_{code}.parquet',inputs)
        div = div.loc[div.div_proc.eq('实施')].drop_duplicates(['record_date','ex_date','pay_date','div_cash'])
        cash = pd.Series(0.,index=raw.index)
        for row in div.itertuples():
            d = iso(row.ex_date)
            if d not in cash:
                continue
            record = iso(row.record_date)
            if code=='511260.SH' and d=='2026-09-18':
                # Previously source-verified in Huatai cash-flow reconstruction.
                assert record == d and abs(row.div_cash-1.2747)<1e-10
                record = '2026-09-17'
            assert record < d <= iso(row.pay_date) and iso(row.imp_anndate) <= record
            assert cash.at[d] == 0
            cash.at[d] = row.div_cash
            distributions.append(dict(key=code,event_id=f'{code}-{d}',record_date=record,ex_date=d,
                                      payment_date=iso(row.pay_date),cash_per_actual_share=float(row.div_cash)))
        ref = raw.close.shift()-cash
        err = (ref-raw.pre_close).abs().dropna()
        mismatch = err[err.gt(.0011)]
        expected = ['2020-02-26'] if code in ('159981.SZ','159985.SZ') else []
        assert mismatch.index.tolist()==expected and (mismatch<=.0021).all(), (code,mismatch)
        factor_jump = adj.pct_change().abs().gt(1e-5)
        assert set(adj.index[factor_jump]) <= set(cash.index[cash.gt(0)])
        ret = (raw.close+cash)/raw.close.shift()-1
        on_calendar=cal[cal>=raw.index[0]]
        assert raw.index.equals(on_calendar)
        assert raw.vol.gt(0).all()
        band=band.reindex(raw.index)
        outside=raw.open.gt(band.up_limit+1e-8) | raw.open.lt(band.down_limit-1e-8)
        bad_dates=outside.index[outside].tolist()
        band.loc[outside | band.index.isin(mismatch.index)]=np.nan
        quotes[code]=raw.rename(columns={'vol':'volume'})
        bars[code]=band
        returns[code]=ret
        audit[code]=dict(listed=first, first=raw.index[0],last=raw.index[-1],days=len(raw),
                         distributions=int(cash.gt(0).sum()), maximum_preclose_residual=float(err.max()),
                         incompatible_open_and_limit=bad_dates, reference_mark_disagreements=mismatch.to_dict(),
                         first_usable_limit=band.up_limit.first_valid_index())
    start = max(z.index[0] for z in quotes.values())
    common=cal[cal>=start]
    panels={f:pd.DataFrame({c:z[f].loc[common] for c,z in quotes.items()}) for f in ('open','close','volume')}
    adv=pd.DataFrame({c:(z.volume*100).rolling(20).mean().shift().loc[common] for c,z in quotes.items()})
    upper=pd.DataFrame({c:z.up_limit.loc[common] for c,z in bars.items()})
    lower=pd.DataFrame({c:z.down_limit.loc[common] for c,z in bars.items()})
    market=SimpleNamespace(panels=panels,returns=pd.DataFrame(returns).loc[common],dividends=distributions,splits=[])
    assert adv.loc[CFG['windows']['actual_starts'][0]:].notna().all().all()
    save('etf_market_checks.json',dict(earliest_common_quote=start,assets=audit,
         negative_or_zero_quote_days=0, raw_quote_limits_policy='No fills where the venue band conflicts with the reported open or the reference mark cannot be reconciled. Unadjusted O/C marks are retained on the latter two dates absent any dated corporate action.'))
    return market,adv,upper,lower


def weights(pair, actual=False):
    w = dict.fromkeys(FUND_KEYS if actual else KEYS,0.)
    for k in pair:
        group = FUNDS[k] if actual else (k,)
        for a in group:
            w[a] = 1 / (len(pair)*len(group))
    assert np.isclose(sum(w.values()),1)
    return w


class WeeklyPolicy:
    def __init__(self, decision_table, name, actual):
        self.data = decision_table
        self.name = name
        self.actual = actual
        self.pending = None
        self.last_trade = -2
        self.receipts = []

    def targets(self, day, known, i, nav):
        # For a cold start after Jan 2015, inherit the last available weekly
        # signal and approximate the existing portfolio with a fresh account.
        refresh = day in self.data.index or (i==0 and len(self.data.loc[:day]))
        if refresh:
            row = self.data.loc[:day].iloc[-1]
            signal = row.signal
            assert signal<=known<day
            if self.name=='equal_four':
                pair = KEYS
            else:
                pair = tuple(row[f'{self.name}_pair'].split('|'))
            w=weights(pair,self.actual)
            self.pending=np.array(list(w.values()))
            self.last_trade=i
            self.receipts.append(dict(date=day,known=known,signal=signal,quadrant=row.quadrant,
                                      state_stage=int(row.stage),selected='|'.join(pair),**w))
            return self.pending,self.pending
        if i==self.last_trade+1:
            return None,self.pending
        return None,None


def economic(name, returns, start, slip):
    """Parent amount model with a cash-feasible minimum commission rule."""
    dates = returns.index
    policy = engine.Policy(name,returns,None)
    position = np.zeros(len(engine.KEYS))
    cash = CAPITAL
    ledger, weight_rows, pnl_rows, trades = [], [], [], []
    suppressed = 0
    active_dates = dates[dates>=start]
    assert dates.get_loc(active_dates[0])>0
    for i, day in enumerate(active_dates):
        j = dates.get_loc(day)
        known = dates[j-1]
        nav = cash + position.sum()
        sell_target,buy_target = policy.targets(day,known,i,nav)

        def difference(target):
            if target is None:
                return np.zeros_like(position)
            change = target*nav-position
            if name=='V12':
                return np.where(np.abs(change)>.05*position,change,0.)
            return change

        ret = returns.iloc[j].to_numpy()
        pnl = position*ret
        sales_requested = -np.minimum(difference(sell_target),0)*(1+ret)
        buys_requested = np.where(sales_requested==0,np.maximum(difference(buy_target),0)*(1+ret),0.)
        position += pnl
        sales = np.minimum(sales_requested,position)
        # A discretionary sale smaller than its own broker charge should
        # not create a fictitious debit in a fully invested account.
        uneconomic = (sales>0) & (sales<=engine.amount_fee(sales,slip))
        suppressed += int(uneconomic.sum())
        sales = np.where(uneconomic,0.,sales)
        sale_cost = engine.amount_fee(sales,slip)
        buys = engine.affordable_amount(buys_requested,cash,slip)
        buy_cost = engine.amount_fee(buys,slip)
        cost = float(sale_cost.sum()+buy_cost.sum())
        cash += float(sales.sum()-buys.sum()-cost)
        position += buys-sales
        nav = cash+position.sum()
        count = 0
        for side,values,fees in [('sell',sales,sale_cost),('buy',buys,buy_cost)]:
            for k in np.flatnonzero(values>1e-9):
                trades.append(dict(date=day,known=known,asset=engine.KEYS[k],side=side,
                                   notional=values[k],cost=fees[k]))
                count += 1
        ledger.append(dict(date=day,nav=nav,cash=cash,receivable=0.,cost=cost,
                           notional=float(sales.sum()+buys.sum()),trade_count=count))
        weight_rows.append(np.r_[position/nav,cash/nav])
        pnl_rows.append(pnl)
    a=engine.pack(ledger,weight_rows,pnl_rows,trades,policy,dates[dates.get_loc(active_dates[0])-1])
    a.suppressed_below_fee_sales=suppressed
    return a


def controls(inputs, labels):
    v12=pd.read_parquet(receipt(ARCHIVE/'returns_v12.parquet',inputs))
    old=pd.read_parquet(receipt(ARCHIVE/'returns_gs.parquet',inputs))
    old,_,_=exchange_metals(old)
    lower=json.loads(receipt(PRIOR_RULES,inputs).read_text(encoding='utf-8'))['index_publication_control']['dates']
    limits={k:old.index[old.index.searchsorted(day,side='right')] for k,day in lower.items()}
    old_labels=pd.read_csv(receipt(ROOT/'guosheng_original_reconstruction_20261114/results/daily_signal.csv',inputs),
                           index_col='known_date',parse_dates=['known_date','growth_available'])
    aligned=old_labels.loc[old_labels.stage.notna()].copy()
    pd.testing.assert_series_equal(labels.loc[aligned.index,'stage'].astype(int),aligned.stage.astype(int),check_names=False,
                                   check_index_type=False)
    return v12,old,limits,old_labels


def save_case(x, mode, start, name, bp):
    location=OUT / 'accounts' / f'{mode}_{start}_{name}_{bp}bp'
    location.mkdir(parents=True,exist_ok=True)
    l,t,p,w,s=x.ledger,x.trades,x.pnl,x.weights,x.signals
    start_value = l.nav.shift(fill_value=CAPITAL)
    err=(start_value+p.sum(axis=1)-l.cost-l.nav).abs().max()
    assert err<=1e-7 and l.cash.min()>=-1e-7 and w.min().min()>=-1e-9
    assert (s.known<s.date).all() and (t.known<t.date).all()
    assert l.index[-1]==END and l.index.is_unique
    if 'signal' in s:
        assert (s.signal<=s.known).all()
    if 'shares' in t:
        assert (t.shares<=t.adv*0.01+1e-8).all()
        assert t.loc[t.side.eq('buy'),'shares'].mod(100).eq(0).all()
    byday=t.groupby('date').agg(cost=('cost','sum'),notional=('notional','sum'),trade_count=('side','size'))
    np.testing.assert_allclose(byday.reindex(l.index,fill_value=0),l[byday.columns],atol=1e-8,rtol=0)
    for k in ('ledger','weights','pnl','trades','signals'):
        getattr(x,k).to_parquet(location / f'{k}.parquet')
    (location/'meta.json').write_text(json.dumps(dict(base_date=x.base_date,max_ledger_error=float(err),capital=CAPITAL,
        suppressed_below_fee_sales=getattr(x,'suppressed_below_fee_sales',0)),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def metrics(x, mode, name, start, bp, tag, lo, hi):
    l=x.ledger.loc[lo:hi]
    if l.empty:
        return None
    n=x.ledger.index.get_loc(l.index[0])
    origin=x.base_date if n==0 else x.ledger.index[n-1]
    value=CAPITAL if n==0 else x.ledger.nav.iloc[n-1]
    r=measures(l,origin,value)
    returns=l.nav / l.nav.shift(fill_value=value)-1
    raw=x.pnl.reindex(l.index).div(l.nav.shift(fill_value=value),axis=0)
    raw=raw.assign(fees=-l.cost / l.nav.shift(fill_value=value))
    np.testing.assert_allclose(raw.sum(axis=1),returns,rtol=0,atol=3e-15)
    scale=np.divide(np.log1p(returns),returns,out=np.ones(len(returns)),where=returns.ne(0))
    linked=raw.mul(scale,axis=0).sum(axis=0)
    np.testing.assert_allclose(linked.sum(),np.log(l.nav.iloc[-1]/value),rtol=0,atol=1e-10)
    a=[dict(mode=mode,account_start=start,strategy=name,slip_bp=bp,period=tag,asset=k,linked_log_pnl=float(v))
       for k,v in linked.items()]
    r.update(mode=mode,strategy=name,account_start=start,slip_bp=bp,period=tag,
             annual_one_way_turnover=r['annual_two_way_turnover']/2,
             signal_count=int(x.signals.date.between(l.index[0],l.index[-1]).sum()),
             mean_cash=float((l.cash/l.nav).mean()))
    if tag=='full':
        nav=pd.concat([pd.Series([value],index=[origin]),l.nav])
        time=pd.DatetimeIndex(pd.to_datetime(nav.index))
        for years in (3,5):
            endpoint=time.searchsorted(time+pd.DateOffset(years=years))
            valid=np.flatnonzero(endpoint<len(time))
            if len(valid):
                right=endpoint[valid]
                elapsed=(time[right]-time[valid]).days/365.25
                growth=nav.to_numpy()[right]/nav.to_numpy()[valid]
                r[f'worst_{years}y_annualized']=float((growth**(1/elapsed)-1).min())
            else:
                r[f'worst_{years}y_annualized']=np.nan
    return r,a


def main():
    h=freeze()
    inputs={}
    signals=pd.read_csv(receipt(OUT/'weekly_targets.csv',inputs),index_col='execute')
    assert signals.index.is_unique and (signals.signal<signals.index).all()
    labels=pd.read_csv(receipt(OUT/'macro_labels.csv',inputs),index_col='date',parse_dates=['date'])
    returns=pd.read_parquet(receipt(OUT/'index_total_prices.parquet',inputs)).pct_change().dropna()
    net=pd.read_parquet(receipt(OUT/'index_net_prices.parquet',inputs)).pct_change().dropna()
    assert returns.index.equals(net.index) and tuple(returns.columns)==KEYS
    assert engine.CAPITAL==CAPITAL
    benchmark,old,limits,old_labels=controls(inputs,labels)
    m=actual_market(inputs)
    actual_v12=load_v12(Snapshots())
    old_market,old_adv,old_hi,old_lo,*_=original_market()
    old_actual=(old_market,old_adv,old_hi,old_lo)
    results,attributions=[],[]
    run_index=CFG['windows']['historical_starts']
    run_actual=CFG['windows']['actual_starts']
    for mode, starts in [('index',run_index),('ETF',run_actual)]:
        actual=mode=='ETF'
        for start in starts:
            for bp in (CFG['trade_cost']['slippage_bps'] if start==starts[0] else [9]):
                slip=bp/10000
                suffix_method=('chart26','expanding','equal_four')
                accounts={}
                for name in suffix_method:
                    def factory(n,r,p):
                        return WeeklyPolicy(signals,name,actual)
                    with apply_policy(factory, FUND_KEYS if actual else KEYS):
                        x=engine.actual(name,*m,None,start,slip) if actual else economic(name,returns,start,slip)
                    accounts[name]=x
                if not actual and start==starts[0] and bp==9:
                    with apply_policy(lambda n,r,p:WeeklyPolicy(signals,'chart26',False),KEYS):
                        accounts['chart26_bond_net']=economic('chart26_bond_net',net,start,slip)
                accounts['V12']=(engine.actual('V12',*actual_v12,None,start,slip) if actual else
                                 economic('V12',benchmark,start,slip))
                if not actual:
                    with apply_policy(lambda n,r,p:PublishedIndexRotation(r,old_labels,limits,True,True),OLD_KEYS):
                        accounts['six_cycle']=economic('six_cycle',old,start,slip)
                elif start>='2025-06-04':
                    with apply_policy(lambda n,r,p:Rotation('GS_stage_switch',r,old_labels),OLD_KEYS):
                        accounts['six_cycle']=engine.actual('six_cycle', *old_actual, None, start,slip)
                dates=accounts['V12'].ledger.index
                for name,x in accounts.items():
                    assert x.ledger.index.equals(dates),(mode,start,name)
                    print(mode,start,bp,name,round(x.ledger.nav.iloc[-1],2),flush=True)
                    save_case(x,mode,start,name,bp)
                    stop=CFG['windows']['report_date_comparison_end']
                    periods = [('full',start,END),('report_date',start,stop),('post_report','2026-05-06',END),
                               ('2026_H2','2026-07-01',END)]
                    if start==starts[0] and bp==9:
                        periods += [('2015–2019','2015-01-01','2019-12-31'),('2020–2022','2020-01-01','2022-12-31'),
                                    ('2023–2025','2023-01-01','2025-12-31')]
                        for year in range(int(start[:4]),2027):
                            periods.append((str(year),f'{year}-01-01',f'{year}-12-31'))
                    for tag,lo,hi in periods:
                        result=metrics(x,mode,name,start,bp,tag,lo,hi)
                        if result is not None:
                            r,a=result
                            results.append(r);attributions.extend(a)
    pd.DataFrame(results).to_csv(OUT/'comparison.csv',encoding='utf-8-sig',index=False)
    pd.DataFrame(attributions).to_csv(OUT/'attribution.csv',encoding='utf-8-sig',index=False)
    save('replay_checks.json',dict(protocol_sha256=h,inputs_sha256=inputs,comparison_rows=len(results),
         prior_reference='Original six-cycle source reconstruction with public index limits in amount mode and actual ETF returns in ETF mode.',
         fees='Minimum CNY5, commission 1.5bp and slippage 9/20bp, per transaction. Same funding semantics.',
         verified=['all accounts have matching daily dates, cold starts and cash basis','continuously reconciled NAV and costs',
                   'date separation, order sizing and lot and participation limits in ETF accounts']))


if __name__=='__main__':
    main()
