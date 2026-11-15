"""Read-only remeasurement of existing paths; no new strategy or order generation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

def measure(nav, base, capital):
    assert nav.index.is_unique and nav.index.is_monotonic_increasing
    values = np.r_[capital, nav.to_numpy(dtype=float)]
    returns = values[1:] / values[:-1] - 1
    years = (pd.Timestamp(nav.index[-1]) - pd.Timestamp(base)).days / 365.25
    return dict(cagr=float((values[-1]/capital)**(1/years)-1),
                vol=float(np.std(returns, ddof=1)*np.sqrt(252)),
                mdd=float((1-values/np.maximum.accumulate(values)).max()),
                total=float(values[-1]/capital-1), days=len(nav))

def main():
    folder = ROOT/'guosheng_original_reconstruction_20261114/results'
    ledger = pd.read_parquet(folder/'daily_ledger.parquet')
    summaries = pd.read_csv(folder/'comparison.csv')
    rows = []
    for case in ['expanded_policy_pmi','rebuilt_monthly']:
        account = ledger[(ledger['case']==case)&(ledger['mode']=='amount')&
                         (ledger.account_start=='2014-01-02')].set_index('date').sort_index()
        for period, low, high in [('paper','2014-01-02','2025-10-31'),
                                  ('2015','2015-01-01','2015-12-31')]:
            nav=account.loc[low:high,'nav']
            before=account.loc[account.index<nav.index[0],'nav']
            base=before.index[-1] if len(before) else '2013-12-31'
            capital=float(before.iloc[-1]) if len(before) else 1_000_000.
            got=measure(nav,base,capital)
            reported=summaries[(summaries['case']==case)&(summaries['mode']=='amount')&
                (summaries.account_start=='2014-01-02')&(summaries.period==period)].iloc[0]
            error=max(abs(got[k]-float(reported[k])) for k in ('cagr','vol','mdd','total'))
            assert error<1e-12, (case,period,error)
            rows.append(dict(study='metals_proxy',case=case,period=period,
                start=nav.index[0],end=nav.index[-1],account_capital=1_000_000,
                max_metric_error=error,**got))

    folder2=ROOT/'guosheng_independent_round2_20261113/pit/results'
    daily=pd.read_csv(folder2/'daily_nav.csv')
    summary2=pd.read_csv(folder2/'summary.csv')
    for case in ('V12_original','V12_earlier_actual'):
        key=case+'__2014-07-16__500000__9'
        account=daily.loc[daily['case'].eq(key)].set_index('date').sort_index()
        assert len(account)>0
        got=measure(account.nav,account.base_date.iloc[0],500_000.)
        reported=summary2[(summary2['case']==case)&(summary2.capital==500000)&
                         (summary2.slip_bps==9)&(summary2.start=='2014-07-16')].iloc[0]
        error=max(abs(got[k]-float(reported[k])) for k in ('cagr','vol','mdd','total'))
        assert error<1e-12
        rows.append(dict(study='V12_earlier_actual',case=case,period='full',
            start=account.index[0],end=account.index[-1],account_capital=500000,
            max_metric_error=error,**got))

    pair=pd.read_csv(folder/'metals_proxy_comparison.csv',index_col=0)
    diff=pair.shfe_close_return-pair.old_proxy
    proxy=dict(n=len(pair),start=str(pair.index[0]),end=str(pair.index[-1]),
        correlation=float(pair.corr().iloc[0,1]),
        annualized_difference_vol=float(diff.std(ddof=1)*np.sqrt(252)),
        caveat='Two prelisting proxies, not proxy versus actual ETF tracking error.')
    sources=[folder/'daily_ledger.parquet',folder/'comparison.csv',
             folder/'metals_proxy_comparison.csv',folder2/'daily_nav.csv',folder2/'summary.csv']
    result=dict(scope='Existing NAV metric remeasurement only; not a fresh raw-data rerun.',
                rows=rows,proxy_pair=proxy,
                hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=True,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT/'remeasured_metrics.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':
    main()
