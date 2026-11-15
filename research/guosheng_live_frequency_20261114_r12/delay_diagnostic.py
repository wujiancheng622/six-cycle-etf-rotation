"""Additional fixed 1/3-calendar-day input delay diagnostics, no parameter selection."""
from pathlib import Path
import sys,json
import pandas as pd
sys.dont_write_bytecode=True
import run_frequency as f
s=f.s;out=f.OUT
(out/'delay_protocol.json').write_text(json.dumps(dict(delay_calendar_days=[1,3],policies=['monthly','weekly','daily'],scopes=[['amount','2014-01-02'],['actual','2025-11-06']],bp=9,note='Diagnostic only: shift whole macro package availability; original previous-trading-day information cutoff and fills unchanged. Not empirical error distribution.'),indent=2),encoding='utf8')
d=s.load();rows=[]
for mode,start in [('amount','2014-01-02'),('actual','2025-11-06')]:
    for lag in [1,3]:
        state=s.macro(d,'all_macro_extra_calendar_lag',lag)
        for kind in ['monthly','weekly','daily']:
            with f.selected_policy():a=s.run(d,s.Case(kind),state=state,start=start,bp=9,mode=mode,limit=mode=='amount')
            name=f'delay_{mode}_{start}_{kind}_{lag}d'
            folder=out/'delay_accounts'/name;folder.mkdir(parents=True,exist_ok=True)
            for field in ['ledger','signals','weights','pnl','trades']:getattr(a,field).to_parquet(folder/f'{field}.parquet')
            (folder/'meta.json').write_text(json.dumps(dict(capital=f.CAPITAL,base_date=a.base_date)),encoding='utf8')
            err=float((a.ledger.nav.shift(fill_value=f.CAPITAL)+a.pnl.sum(axis=1)-a.ledger.cost-a.ledger.nav).abs().max())
            assert err<1e-6 and a.signals.known.lt(a.signals.date).all()
            row=dict(case=name,mode=mode,account_start=start,strategy=kind,lag_days=lag,identity_error=err,**f.metrics(a,start,f.END,9));rows.append(row)
            print(json.dumps(row),flush=True)
pd.DataFrame(rows).to_csv(out/'delay_comparison.csv',index=False)
