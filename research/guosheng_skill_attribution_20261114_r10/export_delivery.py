from pathlib import Path
import json,gzip,base64,hashlib
import pandas as pd
H=Path(__file__).resolve().parent;O=H/'results'
def records(f):return json.loads(f.to_json(orient='records',double_precision=12))
C=pd.read_csv(O/'comparison.csv')
cols=['id','mode','availability','account_start','label','bp','window','cagr','vol','mdd','total','mean_cash','annual_turnover','cost_per_year','underwater_calendar_days','worst_3calendar_year_cagr','worst_5calendar_year_cagr']
main=C.loc[C.account_start.eq('2014-01-02')&C.window.isin(['paper','full'])&~C.zero_fees,cols]
recent=C.loc[C['mode'].eq('actual')&~C.zero_fees,cols]
periods=C.loc[C.account_start.eq('2014-01-02')&C.bp.eq(9)&C.window.isin(['2014_2019','2020_2022','2023_2025','2026_H2'])&C.label.isin(['T0W0E0','T0W1E0','T1W1E0','T1W1E1','V12'])&~C.zero_fees,cols]
starts=C.loc[C.account_start.eq('2020-01-02')&C.bp.eq(9)&C.window.eq('full')&C.label.isin(['T0W0E0','T0W1E0','T1W1E0','T1W1E1','V12']),cols]
selected=set(C.loc[C.bp.eq(9)&C.label.isin(['T1W1E0','T1W1E1','V12'])&C.account_start.isin(['2014-01-02','2025-11-06'])&~C.zero_fees,'id'])
tables={'main':records(main),'recent':records(recent),'periods':records(periods),'starts2020':records(starts)}
for name in ['asset_attribution','regime_attribution','average_exposure_decomposition','exposure_group_attribution','drawdown_attribution','transaction_cost_effect','expost_volatility_match','expost_pairwise_risk_match']:
 f=pd.read_csv(O/(name+'.csv'))
 f=f.loc[f.id.isin(selected)&f.window.isin(['paper','actual_full'])]
 tables[name]=records(f)
f=pd.read_csv(O/'factorial_shapley.csv');f=f.loc[f.account_start.eq('2014-01-02')&f.bp.eq(9)&f.window.eq('paper')]
tables['shapley']=records(f)
Y=pd.read_csv(O/'yearly.csv');tables['annual']=records(Y.loc[Y.id.isin(selected),['id','year','total','vol','mdd']])
tables['validation']=json.loads((O/'independent_summary.json').read_text())
tables['completion']=json.loads((O/'completion.json').read_text())
tables['protocol']=json.loads((H/'protocol.json').read_text())
raw=json.dumps(tables,ensure_ascii=False,separators=(',',':')).encode()
(H/'delivery_summary.json').write_bytes(raw)
text=base64.b64encode(gzip.compress(raw,mtime=0)).decode();(H/'delivery_summary.b64').write_text(text)
print('SHA256',hashlib.sha256(raw).hexdigest(),'JSON_BYTES',len(raw),'BASE64_LENGTH',len(text))
print(text)
