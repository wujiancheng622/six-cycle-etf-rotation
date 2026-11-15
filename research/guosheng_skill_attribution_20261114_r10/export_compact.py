from pathlib import Path
import json,gzip,base64,hashlib
import pandas as pd
H=Path(__file__).resolve().parent;O=H/'results'
def split(f):return json.loads(f.to_json(orient='split',index=False,double_precision=12))
C=pd.read_csv(O/'comparison.csv');main=C.loc[C.account_start.eq('2014-01-02')&C.bp.eq(9)&C.window.eq('paper')&~C.zero_fees]
D={'main':split(main[['availability','label','cagr','vol','mdd','mean_cash','annual_turnover','cost_per_year']])}
sel=['T1W1E0','T1W1E1','V12']
z=C.loc[C.account_start.eq('2014-01-02')&C.bp.eq(9)&C.label.isin(sel)&~C.zero_fees]
D['periods']=split(z[['availability','label','window','cagr','total','vol','mdd','underwater_calendar_days','worst_3calendar_year_cagr','worst_5calendar_year_cagr']])
D['actual']=split(C.loc[C['mode'].eq('actual')&C.bp.eq(9),['label','window','total','vol','mdd']])
D['stress20']=split(C.loc[C.account_start.eq('2014-01-02')&C.bp.eq(20)&C.window.eq('paper')&C.label.isin(sel)&~C.zero_fees,['availability','label','cagr','vol','mdd']])
ids=set(z.id)
for name in ['asset_attribution','regime_attribution','transaction_cost_effect','expost_volatility_match','expost_pairwise_risk_match','average_exposure_decomposition','exposure_group_attribution']:
 f=pd.read_csv(O/(name+'.csv'));f=f.loc[f.id.isin(ids)&f.window.eq('paper')].drop(columns='window')
 f['case']=f.id.str.replace('amount_','',regex=False).str.replace('_2014-01-02_9bp_','/',regex=False);f=f.drop(columns='id')
 D[name]=split(f)
f=pd.read_csv(O/'factorial_shapley.csv');D['shapley']=split(f.loc[f.account_start.eq('2014-01-02')&f.bp.eq(9)&f.window.eq('paper')&f.metric.eq('cagr'),['availability','feature','effect','baseline','full']])
f=pd.read_csv(O/'asset_attribution.csv');D['recent_attribution']=split(f.loc[f.id.eq('actual_actual_quotes_2025-11-06_9bp_T1W1E1')&f.window.eq('actual_full'),['asset','log_contribution','share_net_log','money_pnl']])
D['validation']=json.loads((O/'independent_summary.json').read_text());D['manifest']=json.loads((O/'frozen_manifest.json').read_text());D['skill_sources']=json.loads((O/'skill_sources.json').read_text());D['focused_tests']=dict(count=len(pd.read_csv(O/'focused_tests.csv')),max_error=float(pd.read_csv(O/'focused_tests.csv').max_error.max()))
b=json.dumps(D,ensure_ascii=False,separators=(',',':')).encode();(H/'compact_results.json').write_bytes(b)
text=base64.b64encode(gzip.compress(b,mtime=0)).decode();(H/'compact_results.b64').write_text(text)
print('BYTES',len(b),'BASE64',len(text),'SHA',hashlib.sha256(b).hexdigest());print(text)
