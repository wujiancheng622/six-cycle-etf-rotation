from pathlib import Path
import json
import pandas as pd
H=Path(__file__).resolve().parent;O=H/'results'
def table(p):return json.loads(pd.read_csv(p).to_json(orient='records',double_precision=12))
f=pd.read_csv(O/'comparison.csv')
cols=['case','mode','account_start','strategy','bp','strictFCF','period','cagr','total','vol','mdd','mean_cash','trades','targets','one_way_turnover','commission_yuan','slippage_yuan','fees_yuan','fee_per_nav_year','commission_per_nav_year','min_commission_trades']
f=f.loc[f.period.eq('full')|((f['mode'].eq('actual'))&f.account_start.eq('2025-11-06')&f.bp.eq(9)&f.period.eq('2026_H2')),cols]
y=pd.read_csv(O/'annual.csv');y=y.loc[y.bp.eq(9)&y['mode'].eq('amount')&y.account_start.eq('2014-01-02')&~y.strictFCF,['strategy','year','total','vol','mdd']]
data=dict(comparison=json.loads(f.to_json(orient='records',double_precision=12)),annual=json.loads(y.to_json(orient='records',double_precision=12)),latency=table(O/'signal_latency_verified.csv'),concentration=table(O/'frequency_concentration.csv'),cost_drag=table(O/'cost_drag.csv'),validation=json.loads((O/'independent_summary.json').read_text()),quality=json.loads((O/'data_profile.json').read_text()),delay=table(O/'delay_comparison.csv') if (O/'delay_comparison.csv').exists() else [],source=str(H),protocol=json.loads((H/'protocol.json').read_text()),access=json.loads((H/'data_probe/access_receipts.json').read_text()),lpr=json.loads((H/'data_probe/lpr_diagnose.json').read_text()))
(H/'R12_summary.json').write_text(json.dumps(data,ensure_ascii=False,default=str,indent=2),encoding='utf8')
for key in ['comparison','annual','delay','latency','cost_drag','concentration']:
    print('TABLE',key);print(pd.DataFrame(data[key]).to_csv(index=False))
print('META',json.dumps({k:data[k] for k in ['validation','source','protocol']},ensure_ascii=False))
