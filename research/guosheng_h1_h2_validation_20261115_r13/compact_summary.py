from pathlib import Path
import json
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;O=H/'results'
def records(file,cols=None):
    f=pd.read_csv(O/file)
    if cols:f=f[cols]
    return json.loads(f.to_json(orient='records',double_precision=12))
g=pd.read_csv(O/'monthly_auxiliary_grid.csv');base=g.loc[g.run.eq('base')];weak=base.loc[base.growth.gt(0)&base.x2_official_sync_d3_lag0.eq(1)]
support=pd.read_csv(O/'support_cells.csv');summary=[]
for (h,source),z in support.groupby(['hypothesis','evidence']):
    both=z.loc[z.within_cell_variation]
    summary.append(dict(hypothesis=h,evidence=source,cells=len(z),singleton_cells=int(z.n.eq(1).sum()),cells_with_variation=len(both),observations_in_varying_cells=int(both.n.sum())))
ev=pd.read_csv(O/'leave_event_out.csv');event_summary=[]
for (h,source),z in ev.groupby(['hypothesis','evidence']):
    valid=z.loc[z.status.eq('estimated')]
    event_summary.append(dict(hypothesis=h,evidence=source,events=len(z),estimable=len(valid),min_effect_pp=valid.effect_pp.min(),max_effect_pp=valid.effect_pp.max(),same_as_registered=int((valid.effect_pp.gt(0) if h=='H1' else valid.effect_pp.lt(0)).sum())))
parts=pd.read_csv(O/'partial_regression_contributions.csv');contrib=[]
for (h,source,layer),z in parts.groupby(['hypothesis','evidence','layer']):
    if layer!='price':continue
    z=z.sort_values('beta_contribution');contrib.append(dict(hypothesis=h,evidence=source,total=float(z.beta_contribution.sum()),most_positive=z.iloc[-1].to_dict(),most_negative=z.iloc[0].to_dict()))
result=dict(freeze=json.loads((H/'freeze_receipt.json').read_text(encoding='utf8')),verification=json.loads((O/'verification_summary.json').read_text()),coverage=records('coverage.csv'),
    primary=records('hypothesis_results.csv',['hypothesis','evidence','layer','n','status','effect_pp','lower_pp','upper_pp','bootstrap_valid','df','rank_z','x_residual_fraction','delta_r2']),
    diagnostics=records('registered_diagnostics.csv',['hypothesis','evidence','variant','n','status','effect_pp','baseline_same_dates_pp']),
    segments=records('segments_and_years.csv',['hypothesis','evidence','segment','n','status','effect_pp']),
    baselines=records('unchanged_baseline_reference.csv'),support=summary,event_summary=event_summary,
    weak_order_months=json.loads(weak[['date','end','phase','money','credit','growth','carried','basket','y2','manufacturing_vote','original_pmi_month']].to_json(orient='records',double_precision=12)),
    contributions=contrib,source_comparison=records('same_date_source_check.csv',['hypothesis','evidence','n','effect_pp']),
    scope='No trade overlay, no new strategy backtest, no parameter selection. New information tests only; all legacy data/control assumptions retained.')
(H/'R13_compact.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for key in ['primary','diagnostics','support','event_summary','weak_order_months','baselines','verification']:print(key.upper(),json.dumps(result[key],ensure_ascii=False),flush=True)
