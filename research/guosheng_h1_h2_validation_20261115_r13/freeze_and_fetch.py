"""Seal definitions before downloading new values; fetch only the registered fields."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,shutil
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parent;I=H/'input';I.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[H/'protocol.json',R/'guosheng_data_expansion_review_20261115/国盛新增分析数据源整改方案_R2交叉评审修订稿.md',R/'guosheng_data_expansion_review_20261115/网页端第二轮修订条款原文.md',R/'guosheng_data_expansion_review_20261115/网页端第二轮评审逐项回应.md',R/'guosheng_final_decision_20261114/study.py',R/'guosheng_timing_attribution_20261115/results/month_detail.csv',R/'guosheng_timing_attribution_20261115/results/asset_month_detail.csv',R/'guosheng_macro_auxiliary_20261115/decision_date_credit.csv']
for name in ['monthly_2014-01-02_9bp','V12_2014-01-02_9bp']:
    for f in ['ledger.parquet','signals.parquet','meta.json']:
        sources.append(R/'guosheng_final_decision_20261114/results/accounts'/name/f)
receipt=H/'freeze_receipt.json'
hashes={str(p.relative_to(R)):sha(p) for p in sources}
if receipt.exists():
    prior=json.loads(receipt.read_text(encoding='utf8'));assert prior['hashes']==hashes
else:
    receipt.write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,prior_status='No new H1/H2 features or fitted results read in this run; prior strategy and existing credit diagnostics already known',no_other_models_guarantee='Cannot certify experiments in uninspected outside work'),ensure_ascii=False,indent=2),encoding='utf8')
    snap=H/'protocol_sources';snap.mkdir(exist_ok=True)
    for p in sources[:4]:shutil.copy2(p,snap/p.name)
print('FROZEN',receipt.read_text(encoding='utf8'),flush=True)
sys.path.insert(0,str(R/'ah_premium_20261106'))
from data_access import OfficialData
c=OfficialData();logs=[]
for api,fields in [('cn_ppi','month,ppi_yoy,ppi_mom,ppi_mp_p_yoy,ppi_mp_p_mom,ppi_mp_qm_yoy,ppi_mp_rm_yoy,ppi_cg_yoy'),('cn_pmi','month,pmi010000,pmi010500,pmi010400,pmi011400,pmi010700,pmi010600,pmi011100')]:
    params={'start_m':'201001','end_m':'202610'};p=I/(api+'.parquet')
    started=datetime.now(timezone.utc).isoformat()
    if not p.exists():f=c.query(api,params,fields);f.to_parquet(p,index=False)
    else:
        import pandas as pd
        f=pd.read_parquet(p)
    assert f.month.is_unique and len(f)<1000
    log=dict(api=api,params=params,fields=fields,received_utc=started,rows=len(f),columns=list(f),first_month=str(f.month.min()),last_month=str(f.month.max()),nulls=f.isna().sum().to_dict(),sha256=sha(p),evidence='current_vendor_snapshot_not_historical_vintage')
    logs.append(log);print('FETCH',json.dumps(log,ensure_ascii=False),flush=True)
(I/'vendor_receipts.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding='utf8')
