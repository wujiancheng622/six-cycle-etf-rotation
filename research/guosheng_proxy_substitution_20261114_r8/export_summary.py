from pathlib import Path
import json,gzip,base64,hashlib
import pandas as pd
H=Path(__file__).resolve().parent;O=H/'results'
def table(path):return json.loads(pd.read_csv(path).to_json(orient='records',double_precision=12))
comparisons=[]
for name in ['comparison.csv','capacity_comparison.csv','combined_comparison.csv']:
    f=pd.read_csv(O/name)
    comparisons+=json.loads(f.to_json(orient='records',double_precision=12))
annual=[]
for name in ['annual.csv','capacity_annual.csv','combined_annual.csv']:
    f=pd.read_csv(O/name);f=f.loc[f.account_start.eq('2014-01-02')&f.bp.eq(9)]
    f=f[['label','group','year','total','vol','mdd']]
    annual+=json.loads(f.to_json(orient='records',double_precision=12))
obj=dict(protocol=json.loads((H/'protocol.json').read_text()),comparison=comparisons,annual=annual,
    asset_overlap=table(O/'asset_overlap.csv'),target_changes=table(O/'overlap_target_changes.csv'),
    stress_summary=table(H/'stress/summary.csv'),stress_metadata=json.loads((H/'stress/residual_metadata.json').read_text()),
    capacity=table(O/'fallback_capacity.csv'),validation=json.loads((O/'independent_validation.json').read_text()),
    execution_guard=json.loads((H/'stress/execution_guard_receipts.json').read_text()),
    old_v12_evidence='review_runs/20261107_v12_proxy_validity_v1/FINDINGS.md',
    source_directory=str(H),source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [H/'run_tests.py',H/'stress_tests.py',H/'capacity_replay.py',H/'combined_stress.py',H/'validate.py',O/'comparison.csv',H/'stress/all_scenarios.csv']})
raw=json.dumps(obj,ensure_ascii=False,separators=(',',':')).encode()
(H/'R8_summary.json').write_bytes(raw)
encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
(H/'R8_summary.json.gz.b64').write_text(encoded)
print('BYTES',len(raw),'BASE64',len(encoded),'SHA256',hashlib.sha256(raw).hexdigest())
print(encoded)
