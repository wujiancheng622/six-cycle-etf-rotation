from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ah_premium_20261106'))
from data_access import OfficialData
client=OfficialData();out=[];p=Path(__file__).resolve().parent/'data_probe'
for i,args in enumerate([{}, {'date':'20261020'},{'start_date':'20250101','end_date':'20261113'}]):
    try:
        f=client.query('shibor_lpr',args)
        f.to_csv(p/f'lpr_diagnose_{i}.csv',index=False)
        r=dict(params=args,rows=len(f),columns=list(f),latest=f.head(5).to_dict('records'),oldest=f.tail(2).to_dict('records'))
    except Exception as e:r=dict(params=args,error=str(e))
    out.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
(p/'lpr_diagnose.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
