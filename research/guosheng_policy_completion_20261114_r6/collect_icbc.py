"""Read the public date-selector API used by ICBC's own deposit-rate page."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,hashlib,time,subprocess
from urllib.parse import urlencode
HERE=Path(__file__).resolve().parent
OUT=HERE/'sources';OUT.mkdir(exist_ok=True)
BASE='https://papi.icbc.com.cn/interestRate/deposit/'

def get(action,params,name):
    u=BASE+action
    # Windows Schannel supports this server without disabling certificate checks.
    url=u+'?'+urlencode(params)
    result=subprocess.run(['curl.exe','--silent','--show-error','--fail','--max-time','20',url],capture_output=True,check=True)
    p=OUT/(name+'.json');p.write_bytes(result.stdout)
    record=dict(url=url,status=200,file=p.name,sha256=hashlib.sha256(result.stdout).hexdigest(),fetched_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    try:record['response']=json.loads(result.stdout)
    except Exception:record['error']='Not JSON'
    return record

def fetchdate(d):
    try:return get('queryRMBDepositInfo',dict(type='CH',date=d),'icbc_deposit_'+d)
    except Exception as e:return dict(date=d,error=repr(e))

if __name__=='__main__':
    record=get('queryRMBDepositDateList',dict(type='CH'),'icbc_deposit_dates')
    print('DATE_LIST',json.dumps(record,ensure_ascii=False),flush=True)
    dates=record.get('response',{}).get('data',[])
    dates=[str(d) for d in dates if '2009'<=str(d)[:4]<='2026']
    rows=list(ThreadPoolExecutor(4).map(fetchdate,dates))
    (HERE/'icbc_receipts.json').write_text(json.dumps([record]+rows,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows:print(r.get('response',r),flush=True)
