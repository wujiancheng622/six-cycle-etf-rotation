"""Causal input-cutoff tests for the new diagnostic allocation wrapper."""
import json,itertools
import numpy as np
import pandas as pd
import run_attribution as r

def main():
    local,views=r.ce.latest_local();_,_,state,events=r.ce.r6_states(local)
    allowed={'proxy_marked':{},'three_index_date_control':{k:views.hist.index[views.hist.index.searchsorted(d,side='right')] for k,d in r.ce.CFG['index_publication_control']['dates'].items()},'listed_ETF_history_control':views.first_actual_return}
    records=[]
    for availability,limits in allowed.items():
        for day in ['2014-06-03','2020-03-03','2025-06-04']:
            j=views.hist.index.get_loc(day);known=views.hist.index[j-1]
            future=views.hist.copy();future.loc[day:]=.20
            fs=state.copy();fs.loc[pd.Timestamp(day):,'stage']=3
            for T,W,E in itertools.product((0,1),repeat=3):
                a=r.FactorPolicy(views.hist,state,T,W,E,limits).targets(day,known,0,500000.)[0]
                b=r.FactorPolicy(future,fs,T,W,E,limits).targets(day,known,0,500000.)[0]
                np.testing.assert_array_equal(a,b)
                assert a.min()>=0 and a.sum()<=1+1e-12
                records.append(dict(test='future_input_causality',availability=availability,day=day,T=T,W=W,E=E,max_error=float(abs(a-b).max())))
            if limits:
                poison=views.hist.copy()
                for key,dt in limits.items():
                    mask=poison.index<dt;poison.loc[mask,key]=np.where(np.arange(mask.sum())%2,.30,-.30)
                a=r.FactorPolicy(views.hist,state,1,1,1,limits).targets(day,known,0,500000.)[0]
                b=r.FactorPolicy(poison,state,1,1,1,limits).targets(day,known,0,500000.)[0]
                np.testing.assert_array_equal(a,b)
                records.append(dict(test='forbidden_history_isolation',availability=availability,day=day,T=1,W=1,E=1,max_error=float(abs(a-b).max())))
    pd.DataFrame(records).to_csv(r.OUT/'focused_tests.csv',index=False)
    print(json.dumps({'tests':len(records),'all_passed':True,'scope':'Diagnostic wrapper causality and banned-row isolation; not macro vintage source validation'},ensure_ascii=False))

if __name__=='__main__':main()
