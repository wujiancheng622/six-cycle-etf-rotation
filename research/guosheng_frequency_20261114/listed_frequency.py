"""Restrict GS sleeves to 63 valid returns of the listed ETF itself."""
import pandas as pd
import frequency as f

s=f.s

def main():
    data=s.load()
    rows=[]
    for start in ("2014-01-02","2020-01-02"):
        for bp in (9,20):
            for name in f.CFG["schedules"]:
                frame=data.views.hist
                policy=f.Frequency(frame,data.state,name)
                policy.allowed=data.views.first_actual_return
                with s.capital(s.CAPITAL),s.prior.apply_policy(lambda *_:policy,s.KEYS):
                    a=s.economic(frame,policy,start,bp/10000)
                assert a.signals.known.lt(a.signals.date).all()
                assert a.signals.last_return.le(a.signals.known).all()
                for key, first in policy.allowed.items():
                    assert a.weights.loc[a.weights.index<first,key].abs().sum()==0
                case=f"listed_{start}_{name}_{bp}bp"
                s.OUT=f.OUT
                s.record(a,case,[],{},save=True)
                rows.append(dict(schedule=name,bp=bp,**f.summarize(a,bp,case,start,s.END)))
    pd.DataFrame(rows).to_csv(f.OUT/"listed_frequency.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__":
    main()
