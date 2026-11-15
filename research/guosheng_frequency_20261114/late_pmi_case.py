"""Replay one documented release-date error without perturbing the other inputs."""
import pandas as pd
import frequency as f

s=f.s

def main():
    data=s.load()
    state=data.state.copy()
    start="2025-11-06"
    pmi=data.pmi
    available=pmi.available.copy()
    assert available.loc["2025-12"]==pd.Timestamp("2025-12-31")
    available.loc["2025-12"]=pd.Timestamp("2026-01-10")
    series=pmi.direction.set_axis(pd.DatetimeIndex(available))
    state["growth"]=s.last_asof(series,state.index)
    dates=pd.Series(available.to_numpy(),index=pd.DatetimeIndex(available))
    state["growth_available"]=s.last_asof(dates,state.index)
    state["stage"]=s.state_path(state.money,state.credit,state.growth)
    rows=[]
    for name in f.CFG["schedules"]:
        for label, frame in (("correct",data.state),("late",state)):
            a=f.play(data,"actual",start,name,9,frame,False)
            jan=a.signals.loc[a.signals.date.str.startswith("2026-01")]
            for row in jan.itertuples():
                rows.append(dict(schedule=name,release=label,date=row.date,known=row.known,phase=row.phase,
                                 basket=row.requested,january_return=s.metrics(a,"2026-01-01","2026-01-31")["total"]))
    pd.DataFrame(rows).to_csv(f.OUT/"late_pmi_january_targets.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__":
    main()
