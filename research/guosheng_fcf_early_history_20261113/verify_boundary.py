"""Verify earliest-entry timing and the marginal pre-2011 data gap."""
import pandas as pd

from replay_early import HERE, SOURCE, extended_credit, extended_pmi, macro_inputs


def main():
    t=pd.read_csv(HERE/"sources/credit_2008_extraction.csv",encoding="utf-8-sig")
    missing=t.loc[t[["household_reported_yi","firm_reported_yi"]].isna().any(axis=1),"month"].tolist()
    assert missing==["2008-11"]
    cal=pd.read_csv(SOURCE["calendar"],dtype={"cal_date":str})
    days=pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.eq(1),"cal_date"],format="%Y%m%d")).sort_values()
    c=extended_credit()
    growth=extended_pmi()
    # The November 2008 release gives a total household increment, but
    # omits its maturity split. Do not substitute the total for the long leg.
    print("Incomplete 2008 maturity split:",missing)
    p=macro_inputs(days[(days>="2010-01-01")&(days<="2012-01-31")],c,growth)
    assert p.stage.first_valid_index()==pd.Timestamp("2011-04-14")
    first=HERE/"targets.csv"
    s=pd.read_csv(first).query("study=='extended_first'").iloc[0]
    assert (s.known,s.date,s.credit_date,s.growth_date)==("2011-04-14","2011-04-15","2011-04-14","2011-04-03")
    assert int(s.phase)==5
    print("First decision",s.known,"first trade",s.date,"stage",s.phase,
          "uses",s.selected)
    # What limits the original common five-sleeve run if the gap is solved?
    print("The common 63-return warmup is 2011-04-13, two sessions earlier.")


if __name__=="__main__":
    main()
