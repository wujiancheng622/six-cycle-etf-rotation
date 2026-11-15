"""Read-only source-coverage and state-machine audit; no signal redesign."""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

import components as c


def rate_direction(before,after,before_identity,after_identity):
    if before_identity!=after_identity:return None,'redefined'
    if before is None or after is None:return None,'missing'
    return int(-np.sign(after-before)),'confirmed'


def vote_at(last_change,last_vote,asof,health):
    if health in ('missing','conflict','redefined'):return None
    assert health in ('confirmed','obsolete_documented')
    if health=='obsolete_documented':return 0
    if last_change is None:return 0
    age=(pd.Timestamp(asof)-pd.Timestamp(last_change)).days
    return last_vote if 0<=age<90 else 0


def possible_baskets(observed_sum,unknown_categories,credit,growth,previous_phase,mapping,assets):
    lo=observed_sum-unknown_categories
    hi=observed_sum+unknown_categories
    signs=({-1} if hi<0 else {1} if lo>0 else set(range(-1 if lo<0 else 0,2 if hi>0 else 1)))
    phases={mapping.get((money,credit,growth),previous_phase) for money in signs}
    return sorted({'/'.join(assets[phase]) for phase in phases})


def main():
    data=c.s.load()
    m=c.s.state_path.__globals__['STAGE_MAP']
    state=data.state.loc[c.s.START:].copy()
    states=[]
    for row in state.itertuples():
        key=(int(row.money),int(row.credit),int(row.growth))
        states.append(dict(date=row.Index.date().isoformat(),money=key[0],credit=key[1],growth=key[2],
                           direct_match=key in m,carried_phase=int(row.stage),
                           reason='mapped' if key in m else 'neutral_component' if 0 in key else 'outside_six_states'))
    frame=pd.DataFrame(states)
    frame.to_csv(c.OUT/'state_match_days.csv',index=False,encoding='utf-8-sig')
    episodes=[]
    for _,g in frame.groupby(frame.direct_match.ne(frame.direct_match.shift()).cumsum()):
        if not g.direct_match.iloc[0]:
            episodes.append(dict(start=g.date.iloc[0],end=g.date.iloc[-1],sessions=len(g),
                                 phases='/'.join(map(str,g.carried_phase.unique())),reason='/'.join(g.reason.unique())))
    pd.DataFrame(episodes).to_csv(c.OUT/'unmatched_state_episodes.csv',index=False,encoding='utf-8-sig')
    source=c.ROOT/'guosheng_policy_completion_20261114_r6'
    receipts=json.loads((source/'monthly_receipts.json').read_text(encoding='utf-8'))
    coverage=[]
    for item in receipts:
        match=re.search(r'20\d{2}-\d{2}-\d{2}',item.get('body',''))
        coverage.append(dict(name=item['name'],url=item['url'],publication_date=match.group(0) if match else None,status=item.get('status')))
    dates=pd.DataFrame(coverage)
    dates.to_csv(c.OUT/'monthly_source_coverage.csv',index=False,encoding='utf-8-sig')
    # Mechanism/implementation unit checks, not performance parameter search.
    assert rate_direction(3.,2.9,'SLF_7D','SLF_7D')==(1,'confirmed')
    assert rate_direction(2.9,3.,'SLF_7D','SLF_7D')==(-1,'confirmed')
    assert rate_direction(1.4,1.25,'REPO_7D','REPO_ON')==(None,'redefined')
    assert vote_at('2026-01-03',1,'2026-02-03','confirmed')==1
    assert vote_at('2026-01-03',1,'2026-04-03','confirmed')==0
    assert vote_at('2026-01-03',1,'2026-02-03','missing') is None
    assert vote_at('2026-01-03',1,'2026-01-02','confirmed')==0
    stable=possible_baskets(1,2,-1,-1,5,m,c.s.ASSETS)
    ambiguous=possible_baskets(1,2,1,1,2,m,c.s.ASSETS)
    assert len(stable)==1 and len(ambiguous)>1
    for numerator in range(-8,9):
        assert np.sign(numerator/8)==np.sign(numerator/6)
    original=c.s.risk_parity
    for correlation in (-.7,0.,.7):
        cov=np.array([[4.,6*correlation],[6*correlation,9.]])
        np.testing.assert_allclose(original(cov),[.6,.4],atol=1e-9)
    path=c.ROOT/'guosheng_final_decision_20261114/results/accounts/monthly_2014-01-02_9bp'
    signals=pd.read_parquet(path/'signals.parquet')
    weights=pd.read_parquet(path/'weights.parquet')
    keys=list(c.s.KEYS)
    arities=signals.selected.str.split('/').str.len().value_counts().to_dict()
    observed=data.state.reindex(pd.to_datetime(signals.known))
    matched=[(int(a),int(b),int(d)) in m for a,b,d in observed[['money','credit','growth']].itertuples(index=False,name=None)]
    maxweights=weights[keys].max(axis=1)
    peak=maxweights.idxmax()
    summary=dict(days=len(frame),unmatched_days=int((~frame.direct_match).sum()),
                 reason_counts=frame.reason.value_counts().to_dict(),
                 longest_unmatched=max(episodes,key=lambda z:z['sessions']),
                 monthly_targets=len(signals),monthly_targets_with_carried_phase=int(np.sum(~np.array(matched))),
                 basket_sizes={str(k):int(v) for k,v in arities.items()},
                 median_max_capital_weight=float(maxweights.median()),p95_max_capital_weight=float(maxweights.quantile(.95)),
                 peak_capital_weight=dict(date=peak,asset=weights.loc[peak,keys].idxmax(),weight=float(maxweights.loc[peak])),
                 monthly_source_count=len(dates),monthly_source_latest=dates.publication_date.dropna().max(),
                 data_governance_micro_checks=True,denominator_positive_rescaling_changes_no_stage=True,
                 two_asset_inverse_vol_equals_erc=True,
                 unknown_policy_examples={'same_basket':stable,'different_baskets':ambiguous},
                 limitations='Local source archive coverage only; unmatched states are an existing declared carry rule, not a newly proven bug. Governance examples are offline pure functions, not integrated trading logic.')
    (c.OUT/'input_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
