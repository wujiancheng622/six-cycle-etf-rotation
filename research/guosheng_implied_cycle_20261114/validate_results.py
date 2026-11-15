"""Verify saved account and target evidence and compare the report's dated examples."""
import json

import numpy as np
import pandas as pd

from build_signals import OUT, ROOT, END, digest, freeze, save


SOURCE_EXAMPLES = [
    ('2020-01-19', 2, 1, [.26,.13,.07,.11,.24,.18]),
    ('2020-01-26', 2, 6, [.12,.19,.07,.10,.18,.34]),
    ('2020-02-09', 2, 6, [.01,.03,.02,.04,.08,.83]),
    ('2020-02-16', 2, 6, [.04,.06,.06,.08,.14,.63]),
    ('2020-02-23', 2, 6, [.12,.20,.05,.03,.22,.38]),
    ('2020-03-01', 2, 6, [.06,.14,.09,.03,.09,.58]),
    ('2020-03-08', 1, 1, [.36,.07,.14,.05,.19,.20]),
    ('2024-09-22', 6, 2, [.16,.22,.11,.17,.20,.14]),
    ('2024-09-29', 6, 1, [.41,.25,.07,.07,.13,.07]),
    ('2024-10-06', 6, 1, [.93,.03,0,.02,.01,0]),
    ('2024-10-13', 6, 1, [.87,.06,.01,.03,.03,0]),
    ('2024-10-20', 6, 1, [.88,.08,0,.01,.02,.01]),
    ('2024-10-27', 6, 2, [.27,.34,.08,.11,.10,.10]),
    ('2026-04-19', 2, 2, [.16,.36,.08,.14,.11,.14]),
]


def charts_examples():
    rows=[]
    z=pd.read_csv(OUT/'weekly_predictions.csv',index_col='week_end')
    for day,true,stage,p in SOURCE_EXAMPLES:
        row=z.loc[day]
        expansion=sum(p[:3])>.50
        near_threshold=abs(sum(p[:3])-.50)<.015
        rows.append(dict(report_week_end=day,signal_close=row.signal,report_macro_phase=true,
                         local_macro_phase=int(row.stage),report_implied_phase=stage,
                         local_implied_phase=int(row.argmax_stage),local_quadrant=row.quadrant,
                         report_expansion_proxy=expansion,near_rounding_line=near_threshold,
                         local_expansion=bool(row.expect_expansion),
                         report_p='|'.join(str(round(v*100)) for v in p),
                         local_p='|'.join(f'{row[f"p{k}"]*100:.2f}' for k in range(1,7))))
    return pd.DataFrame(rows)


def main():
    h=freeze()
    signals=json.loads((OUT/'signal_checks.json').read_text(encoding='utf-8'))
    replay=json.loads((OUT/'replay_checks.json').read_text(encoding='utf-8'))
    for r in (signals,replay):
        assert r['protocol_sha256']==h
        for p,sha in r['inputs_sha256'].items():
            assert digest(ROOT / p)==sha,p
    count=0
    for loc in (OUT/'accounts').iterdir():
        l,w,p,o,s=(pd.read_parquet(loc / f'{name}.parquet') for name in ['ledger','weights','pnl','trades','signals'])
        nav=l.nav.shift(fill_value=1000000.)
        np.testing.assert_allclose(l.nav,nav+p.sum(axis=1)-l.cost,rtol=0,atol=1e-7)
        np.testing.assert_allclose(w.sum(axis=1),1,rtol=0,atol=1e-10)
        assert l.cash.ge(-1e-7).all() and (o.known<o.date).all() and (s.known<s.date).all()
        if 'signal' in s:
            assert (s.signal<=s.known).all()
        fees=np.maximum(5.0,.00015*o.notional) + int(loc.name.rsplit('_',1)[1][:-2])/10000*o.notional
        np.testing.assert_allclose(o.cost,fees,rtol=0,atol=1e-8)
        assert l.index[-1]==END
        count+=1
    assert count == 48,count
    examples=charts_examples()
    assert len(examples)==14
    examples.to_csv(OUT/'paper_example_check.csv',encoding='utf-8-sig',index=False)
    compare=pd.read_csv(OUT/'comparison.csv')
    full=compare[compare.period.eq('full')].copy()
    m=full[full.strategy.eq('V12')].set_index(['mode','account_start','slip_bp'])
    v=full[~full.strategy.eq('V12')].copy()
    join=['mode','account_start','slip_bp']
    v=v.join(m[['cagr','mdd','vol']].add_prefix('v12_'),on=join)
    v['cagr_difference']=v.cagr-v.v12_cagr
    v['mdd_difference']=v.mdd-v.v12_mdd
    v['metric_screen']=(v.cagr_difference>=-.03)&(v.mdd_difference<=.05)
    v[join+['strategy','cagr','vol','mdd','v12_cagr','v12_mdd','cagr_difference','mdd_difference','metric_screen']].to_csv(
        OUT/'same_basis_screen.csv',encoding='utf-8-sig',index=False)
    sample=examples.loc[~examples.near_rounding_line]
    print('Accounts',count,'source cases',len(examples),'phase agrees',int(examples.local_implied_phase.eq(examples.report_implied_phase).sum()),
          'expansion/ contraction agrees',int(sample.local_expansion.eq(sample.report_expansion_proxy).sum()),'/',len(sample),flush=True)
    print(v.loc[v.strategy.isin(['chart26','expanding'])&v.slip_bp.eq(9),join+['strategy','metric_screen']].to_string(index=False))
    save('validation.json',dict(protocol_sha256=h,verified_generated_accounts=count,
        report_macro_match=int(examples.local_macro_phase.eq(examples.report_macro_phase).sum()),
        report_argmax_match=int(examples.local_implied_phase.eq(examples.report_implied_phase).sum()),weekly_examples=len(examples),
        report_expansion_agrees_without_rounding=int(sample.local_expansion.eq(sample.report_expansion_proxy).sum()),
        report_expansion_compared=len(sample),daily_nav_fills_limits='Price/volume and completed-cash accounting validated by engine and replay; no exchange fill guarantee.'))


if __name__=='__main__':
    main()
