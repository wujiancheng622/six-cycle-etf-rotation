"""Generate causal weekly state forecasts, allocation diagnostics, and receipts."""
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / 'data'
OUT = HERE / 'results'
CFG = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
END = CFG['local_cutoff']
PAPER = ROOT / 'hibor_fxbaogao_sources_20261114/pdf/国盛-宏观偏离与市场隐含周期.pdf'
EARLY = ROOT / 'guosheng_fcf_early_history_20261113/phases.csv'
ARCHIVE = ROOT / 'guosheng_original_reconstruction_20261114/results'
SESSION_SOURCE = ROOT / 'yuanda_dividend_momentum_20261114/data/probe/trade_cal.parquet'
KEYS = ('equity', 'treasury', 'commodity', 'gold')
CATEGORIES = ('reverse_repo', 'slf', 'mlf', 'reserve', 'reserve_interest', 'loan', 'deposit')
STAGES = {(1,1,-1):1, (1,1,1):2, (-1,1,1):3, (-1,-1,1):4, (-1,-1,-1):5, (1,-1,-1):6}
CART = {name: tuple(assets) for name, assets in CFG['chart26_frozen'].items()}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def receipt(path, inputs):
    inputs[str(path.relative_to(ROOT)).replace('\\','/')] = digest(path)
    return path


def panel(path, inputs):
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    receipt(path, inputs)
    assert meta['sha256'] == inputs[path.relative_to(ROOT).as_posix()]
    f = pd.read_parquet(path)
    assert len(f) == meta['rows']
    return f


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def freeze():
    h = digest(HERE / 'protocol.json')
    assert h == (HERE / 'protocol.sha256').read_text(encoding='ascii').strip()
    assert digest(PAPER) == CFG['source_sha256']
    return h


def sessions(inputs, cutoff=END):
    f = pd.read_parquet(receipt(SESSION_SOURCE, inputs))
    dates = pd.DatetimeIndex(pd.to_datetime(f.cal_date[f.is_open.eq(1)].astype(str)), name='date').sort_values()
    assert dates.is_unique and pd.Timestamp(cutoff) in dates
    return dates[(dates >= '2010-01-04') & (dates <= cutoff)]


def merge_index(name, inputs):
    frame = pd.concat([panel(RAW / f'{name}_{era}.parquet', inputs) for era in ('early','late')])
    series = frame.set_index(pd.to_datetime(frame.trade_date.astype(str))).close.sort_index().astype(float)
    assert series.index.is_unique and series.gt(0).all()
    return series


def price_inputs(dates, inputs):
    equities, growth, value, commodity, gold, equity_tr = (merge_index(name,inputs) for name in
        ('stock_all','cni_growth','cni_value','nanhua','gold','equity_tr'))
    # Exchange weekly table confirms the vendor gap, also implied by
    # the next published close of 390.64 less its stated +0.47 change.
    gold_day = pd.Timestamp('2022-08-12')
    assert gold_day not in gold.index
    gold.loc[gold_day] = 390.17
    gold = gold.sort_index()
    bond_path = receipt(RAW / 'chinabond_7_10.json', inputs)
    d = json.loads(bond_path.read_text(encoding='utf-8'))
    id_note = json.loads(bond_path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    assert d['dqcName'] == ['中债-7-10年国债指数'] or id_note['name'] == '中债-7-10年国债指数'
    assert id_note['sha256'] == digest(bond_path)

    def bond(k):
        s = pd.Series(d[k], dtype=float)
        s.index = pd.to_datetime(pd.to_numeric(s.index), unit='ms', utc=True).tz_convert('Asia/Shanghai').tz_localize(None).normalize()
        assert s.index.is_unique and s.gt(0).all()
        return s.sort_index()

    net, wealth = bond('JJZS_00'), bond('CFZS_00')
    block = pd.DataFrame(dict(stock_all=equities, growth_value=growth/value, treasury_net=net,
                              commodity=commodity, gold=gold), index=dates)
    alloc = pd.DataFrame(dict(equity=equity_tr, treasury=wealth, commodity=commodity, gold=gold), index=dates)
    alloc_net = alloc.assign(treasury=net.reindex(dates))
    first = pd.Timestamp('2011-08-02')
    for df in (block,alloc,alloc_net):
        assert not df.loc[first:].isna().any().any(), df.loc[first:][df.loc[first:].isna().any(axis=1)]
        assert (df.loc[first:]>0).all().all()
    return block.loc[first:], alloc.loc[first:], alloc_net.loc[first:]


def phases(dates, inputs, events=None):
    def load(p):
        return pd.read_csv(receipt(p,inputs),index_col='known_date',parse_dates=['known_date','credit_published','growth_available']).sort_index()

    early = load(EARLY).loc[:'2012-06-10']
    later = load(ARCHIVE / 'daily_signal.csv').loc['2012-06-11':]
    cols = ['market','credit','growth','credit_published','growth_available']
    states = pd.concat([early[cols],later[cols]]).reindex(dates)
    for col in ('credit_published','growth_available'):
        s = states[col].dropna()
        assert (s <= s.index).all()
    if events is None:
        events = pd.read_csv(receipt(ARCHIVE / 'policy_events_used.csv',inputs),
                             parse_dates=['known_date','effective_date','usable_date'])
    assert (events[['known_date','effective_date']].max(axis=1)==events.usable_date).all()
    votes = pd.DataFrame(index=dates)
    for cat in CATEGORIES:
        options = events[events.category.eq(cat)].groupby('usable_date').vote.agg(['min','max'])
        options['direction'] = options['min'].where(options['min'].eq(options['max']), 0)
        effective = pd.Series(options.index, index=options.index).reindex(dates,method='ffill')
        active = (dates - effective) < pd.Timedelta(days=90)
        votes[cat] = options.direction.reindex(dates,method='ffill').where(active,0.)
    states['money_score'] = (votes.sum(axis=1)+states.market)/8
    states['money'] = np.sign(states.money_score)
    result = []
    current = None
    for values in states[['money','credit','growth']].itertuples(index=False,name=None):
        current = STAGES.get(values,current)
        result.append(current)
    states['stage'] = pd.array(result,dtype='Int64')
    # Verify the causal merge matches the frozen post-2012 archive away from its initial latch.
    compare = states.index.intersection(later.index)
    np.testing.assert_allclose(states.loc[compare,'money_score'],later.loc[compare,'money_score'],rtol=0,atol=1e-12)
    assert states.loc['2014':,'stage'].notna().all()
    return votes.join(states)


def observations(dates, features, labels):
    pairs = pd.DataFrame({'close_date': dates}, index=dates.to_period('W-SUN'))
    weeks = pairs.groupby(level=0).last()
    weeks.index = weeks.index.to_timestamp(how='end').normalize()
    weeks = weeks.loc[weeks.close_date.between(features.index[0],features.index[-1])]
    data = []
    for week_end, t in weeks.close_date.items():
        lower = t-pd.DateOffset(months=1)
        i = features.index.searchsorted(lower,side='right')-1
        if i < 0 or pd.isna(labels.at[t,'stage']):
            continue
        d = features.index[i]
        assert d<=lower<t and t<=week_end
        v = features.loc[t] / features.loc[d] - 1
        data.append(dict(signal=t, week_end=week_end, return_from=d, stage=int(labels.at[t,'stage']),**v.to_dict()))
    return pd.DataFrame(data).set_index('signal')


class SoftmaxFit:
    def __init__(self, x, y):
        self.classes, ix, counts = np.unique(y, return_inverse=True, return_counts=True)
        assert len(self.classes)>=3
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        assert (self.scale>0).all()
        design = self.design(x)
        sample = 1 / counts[ix]
        sample = len(y) * sample / sample.sum()
        target = np.eye(len(self.classes))[ix]
        n, p = design.shape
        shape = (p,len(self.classes))

        def loss(v):
            coeff = v.reshape(shape)
            logits = design @ coeff
            normal = logsumexp(logits,axis=1)
            prob = np.exp(logits-normal[:,None])
            reg = 0.5 * (coeff[1:]**2).sum()
            err = (sample * (normal - logits[np.arange(n),ix])).sum()
            grad = design.T @ (sample[:,None] * (prob-target))
            grad[1:] += coeff[1:]
            return (err+reg)/n, grad.ravel()/n

        fit = minimize(loss,np.zeros(np.prod(shape)),jac=True,method='L-BFGS-B',
                       options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
        assert fit.success and np.max(np.abs(fit.jac))<1e-5, fit.message
        self.coef = fit.x.reshape(shape)
        self.iterations = fit.nit

    def design(self, x):
        x = (x-self.mean)/self.scale
        cross = np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(x.shape[1]),2)])
        return np.column_stack([np.ones(len(x)),x,x**2,cross])

    def predict(self, vector):
        prob = np.zeros((len(vector),6))
        prob[:,self.classes-1] = softmax(self.design(vector) @ self.coef,axis=1)
        np.testing.assert_allclose(prob.sum(axis=1),1.,rtol=0,atol=1e-12)
        return prob


def forecast(obs, drivers):
    records, histories = [], []
    for month, check in obs.loc['2015':].groupby(obs.loc['2015':].index.to_period('M')):
        train = obs.loc[obs.index<month.start_time]
        assert train.index[-1] < check.index[0]
        x, y = train[drivers].to_numpy(),train.stage.to_numpy()
        model = SoftmaxFit(x,y)
        values = model.predict(check[drivers].to_numpy())
        records.append(dict(month=str(month),train_start=train.index[0],train_end=train.index[-1],
                            first_forecast=check.index[0],train_rows=len(train),iterations=model.iterations,
                            **{f'train_stage_{k}': int((y==k).sum()) for k in range(1,7)}))
        row = pd.DataFrame(values,index=check.index,columns=[f'p{k}' for k in range(1,7)])
        histories.append(row.join(check[['week_end','stage','return_from']]).assign(train_end=train.index[-1]))
    table = pd.concat(histories)
    prob = table[[f'p{k}' for k in range(1,7)]].to_numpy()
    onehot = np.eye(6)[table.stage.to_numpy()-1]
    table['loss'] = np.square(prob-onehot).mean(axis=1)
    table['expect_expansion'] = prob[:,:3].sum(axis=1)>0.5
    table['argmax_stage'] = prob.argmax(axis=1)+1
    table['opposite_halves'] = (table.argmax_stage<=3) != (table.stage<=3)
    z, n = [], []
    for t, row in table.iterrows():
        past = table.loc[(table.index>=t-pd.DateOffset(years=6)) & (table.index<t),'loss']
        n.append(len(past))
        std = past.std(ddof=1)
        z.append((row.loss-past.mean())/std if len(past)>=26 and std>0 else np.nan)
    table['prior_divergences'] = n
    table['z'] = z
    table['high_divergence'] = table.opposite_halves | table.z.gt(1)
    table['quadrant'] = [classify(x,h) for x,h in zip(table.expect_expansion,table.high_divergence)]
    return table,pd.DataFrame(records)


def classify(expansion, high):
    if expansion:
        return 'overheated' if high else 'recovery'
    return 'overcold' if high else 'depressed'


def allocate(table, alloc, dates):
    use = table.index[table.index<dates[-1]]
    f = table.loc[use].copy()
    f['execute'] = dates[dates.searchsorted(use,side='right')]
    assert f.execute.is_unique and (f.execute>f.index).all()
    y = f.copy()
    ends = pd.DatetimeIndex(f.execute.iloc[1:]).append(pd.DatetimeIndex([pd.NaT]))
    y['fully_observed_at'] = ends
    x0 = alloc.reindex(f.execute).set_axis(f.index)
    x1 = alloc.reindex(ends).set_axis(f.index)
    realized = np.log(x1/x0)
    checks = []
    for t, r in f.iterrows():
        prior = y.quadrant.eq(r.quadrant) & y.fully_observed_at.le(t)
        assert (y.loc[prior,'fully_observed_at']<=t).all()
        means = realized.loc[prior,KEYS].mean()
        chosen = tuple(means.sort_values(ascending=False,kind='stable').index[:2]) if int(prior.sum())>=12 else KEYS
        pair = CART[r.quadrant]
        checks.append(dict(signal=t, execute=r.execute, quadrant=r.quadrant, stage=int(r.stage),
                           chart26_pair='|'.join(pair), expanding_pair='|'.join(chosen),
                           matured_same_quadrant=int(prior.sum()),
                           last_completed=y.loc[prior,'fully_observed_at'].max()))
    return pd.DataFrame(checks)


def main():
    h=freeze()
    inputs={}
    dates=sessions(inputs)
    signals, alloc, alloc_net = price_inputs(dates,inputs)
    labels=phases(dates,inputs)
    drivers=list(signals.columns)
    obs=observations(sessions(inputs, cutoff='2026-11-13'),signals,labels)
    f, models=forecast(obs,drivers)
    targets=allocate(f,alloc,dates)

    # Causality verification by physically removing all data after a checkpoint.
    cut=pd.Timestamp('2024-09-30')
    before=obs.loc[:cut]
    prefix, _ = forecast(before,drivers)
    pd.testing.assert_frame_equal(prefix,f.loc[:cut])
    # Recompute the already-matured mappings with later state and return data absent.
    limited = allocate(f.loc[:cut],alloc.loc[:cut],dates[dates<=cut])
    pd.testing.assert_frame_equal(limited,targets[targets.signal<cut].reset_index(drop=True))
    example=np.square(np.array([.03,.01,.78,.06,.11,.01]) - np.eye(6)[1]).mean()
    assert round(example*100)==27
    assert len(targets)>550 and targets.execute.min()>=pd.Timestamp('2015-01-05')
    OUT.mkdir(exist_ok=True)
    for name,df in [('macro_labels',labels),('weekly_observations',obs),('weekly_predictions',f),
                    ('monthly_fits',models),('weekly_targets',targets)]:
        df.to_csv(OUT / f'{name}.csv',encoding='utf-8-sig',index=name not in {'monthly_fits','weekly_targets'})
    for name,df in [('index_total_prices',alloc),('index_net_prices',alloc_net)]:
        df.set_axis(df.index.strftime('%Y-%m-%d')).to_parquet(OUT / f'{name}.parquet')
    initial=models.iloc[0]
    save('signal_checks.json',dict(protocol_sha256=h,inputs_sha256=inputs,
         documented_gold_vendor_gap=dict(date='2022-08-12',close=390.17,source='https://www.sge.com.cn/upload/file/202208/16/b4e30b078b584a199c44f0e4cf238bb0.pdf'),
         first_training=str(initial.train_start.date()),initial_training_rows=int(initial.train_rows),
         initial_class_counts=[int(initial[f'train_stage_{i}']) for i in range(1,7)],
         first_week=str(f.index[0].date()),last_forecast=str(f.index[-1].date()),
         weeks=len(f),monthly_fits=len(models),probability_gap_figure19=float(example),
         prefix_check=str(cut.date()),all_mapping_history_observed=True,
         expanding_full_diversification=int(targets.expanding_pair.str.count(r'\|').eq(3).sum()),
         direct_label_accuracy=float(f.argmax_stage.eq(f.stage).mean()),
         four_quadrants=f.quadrant.value_counts().to_dict()))
    print('Frozen signals',len(f),len(models),'monthly fits; initial samples',int(initial.train_rows),
          'classes',[int(initial[f'train_stage_{i}']) for i in range(1,7)], flush=True)
    print(f.quadrant.value_counts().to_dict(),flush=True)


if __name__ == '__main__':
    main()
