"""Round 2: frozen cash decomposition and symmetric actual-ETF availability diagnostics."""
import sys,socket,json,hashlib,contextlib,gzip,base64
from pathlib import Path
sys.dont_write_bytecode=True
def offline(*a,**k): raise RuntimeError("Round2 archive-only: network disabled")
socket.socket.connect=offline
socket.create_connection=offline
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"guosheng_parameter_robustness_20261113"))
from study_inputs import DATA,PARENT,parent,engine,apply_policy,KEYS,PROVIDER,risk_parity,Rotation
from run_study import ASSETS_BY_STAGE,measures
sys.path.insert(0,str(PARENT))
from verify_long import PriorToCash
from v12.config import BASE as VBASE,KEYS as VKEYS,CODES
import numpy as np,pandas as pd
PROTOCOL=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
digest=hashlib.sha256((HERE/"protocol.json").read_bytes()).hexdigest()
assert digest==(HERE/"protocol.sha256").read_text().strip()
OUT=HERE/"results";OUT.mkdir(exist_ok=True)
HIST=pd.read_parquet(DATA/"returns_gs.parquet")
VOLD=pd.read_parquet(DATA/"returns_v12.parquet")
MACRO=pd.read_parquet(DATA/"macro.parquet")
COVER=json.loads((PARENT/"coverage.json").read_text(encoding="utf-8"))
orig_policy=engine.Policy
manifest={str(p.relative_to(ROOT)):PROVIDER.sha(p) for p in [DATA/"returns_gs.parquet",DATA/"returns_v12.parquet",DATA/"macro.parquet",PARENT/"coverage.json"]}
cut=HIST.index[HIST.index.searchsorted("2024-08-15",side="right")]

def train(r,known,assets):
    if not assets: return r.loc[:known,[]].tail(63)
    ex={d for k in assets for d in r.attrs.get("excluded_risk_dates",{}).get(k,())}
    return r.loc[:known,list(assets)].drop(index=sorted(ex),errors="ignore").dropna().tail(63)
def erc(t):
    if len(t.columns)==1:return np.ones(1)
    return risk_parity(t.cov().to_numpy())
class BoundaryGS(Rotation):
    def __init__(self,r,phases,cutoffs,risk_mode="eligible",cash_mode="equal",fcf_only=False):
        super().__init__("GS_monthly",r,phases)
        self.cutoffs=cutoffs;self.risk_mode=risk_mode;self.cash_mode=cash_mode;self.fcf_only=fcf_only
        self.clean=r.copy()
        for k,d in cutoffs.items(): self.clean.loc[self.clean.index<d,k]=np.nan
    def targets(self,day,known,i,nav):
        macro=self.states.loc[pd.Timestamp(known)]
        if pd.isna(macro.stage):return None,None
        if self.pending is None or day[:7]!=known[:7]:
            phase=int(macro.stage);original=list(ASSETS_BY_STAGE[phase])
            available=original.copy()
            if self.fcf_only:
                if "fcf" in original and len(train(self.clean,known,original))<63:available.remove("fcf")
            else:
                available=[k for k in original if len(train(self.clean,known,[k]))==63]
                if available and len(train(self.clean,known,available))<63:available=[]
            # Full-proxy reference is deliberately needed only in contaminated diagnostic arms.
            if self.risk_mode=="proxy" or self.cash_mode=="reference":
                full=erc(train(self.returns,known,original));reference=dict(zip(original,full))
            else:reference={}
            funded=len(available)/len(original) if self.cash_mode=="equal" else sum(reference[k] for k in available)
            t=train(self.clean,known,available)
            if not available:w=np.zeros(len(KEYS))
            else:
                assert len(t)==63
                if self.risk_mode=="proxy":
                    a=np.array([reference[k] for k in available]);a=a/a.sum()
                else:a=erc(t)
                w=pd.Series(a*funded,index=available).reindex(KEYS,fill_value=0.).to_numpy()
            self.last_rebalance=i;self.pending=w;self.previous_assets=tuple(available)
            self.receipts.append(dict(date=day,known=known,phase=phase,requested="/".join(original),selected="/".join(available),
                first_return=t.index[0] if available else None,last_return=t.index[-1] if available else None,
                cash_target=1-float(w.sum()),**dict(zip(KEYS,w))))
            return w,w
        if i==self.last_rebalance+1:return None,self.pending
        return None,None
class BoundaryV12:
    def __init__(self,r,first):
        self.r=r;self.clean=r.copy();self.receipts=[]
        for k,d in first.items():self.clean.loc[self.clean.index<d,k]=np.nan
    def targets(self,day,known,i,nav):
        avail=[k for k in VKEYS if len(train(self.clean,known,[k]))==63]
        t=train(self.clean,known,avail)
        if avail and len(t)<63:avail=[]
        w=np.zeros(len(VKEYS));scale=0.
        if avail:
            ix=[VKEYS.index(k) for k in avail];b=np.array(VBASE)[ix]
            sample=t.cov().to_numpy()
            cov=.75*sample+.25*np.diag(np.diag(sample))
            vol=float(np.sqrt(252*b@cov@b))
            scale=min(1.,.12/vol) if vol>1e-12 else 1.
            w[ix]=b*scale
        self.receipts.append(dict(date=day,known=known,selected="/".join(avail),cash_target=1-float(w.sum()),scale=scale,**dict(zip(VKEYS,w))))
        return w,w


class AlignedV12:
    def __init__(self,r):
        self.r=r;self.receipts=[]
    def targets(self,day,known,i,nav):
        t=train(self.r,known,VKEYS)
        assert len(t)==63
        cov=t.cov().to_numpy();cov=.75*cov+.25*np.diag(np.diag(cov))
        b=np.array(VBASE);vol=float(np.sqrt(252*b@cov@b));scale=min(1.,.12/vol)
        w=b*scale
        self.receipts.append(dict(date=day,known=known,first_return=t.index[0],last_return=t.index[-1],
            cash_target=1-float(w.sum()),scale=scale,**dict(zip(VKEYS,w))))
        return w,w

def economic_masked(name,returns,prices,start,slip=.0009):
    """Original amount ledger with explicit no-fill gates on missing-quote dates."""
    calendar=returns.index;dates=calendar[calendar>=start];keys=engine.KEYS
    policy=engine.Policy(name,returns,prices)
    position=np.zeros(len(keys));cash=engine.CAPITAL
    ledger=[];weights=[];pnls=[];trades=[]
    missing=returns.attrs.get("untradable_dates",{})
    for i,day in enumerate(dates):
        j=calendar.get_loc(day);known=calendar[j-1];nav=cash+position.sum()
        sell_target,buy_target=policy.targets(day,known,i,nav)
        def difference(w):
            if w is None:return np.zeros(len(keys))
            change=w*nav-position
            if name=="V12":change=np.where(np.abs(change)>.05*position,change,0.)
            return change
        r=returns.iloc[j].to_numpy();pnl=position*r
        can=np.array([day not in missing.get(k,[]) for k in keys])
        sales=-np.minimum(difference(sell_target),0)*(1+r)*can
        buys_requested=np.where(sales==0,np.maximum(difference(buy_target),0)*(1+r),0.)*can
        position+=pnl;sales=np.minimum(sales,position)
        sale_cost=engine.amount_fee(sales,slip)
        buys=engine.affordable_amount(buys_requested,min(cash,cash+float((sales-sale_cost).sum())),slip)
        buy_cost=engine.amount_fee(buys,slip);cost=float(sale_cost.sum()+buy_cost.sum())
        cash+=float(sales.sum()-buys.sum()-cost);position+=buys-sales;nav=cash+position.sum();count=0
        for side,values,fees in [("sell",sales,sale_cost),("buy",buys,buy_cost)]:
            for k in np.flatnonzero(values>1e-9):
                trades.append(dict(date=day,known=known,asset=keys[k],side=side,notional=values[k],cost=fees[k]));count+=1
        ledger.append(dict(date=day,nav=nav,cash=cash,receivable=0.,cost=cost,notional=float(sales.sum()+buys.sum()),trade_count=count))
        weights.append(np.r_[position/nav,cash/nav]);pnls.append(pnl)
    result=engine.pack(ledger,weights,pnls,trades,policy,calendar[calendar.get_loc(dates[0])-1])
    for k,days in missing.items():
        assert not ((result.trades.asset==k)&result.trades.date.isin(days)).any()
    return result

rows=[];navrows=[];checks={};accounts={}
def run(case,r,start,capital,slip,factory=None,keys=None,engine_name=None,save=True):
    previous=engine.CAPITAL;engine.CAPITAL=capital
    try:
        runner=economic_masked if r.attrs.get("untradable_dates") else engine.economic
        if factory is None:a=runner(engine_name or "V12",r,None,start,slip)
        else:
            with apply_policy(factory,keys):a=runner(engine_name or case,r,None,start,slip)
    finally:engine.CAPITAL=previous
    assert a.ledger.index.is_unique and a.ledger.index[-1]=="2026-11-12"
    assert (a.signals.known<a.signals.date).all()
    m=measures(a.ledger,a.base_date,capital)
    row=dict(case=case,capital=capital,slip_bps=slip*10000,**m)
    row.update(mean_cash=float(a.weights["cash"].mean()),model_cost_total=float(a.ledger.cost.sum()))
    if save:
        rows.append(row)
        key=case+"__"+start+"__"+str(capital)+"__"+str(int(slip*10000))
        navrows.append(pd.DataFrame(dict(date=a.ledger.index,case=key,base_date=a.base_date,capital=capital,nav=a.ledger.nav.to_numpy())))
        folder=OUT/"accounts"/key;folder.mkdir(parents=True,exist_ok=True)
        for name in ("ledger","weights","pnl","signals","trades"):getattr(a,name).to_parquet(folder/(name+".parquet"))
        (folder/"meta.json").write_text(json.dumps(dict(base_date=a.base_date,capital=capital,metrics=m),default=str))
        accounts[key]=a
        print(json.dumps({k:row[k] for k in ("case","start","capital","slip_bps","cagr","vol","mdd","mean_cash")}),flush=True)
    return a
def export():
    pd.DataFrame(rows).to_csv(OUT/"summary.csv",index=False)
    pd.concat(navrows,ignore_index=True).to_csv(OUT/"daily_nav.csv",index=False)
    (OUT/"checks.json").write_text(json.dumps(checks,indent=2,default=str))
    (OUT/"input_manifest.json").write_text(json.dumps(dict(protocol_sha256=digest,inputs=manifest),indent=2))
def decompose():
    run("legacy_mask_after_risk",HIST,"2013-04-12",1000000,.0009,
        lambda n,r,p:PriorToCash("GS_monthly",r,MACRO,{"fcf":"2024-08-15"}),KEYS)
    for risk in ("proxy","eligible"):
        for cash in ("reference","equal"):
            run("matched_"+risk+"_risk_"+cash+"_cash",HIST,"2013-04-12",1000000,.0009,
                lambda n,r,p,risk=risk,cash=cash:BoundaryGS(r,MACRO,{"fcf":cut},risk,cash,True),KEYS)
    # Excluded source values must have no influence in the strict arm.
    pert=HIST.copy();mask=pert.index<cut;pert.loc[mask,"fcf"]=np.resize([.30,-.30],mask.sum())
    a=run("perturbation_check",pert,"2013-04-12",1000000,.0009,
        lambda n,r,p:BoundaryGS(r,MACRO,{"fcf":cut},"eligible","equal",True),KEYS,save=False)
    baseline=accounts["matched_eligible_risk_equal_cash__2013-04-12__1000000__9"]
    for name in ("ledger","weights","pnl","signals","trades"):
        pd.testing.assert_frame_equal(getattr(a,name),getattr(baseline,name),check_exact=False,atol=1e-8,rtol=0)
    checks["FCF_strict_perturbation_invariant"]=True
    export()
def actual_v12():
    raw=Path(r"\\wsl.localhost\Ubuntu-24.04\home\wjc\src\new_polict\data\ext14_raw")
    real=VOLD.copy();first={};sources=[];risk_exclusions=dict(VOLD.attrs.get("excluded_risk_dates",{}));missing_dates={}
    for k,c in list(CODES.items())[:8]:
        fp=raw/("fund_daily_"+c+".csv");ap=raw/("fund_adj_"+c+".csv")
        f=pd.read_csv(fp,dtype={"trade_date":str}).set_index("trade_date").sort_index()
        af=pd.read_csv(ap,dtype={"trade_date":str}).set_index("trade_date").sort_index()
        nav=f.close*af.adj_factor.reindex(f.index)
        assert nav.notna().all(),(k,"missing factor on observed quote")
        nav.index=pd.to_datetime(nav.index).strftime("%Y-%m-%d")
        first[k]=nav.pct_change(fill_method=None).first_valid_index()
        use=real.index[(real.index<"2020-01-20")&(real.index>=first[k])]
        missing=list(use.difference(nav.index))
        risk_bad=set(risk_exclusions.get(k,[]))
        for d in missing:
            j=real.index.get_loc(d);risk_bad.add(d)
            if j+1<len(real.index):risk_bad.add(real.index[j+1])
        if missing:missing_dates[k]=missing
        risk_exclusions[k]=sorted(risk_bad)
        marked=nav.reindex(real.index).ffill()
        x=marked.pct_change(fill_method=None)
        matched=x.reindex(use)
        assert matched.notna().all(),(k,list(matched[matched.isna()].index[:5]))
        real.loc[use,k]=matched
        sources.append(dict(key=k,first_actual_return=first[k],source=str(fp),rows_replaced=len(use),
            missing_quote_dates=missing,excluded_one_day_risk_dates=sorted(risk_bad),
            conditional_stale_marks=bool(missing)))
        manifest[str(fp)]=PROVIDER.sha(fp);manifest[str(ap)]=PROVIDER.sha(ap)
    # The historical global panel already uses actual QDII market prices after these source starts;
    # primary first eight replacements are audited separately. If gaps are found, abort rather than invent returns.
    for k,day in [("nasdaq","2013-05-16"),("sp500","2014-01-16")]:
        first[k]=day
        sources.append(dict(key=k,first_actual_return=day,source="p75_long_period_comparison_v1/input/global/"+k+"_processed.csv; independently checked boundary",rows_replaced=0))
    real.attrs.update(VOLD.attrs)
    real.attrs["excluded_risk_dates"]=risk_exclusions
    real.attrs["untradable_dates"]=missing_dates
    checks["conditional_missing_quote_handling"]={"missing":missing_dates,"policy":"carry adjusted last mark, forbid fills, retain next-quote cumulative P&L, exclude missing/resumption observations from daily covariance; not proof of actual halt"}
    pd.DataFrame(sources).to_csv(OUT/"availability_v12.csv",index=False)
    real.to_parquet(OUT/"returns_v12_earlier_actual.parquet")
    return real,first
def symmetric():
    vr,vcuts=actual_v12()
    gcuts=COVER["actual_return_from"]
    pd.DataFrame([dict(key=k,first_actual_return=d) for k,d in gcuts.items()]).to_csv(OUT/"availability_gs.csv",index=False)
    for start in ("2014-07-16","2020-04-27"):
        combos=[(500000,.0009),(500000,.002)]
        if start=="2014-07-16":combos.append((1000000,.0009))
        for capital,slip in combos:
            run("GS_original",HIST,start,capital,slip,lambda n,r,p:Rotation("GS_monthly",r,MACRO),KEYS)
            run("GS_ETF_available",HIST,start,capital,slip,lambda n,r,p:BoundaryGS(r,MACRO,gcuts),KEYS)
            run("V12_original",VOLD,start,capital,slip)
            run("V12_earlier_actual",vr,start,capital,slip,lambda n,r,p:AlignedV12(r),VKEYS,engine_name="V12")
            run("V12_ETF_available",vr,start,capital,slip,lambda n,r,p:BoundaryV12(r,vcuts),VKEYS,engine_name="V12")
            export()
    checks["symmetric_scope"]="Actual ETF eligibility/63-day risk mask; macro vintages and current asset-map selection remain unverified"
    export()
if __name__=="__main__":
    decompose()
    symmetric()
    print("ROUND2_PIT_COMPLETE",flush=True)
