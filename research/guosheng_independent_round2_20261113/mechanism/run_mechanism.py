"""Fixed 2x2 mechanism experiment. Writes only this isolated directory."""
from pathlib import Path
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import sys, os, json, hashlib, socket, traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
READS=set()
def no_network(*a,**k):
    raise RuntimeError("Offline archived replay: network connections are prohibited")
socket.socket.connect=no_network
socket.create_connection=no_network
def audit(event,args):
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
        if os.fsdecode(args[0]).lower()==os.devnull.lower():
            return
        p=Path(os.fsdecode(args[0])).resolve()
        mode=args[1] or ""
        flags=args[2] or 0
        writing=any(c in str(mode) for c in "wax+") or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and not p.is_relative_to(HERE):
            raise RuntimeError("Write outside isolated experiment blocked: "+str(p))
        if not writing and p.is_relative_to(ROOT) and not p.is_relative_to(HERE) and p.is_file():
            READS.add(p)
sys.addaudithook(audit)
import numpy as np
import pandas as pd
CFG=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def save_csv(frame,name):
    frame.to_csv(HERE/name,index=False,encoding="utf-8-sig")
def stamp():
    return datetime.now(timezone.utc).isoformat()

sys.path.insert(0,str(ROOT/"guosheng_frequency_cost_20261113"))
import frequency_replay as old
Risk=old.archived.Risk
KEYS=old.KEYS
ENGINE=old.engine
MAPPING=old.parent.ASSETS_BY_STAGE
CELLS={"M00":(False,False),"B10":(True,False),"W01":(False,True),"BW11":(True,True)}

class MechanismPolicy:
    def __init__(self, cell, risk, macro):
        self.cell=cell
        self.switch,self.refresh=CELLS[cell]
        self.risk=risk
        self.macro=macro
        self.active=None
        self.pending=None
        self.last=-2
        self.receipts=[]
    def targets(self,day,known,i,nav):
        tick=self.macro.loc[pd.Timestamp(known)]
        assert pd.notna(tick.stage)
        phase=int(tick.stage)
        desired=MAPPING[phase]
        monthly=i==0 or day[:7]!=known[:7]
        changed=not monthly and self.switch and desired!=self.active
        if monthly or changed:
            self.active=desired
        if monthly or changed or self.refresh:
            w,first,last,selected,cash=self.risk.weights(known,self.active,63)
            assert first<=last<=known<day
            assert tick.credit_published<=known and tick.growth_available.date().isoformat()<=known
            if self.cell=="W01" and not monthly:
                assert self.active==self.locked
            if monthly:
                self.locked=self.active
            self.pending=w
            self.last=i
            self.receipts.append(dict(date=day,known=known,phase=phase,
                desired="/".join(desired),requested="/".join(self.active),selected=selected,
                trigger="monthly" if monthly else "basket_change" if changed else "within_basket",
                first_return=first,last_return=last,cash_target=cash,
                credit_date=tick.credit_published,growth_date=tick.growth_available.date().isoformat(),
                **dict(zip(KEYS,w))))
            return w,w
        if i==self.last+1:
            return None,self.pending
        return None,None

def dates_and_returns(a,capital):
    p=a.ledger.nav.shift(1,fill_value=capital)
    r=a.ledger.nav/p-1
    return r,p
def metrics(a,capital,slip,zero_months=()):
    r,p=dates_and_returns(a,capital)
    if zero_months:
        r=r.where(~r.index.str[:7].isin(zero_months),0.)
    nav=np.r_[capital,capital*np.exp(np.log1p(r).cumsum().to_numpy())]
    years=(pd.Timestamp(a.ledger.index[-1])-pd.Timestamp(a.base_date)).days/365.25
    draw=1-nav/np.maximum.accumulate(nav)
    t=a.trades
    commission=(t.notional*.00015).clip(lower=5)
    slipcost=t.notional*slip
    charges=t.assign(commission=commission,slippage=slipcost).groupby("date")[["commission","slippage"]].sum().reindex(a.ledger.index,fill_value=0)
    return dict(base_date=a.base_date,first=a.ledger.index[0],last=a.ledger.index[-1],sessions=len(r),
        total=float(nav[-1]/capital-1),cagr=float(np.expm1(np.log1p(r).sum()/years)),
        vol=float(r.std(ddof=1)*np.sqrt(252)),mdd=float(draw.max()),
        log_return=float(np.log1p(r).sum()),commission=float(commission.sum()),
        slippage=float(slipcost.sum()),total_cost=float(t.cost.sum()),orders=len(t),
        minimum_fee_orders=int(t.notional.lt(5/.00015).sum()),
        annual_cost=float(charges.sum(axis=1).div(p).sum()/years),
        annual_turnover=float(a.ledger.notional.div(p).sum()/years),
        cash_mean=float(a.weights.cash.mean()),
        zero_months="/".join(zero_months),
        note="Costs shown are from actual replay; zero-month diagnostic suppresses net returns, not rerun trades." if zero_months else "")

def verify(a,capital,mode,slip,cell):
    r,p=dates_and_returns(a,capital)
    error=float((p+a.pnl.sum(axis=1)-a.ledger.cost-a.ledger.nav).abs().max())
    assert error<1e-7 and a.ledger.cash.ge(-1e-7).all()
    assert a.ledger.index.is_unique and a.ledger.index.is_monotonic_increasing
    assert a.ledger.index[-1]==CFG["end"]
    assert (a.signals.known<a.signals.date).all()
    np.testing.assert_allclose(a.trades.cost,(a.trades.notional*.00015).clip(lower=5)+a.trades.notional*slip,atol=1e-9,rtol=0)
    if mode=="actual":
        assert a.trades.loc[a.trades.side.eq("buy"),"shares"].mod(100).eq(0).all()
        assert a.trades.shares.le(a.trades.adv*.01+1e-8).all()
        spend=a.trades.loc[a.trades.side.eq("buy")].assign(x=lambda t:t.notional+t.cost).groupby("date").x.sum()
        prior=a.ledger.cash.shift(1,fill_value=capital)
        assert spend.le(prior.reindex(spend.index)+1e-7).all()
    if cell=="W01":
        s=a.signals
        for _,g in s.groupby(s.date.str[:7],sort=True):
            assert g.requested.nunique()==1, "Within-only changed basket within a month"
    return error

def save_account(a,folder,capital,mode,slip,cell):
    folder.mkdir(parents=True,exist_ok=True)
    for k in ("ledger","weights","pnl","trades","signals"):
        getattr(a,k).to_parquet(folder/(k+".parquet"))
    save_json(folder/"meta.json",dict(base_date=a.base_date,capital=capital,mode=mode,slip=slip,cell=cell))

def compare_archived(a,cell):
    mapped={"M00":"monthly","B10":"stage_change","BW11":"daily"}.get(cell)
    if mapped is None:return {"cell":cell,"status":"new_control_no_archive"}
    base=ROOT/"guosheng_frequency_cost_20261113"/"runs"/"actual_2025-11-06"
    candidates=list(base.rglob(mapped+"_ledger.parquet"))
    if not candidates:
        candidates=list((ROOT/"guosheng_frequency_cost_20261113").rglob(mapped+"_ledger.parquet"))
        candidates=[p for p in candidates if "actual_2025-11-06" in str(p)]
    if len(candidates)!=1:
        return dict(cell=cell,status="archive_path_unresolved",candidates=[str(p) for p in candidates])
    ref=pd.read_parquet(candidates[0])
    pd.testing.assert_frame_equal(a.ledger,ref,check_exact=False,rtol=0,atol=1e-8)
    return dict(cell=cell,status="ledger_equal",path=str(candidates[0]),max_nav_error=float((a.ledger.nav-ref.nav).abs().max()))

def aggregate(scope,accounts,capital,macro):
    daily=pd.DataFrame({k:np.log1p(dates_and_returns(a,capital)[0]) for k,a in accounts.items()})
    for k in (*CELLS,"V12"):
        assert k in daily
    for k in ("B10","W01","BW11"):
        daily[k+"_vs_M00"]=daily[k]-daily.M00
    daily["interaction"]=daily.BW11-daily.B10-daily.W01+daily.M00
    daily["basket_main_effect"]=(daily.B10-daily.M00+daily.BW11-daily.W01)/2
    daily["within_main_effect"]=(daily.W01-daily.M00+daily.BW11-daily.B10)/2
    for k in CELLS:daily[k+"_vs_V12"]=daily[k]-daily.V12
    daily.index.name="date"
    daily.reset_index().to_csv(HERE/(scope+"_daily_contrasts.csv"),index=False,encoding="utf-8-sig")
    monthly=daily.groupby(daily.index.str[:7],sort=True).sum().rename_axis("month").reset_index()
    monthly.to_csv(HERE/(scope+"_monthly_contrasts.csv"),index=False,encoding="utf-8-sig")
    s=accounts["BW11"].signals.set_index("date").reindex(daily.index)
    changed=s.requested.ne(s.requested.shift())
    episode=changed.cumsum()
    rows=[]
    for eid,g in daily.groupby(episode,sort=True):
        first,last=g.index[0],g.index[-1]
        row=dict(scope=scope,event_id=int(eid),event_date=first,known=s.at[first,"known"],
            basket=s.at[first,"requested"],end=last,sessions=len(g),**g.sum().to_dict())
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/(scope+"_event_episodes.csv"),index=False,encoding="utf-8-sig")
    np.testing.assert_allclose(pd.DataFrame(rows)[list(daily.columns)].sum(),daily.sum(),atol=1e-12,rtol=0)
    return monthly

def main():
    started=stamp()
    protocol_hash=sha(HERE/"protocol.json")
    source_hash=sha(__file__)
    frozen=HERE/"freeze.json"
    if frozen.exists():
        prior=json.loads(frozen.read_text(encoding="utf-8"))
        assert prior["protocol_sha256"]==protocol_hash and prior["script_sha256"]==source_hash
    else:
        save_json(frozen,dict(frozen_at=started,protocol_sha256=protocol_hash,script_sha256=source_hash,
            note="Frozen before Inputs.load or account replay; all history already observed."))
    data=old.Inputs.load()
    actual_risk=Risk(data.risk_actual)
    hist_risk=Risk(data.hist)
    source_paths={p for p in READS}
    for mod in list(sys.modules.values()):
        value=getattr(mod,"__file__",None)
        if value:
            p=Path(value).resolve()
            if p.is_relative_to(ROOT) and not p.is_relative_to(HERE) and p.is_file():source_paths.add(p)
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(source_paths)}
    save_json(HERE/"input_manifest.json",dict(frozen_at=stamp(),files=manifest))
    print("FROZEN",protocol_hash,"inputs",len(manifest),flush=True)
    all_rows=[];failures=[];reconciles=[];completed=[]
    scopes=[("actual",CFG["recent"]["start"],c,bp) for c in CFG["recent"]["capital"] for bp in CFG["recent"]["slip_bps"]]
    scopes += [("amount",CFG["supplemental"]["start"],c,bp) for c in CFG["supplemental"]["capital"] for bp in CFG["supplemental"]["slip_bps"]]
    for mode,start,capital,bp in scopes:
        scope=f"{mode}_{start}_{capital}_{bp}bp"
        accounts={}
        for cell in (*CELLS,"V12"):
            try:
                risk=actual_risk if mode=="actual" else hist_risk
                patch=old.apply_policy(lambda *args:MechanismPolicy(cell,risk,data.macro),KEYS) if cell!="V12" else nullcontext()
                with old.equity_size(capital),patch:
                    if mode=="actual":
                        q,adv,up,down=data.v12_quotes if cell=="V12" else data.gs_quotes
                        a=ENGINE.actual(cell,q,adv,up,down,None,start,bp/10000)
                    else:
                        a=ENGINE.economic(cell,data.v12 if cell=="V12" else data.hist,None,start,bp/10000)
                error=verify(a,capital,mode,bp/10000,cell)
                save_account(a,HERE/"accounts"/scope/cell,capital,mode,bp/10000,cell)
                if mode=="actual" and capital==1000000 and bp==9 and cell!="V12":
                    reconciles.append(compare_archived(a,cell))
                accounts[cell]=a
                for window,zero in [("full",()),("zero_Jan_Jul_2026",("2026-01","2026-07"))]:
                    all_rows.append(dict(scope=scope,mode=mode,start=start,capital=capital,slip_bp=bp,cell=cell,window=window,**metrics(a,capital,bp/10000,zero)))
                completed.append(dict(scope=scope,cell=cell,error=error))
                print(scope,cell,round(all_rows[-2]["cagr"]*100,4),round(all_rows[-2]["mdd"]*100,4),flush=True)
            except Exception:
                failure=dict(scope=scope,cell=cell,time=stamp(),traceback=traceback.format_exc())
                failures.append(failure)
                save_json(HERE/"failures.json",failures)
                print("FAIL",scope,cell,failure["traceback"],flush=True)
        if len(accounts)==5:aggregate(scope,accounts,capital,data.macro)
        save_csv(pd.DataFrame(all_rows),"metrics.csv")
    unchanged=all(sha(ROOT/p)==h for p,h in manifest.items())
    assert unchanged,"An archived input changed during the experiment"
    save_json(HERE/"reconciliation.json",reconciles)
    save_json(HERE/"failures.json",failures)
    save_json(HERE/"verification.json",dict(started=started,finished=stamp(),completed=completed,
        failed=failures,original_inputs_unchanged=unchanged,source_file_count=len(manifest),
        offline=True,protocol_sha256=protocol_hash,script_sha256=source_hash))
    print("COMPLETE",len(completed),"failures",len(failures),"archive",reconciles,flush=True)
if __name__=="__main__":
    try:main()
    except Exception:
        save_json(HERE/"fatal_error.json",dict(at=stamp(),traceback=traceback.format_exc()))
        raise

