"""Fetch report price inputs through the existing read-only Tushare client."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / 'data'
END = '20261112'
ETF_CODES = ('515800.SH', '511260.SH', '518880.SH', '159980.SZ', '159981.SZ', '159985.SZ')
IDX = (
    ('stock_all', '000985.CSI'), ('wind_all', '881001.WI'),
    ('cni_growth', '399370.SZ'), ('cni_value', '399371.SZ'),
    ('bond_net', 'CBA00631.CS'), ('bond_wealth', 'CBA00601.CS'),
    ('bond_wealth_alt', 'CBA00651.CS'),
    ('nanhua', 'NHCI.NH'), ('equity_tr', 'H00906.CSI'), ('equity_price', '000906.SH'),
)
FIELDS = {
    'index_daily': 'ts_code,trade_date,close,pre_close',
    'fund_daily': 'ts_code,trade_date,open,high,low,close,pre_close,vol,amount',
    'fund_adj': 'ts_code,trade_date,adj_factor',
    'etf_limit': 'ts_code,trade_date,up_limit,down_limit',
    'fund_basic': 'ts_code,name,list_date,delist_date,benchmark,management',
    'fund_div': 'ts_code,ann_date,imp_anndate,div_proc,record_date,ex_date,pay_date,div_cash',
    'sge_daily': 'ts_code,trade_date,close',
}


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def client():
    path = ROOT / 'ah_premium_20261106/data_access.py'
    spec = importlib.util.spec_from_file_location('gs_implied_tushare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OfficialData()


def fetch(api, name, endpoint, params):
    path = RAW / f'{name}.parquet'
    notes = RAW / f'{name}.json'
    if path.is_file() and notes.is_file():
        receipt = json.loads(notes.read_text(encoding='utf-8'))
        assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        return receipt
    receipt = {'name': name, 'endpoint': endpoint, 'params': params,
               'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    try:
        frame = api.query(endpoint, params, FIELDS[endpoint])
    except (RuntimeError, OSError, ValueError) as exc:
        receipt.update(rows=0, error=str(exc)[:225])
    else:
        assert 'ts_code' in frame and set(frame.ts_code.unique()) <= {params['ts_code']}, name
        if 'trade_date' in frame:
            assert frame.trade_date.is_unique
            receipt['first'] = str(frame.trade_date.min()) if len(frame) else None
            receipt['last'] = str(frame.trade_date.max()) if len(frame) else None
            assert len(frame) < 5000, (name, 'possible vendor result truncation')
        frame.to_parquet(path, index=False)
        receipt.update(rows=len(frame), columns=list(frame.columns),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if endpoint == 'fund_basic':
            receipt['identity'] = frame.fillna('').to_dict('records')
    save_json(notes, receipt)
    print(name, receipt['rows'], receipt.get('first'), receipt.get('last'), receipt.get('error', ''), flush=True)
    return receipt


def run(collect_etfs):
    RAW.mkdir(exist_ok=True)
    api = client()
    entries = []
    jobs = []
    if collect_etfs:
        for code in ETF_CODES:
            jobs.append((f'fund_basic_{code}', 'fund_basic', {'ts_code': code, 'market': 'E'}))
            jobs.append((f'fund_div_{code}', 'fund_div', {'ts_code': code}))
            for endpoint in ('fund_daily', 'fund_adj', 'etf_limit'):
                jobs.append((f'{endpoint}_{code}', endpoint,
                             {'ts_code': code, 'start_date': '20190101', 'end_date': END}))
    else:
        for name, code in IDX:
            jobs.extend((f'{name}_{y}', 'index_daily', {'ts_code': code, 'start_date': lo, 'end_date': hi})
                        for y, lo, hi in [('early', '20100101', '20181231'), ('late', '20190101', END)])
        jobs.extend((f'gold_{y}', 'sge_daily', {'ts_code': 'Au99.99', 'start_date': lo, 'end_date': hi})
                    for y, lo, hi in [('early', '20100101', '20181231'), ('late', '20190101', END)])
    for j in jobs:
        entries.append(fetch(api, *j))
    save_json(HERE / ('etf_inputs.json' if collect_etfs else 'index_inputs.json'), entries)


if __name__ == '__main__':
    run(sys.argv[1:] == ['etf'])
