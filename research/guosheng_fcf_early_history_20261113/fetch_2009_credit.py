"""Fetch and extract the dated 2009 PBOC loan announcements."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import re
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "guosheng_six_regime_long_20261112"
sys.dont_write_bytecode = True
sys.path.insert(0,str(PARENT))
from early_probe import pbc,snapshot

OUT = HERE / "sources"
CREDIT_2009 = {
    "01": ("2009-02-12", "2882662"),
    "02": ("2009-03-12", "2827414"),
    "03": ("2009-04-11", "2840978"),
    "04": ("2009-05-11", "2841217"),
    "05": ("2009-06-12", "2893134"),
    "06": ("2009-07-15", "2841454"),
    "07": ("2009-08-11", "2841671"),
    "08": ("2009-09-11", "2884553"),
    "09": ("2009-10-14", "2907152"),
    "10": ("2009-11-11", "2843205"),
    "11": ("2009-12-11", "2843462"),
    "12": ("2010-01-15", "2843694"),
}
CREDIT_2008 = {
    "01": ("2008-02-14", "2826120"),
    "02": ("2008-03-12", "2826193"),
    "03": ("2008-04-11", "2873783"),
    "04": ("2008-05-13", "2826372"),
    "05": ("2008-06-12", "2873804"),
    "06": ("2008-07-14", "2826961"),
    "07": ("2008-08-13", "2826656"),
    "08": ("2008-09-12", "2826723"),
    "09": ("2008-10-14", "2827013"),
    "10": ("2008-11-12", "2896312"),
    "11": ("2008-12-15", "2827035"),
    "12": ("2009-01-13", "2827578"),
}


def source(pair):
    month, (released, file_id) = pair
    url = pbc.ROOT+file_id+"/index.html"
    old_file = PARENT/"data/pbc/releases"/f"{month}.html"
    raw = old_file.read_bytes() if old_file.is_file() else snapshot(url,OUT/f"{month}.html")
    clean = re.sub(r"\s+","",pbc.clean(raw.decode("utf-8")))
    candidates = []
    for m in re.finditer(r"居民户贷款|住户贷款",clean):
        block=clean[m.start():m.start()+300]
        matches=pbc.NUMBERS.findall(block)
        if len(matches)>=2 and ("非金融性公司" in block or "非金融企业" in block
                                or "非金融公司" in block):
            candidates.append(dict(month=month,released=released,url=url,
                                   household_reported_yi=pbc.amount(matches[0]),
                                   firm_reported_yi=pbc.amount(matches[1]),
                                   sha256=sha256(raw).hexdigest(),
                                   lead=clean[max(m.start()-160,0):m.start()],
                                   excerpt=block[:220]))
    if not candidates:
        excerpt = next((clean[max(m.start()-15,0):m.start()+250]
                        for m in re.finditer(r"居民户贷款|住户贷款|中长期贷款",clean)),"")
        return [dict(month=month,released=released,url=url,sha256=sha256(raw).hexdigest(),excerpt=excerpt)]
    return candidates


if __name__=="__main__":
    year = sys.argv[1] if len(sys.argv)>1 else "2009"
    sources = CREDIT_2008 if year=="2008" else CREDIT_2009
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[r for candidates in pool.map(source,[(year+"-"+m, v) for m,v in sources.items()])
              for r in candidates]
    t=pd.DataFrame(rows)
    t.to_csv(OUT/f"credit_{year}_extraction.csv",index=False,encoding="utf-8-sig")
    for row in t.itertuples():
        print(row.month,getattr(row,"household_reported_yi",None),getattr(row,"firm_reported_yi",None),
              getattr(row,"lead",""),row.excerpt,flush=True)
