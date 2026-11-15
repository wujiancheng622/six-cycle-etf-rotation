"""Package unchanged downloaded originals and local audit evidence."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DOWNLOADS=Path.home()/"Downloads"
PRIOR=ROOT/"guosheng_pro_handoff_20261113"
ORIGINAL=ROOT/"guosheng_six_regime_20261112"
LONG=ROOT/"guosheng_six_regime_long_20261112"

PDFS = {
    "2024":("2024_国盛_中国经济六周期模型与多资产策略应用.pdf",34,"https://www.fxbaogao.com/detail/4647860"),
    "2025":("2025_国盛_六周期框架下的多资产ETF配置.pdf",10,"https://www.fxbaogao.com/detail/5137956"),
    "2026":("2026_国盛_基于市场隐含状态与宏观偏离度的配置模型.pdf",19,"https://www.fxbaogao.com/detail/5399157"),
}

def main():
    files = {}
    for y,(name,pages,url) in PDFS.items():
        src,= [p for p in DOWNLOADS.glob(y+"*.pdf")
               if "发现报告" in p.name and ("六周期" in p.name or "宏观偏离度" in p.name)]
        data=src.read_bytes()
        assert data.startswith(b"%PDF-")
        files["originals/"+name]=(src,dict(pages=pages,url=url,sha256=sha256(data).hexdigest()))
    archive=HERE/"国盛原文审计与研报原件.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:
        for file in ("国盛六周期原文对照审核.md","给网页端Pro的提示词.txt"):
            z.write(HERE/file,file)
        for file,(src,_) in files.items():
            z.write(src,file)
        z.writestr("originals/manifest.json",
                   json.dumps({k:v[1] for k,v in files.items()},ensure_ascii=False,indent=2))
        for f in sorted((HERE/"evidence").glob("*.csv")):
            z.write(f,"evidence/"+f.name)
        z.write(HERE/"verify_money_group.py","evidence/verify_money_group.py")
        for f in ("regimes.py","run_study.py","pbc_loans.py","protocol.json"):
            z.write(ORIGINAL/f,"implementation/"+f)
        for f in ("early_inputs.py","data/pmi_as_reported.csv","data/credit_monthly.csv"):
            z.write(LONG/f,"implementation/"+f)
        for f in ("国盛六周期综合分析.md","Pro复核证据包.zip"):
            z.write(PRIOR/f,"prior/"+f)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        for name,(_,m) in files.items():
            assert sha256(z.read(name)).hexdigest()==m["sha256"]
    print(archive,archive.stat().st_size)

if __name__=="__main__":
    main()
