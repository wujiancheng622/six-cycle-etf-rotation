"""Build a dated middle/long loan series from public PBOC announcements."""
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen
import hashlib
import json
import re
import time
import unicodedata

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/"
START = "2013-01"
END = "2026-11-12"
ARCHIVE = HERE / "data/pbc"
JAN_2025 = "https://jingji.cctv.cn/2025/02/14/ARTI2Qy8fsueEEJNOCjrjBPu250214.shtml"
APR_2022_TABLE = "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111816422059671.pdf"
APR_2022_ASOF = ("https://www.cspengyuan.com/pengyuancmscn/credit-research/macro-research/20220609115520432/"
                 "%E5%AE%8F%E8%A7%82%E4%B8%8E%E6%94%BF%E7%AD%96%E5%91%A8%E6%8A%A5-%E5%85%AC%E5%8B%9FREITs"
                 "%E5%8F%91%E8%A1%8C%E5%BA%94%E6%8F%90%E6%8F%90%E9%80%9F%EF%BC%882022-05-16%EF%BC%89.pdf")
INDEX_PATTERN = re.compile(
    r'<a href="([^"]+)"[^>]*?title="([^"]+)"[^>]*>[^<]*</a>\s*</font>\s*'
    r'<span class="hui12">(\d{4}-\d{2}-\d{2})</span>',
    re.S,
)
NUMBERS = re.compile(r"中长期贷款\s*(增加|减少|新增|净减少|净增加)\s*"
                     r"([+-]?[\d,.]+)\s*(万亿元|亿元|万元|元)")
QUARTER = {"一": 3, "二": 6, "三": 9, "四": 12}


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        for attempt in range(3):
            try:
                with urlopen(url, timeout=16) as response:
                    data = response.read()
                assert len(data) > 1800, (url, len(data))
                path.write_bytes(data)
                break
            except OSError as e:
                if attempt == 2 or getattr(e, "code", None) in (401, 403, 404):
                    raise
                time.sleep(attempt + .4)
    return path.read_bytes()


def clean(text):
    text = re.sub(r"<[^>]+>", "", text)
    return unicodedata.normalize("NFKC", unescape(text)).replace("\u200b", "")


def period(title):
    m = re.search(r"(\d{4})年\s*(.*?)金融统计数据报告", title)
    if not m:
        return None
    year, label = m.groups()
    label = label.strip()
    num = re.fullmatch(r"(\d{1,2})月(?:份)?", label)
    if num:
        month = int(num.group(1))
    elif label in ("", "全年", "第四季度", "四季度"):
        month = 12
    elif label in ("上半年", "前两季度", "前二季度"):
        month = 6
    elif label == "前三季度":
        month = 9
    else:
        q = re.fullmatch(r"第?([一二三四])季度", label)
        if not q:
            return None
        month = QUARTER[q.group(1)]
    assert 1 <= month <= 12, title
    return f"{year}-{month:02}"


def archive_page(n):
    page = "index.html" if n == 1 else f"11871-{n}.html"
    payload = fetch(urljoin(ROOT, page), ARCHIVE / "lists" / page)
    text = payload.decode("utf-8")
    entries = INDEX_PATTERN.findall(text)
    assert len(entries) >= 5, (page, len(entries))
    result = []
    for link, title, release in entries:
        title = clean(title)
        month = period(title)
        if month and START <= month and release <= END:
            result.append(dict(month=month, released=release, title=title, url=urljoin(ROOT, link)))
    return result


def get_catalog():
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for page in pool.map(archive_page, range(1, 34)) for r in page]
    rows.append(dict(month="2025-01", released="2025-02-14",
                     title="2025年1月金融统计数据报告（央视网同日刊发央行原文）", url=JAN_2025))
    return pd.DataFrame(rows).drop_duplicates().sort_values(["month", "released", "url"])


def amount(match):
    direction, value, unit = match
    scale = {"万亿元": 1e4, "亿元": 1., "万元": 1e-4, "元": 1e-8}
    numeric = float(value.replace(",", ""))
    return abs(numeric) * scale[unit] * (-1 if "减" in direction or numeric < 0 else 1)


def announcement(entry):
    month = entry["month"]
    url = entry["url"]
    path = ARCHIVE / "releases" / f"{month}.html"
    raw = fetch(url, path)
    html = raw.decode("utf-8", errors="strict")
    text = clean(html)
    text = re.sub(r"\s+", "", text)
    if month == "2022-04":
        assert "中长期贷款增加2652亿元" in text and "住房贷款减少605亿元" in text
        # The balance difference is corroborated in the May 16, 2022 credit report.
        # Delay both legs to that day since the May 13 press release omitted one leg.
        return dict(**(entry | {"released": "2022-05-16"}),
                    household_reported_yi=round(548157.06-548470.78, 2), firm_reported_yi=2652.,
                    basis="monthly", hint="当月", prefix="央行住户中长期贷款余额四月减三月",
                    extracted="原公告企业2652亿元；住户余额548157.06亿元、548470.78亿元",
                    supplementary_url=APR_2022_TABLE, available_by_url=APR_2022_ASOF,
                    sha256=hashlib.sha256(raw).hexdigest())
    blocks = list(re.finditer(r"(?:分部门看|分部门计|分部门)", text))
    for block in blocks:
        piece = text[block.start():block.start() + 450]
        matches = NUMBERS.findall(piece)
        first_loan = NUMBERS.search(piece)
        if len(matches) == 2 and "住户" in piece[:first_loan.start()] and (
            "非金融企业" in piece or "事业单位" in piece or "事)业单位" in piece
        ):
            lead = text[max(0, block.start() - 110):block.start()]
            hints = list(re.finditer(
                r"(当月|本月|\d{1,2}月份?|前[一二两三四五六七八九十0-9]+个?月|"
                r"第?[一二三四]季度|前三季度|上半年|全年|\d{4}年|年初以来|前[一二两]个月)"
                r"[,]?人民币贷款(?:增加|减少|新增)", lead))
            if not hints:
                hint = ""
            else:
                hint = hints[-1][1]
            basis = "monthly" if re.fullmatch(r"当月|本月|\d{1,2}月份?", hint) else (
                "year_to_date" if hint else "unknown"
            )
            return dict(**entry, household_reported_yi=amount(matches[0]),
                        firm_reported_yi=amount(matches[1]), basis=basis, hint=hint,
                        prefix=lead, extracted=piece[:220],
                        sha256=hashlib.sha256(raw).hexdigest())
    return dict(**entry, basis="unparsed", extracted="", sha256=hashlib.sha256(raw).hexdigest())


def collect():
    cat = get_catalog()
    cat.to_csv(ARCHIVE / "source_index.csv", index=False, encoding="utf-8-sig")
    duplicate = cat[cat.duplicated("month", keep=False)]
    duplicate.to_csv(ARCHIVE / "duplicate_links.csv", index=False, encoding="utf-8-sig")
    unique = cat.drop_duplicates("month", keep="first")
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(announcement, unique.to_dict("records")), 1):
            results.append(row)
            if i % 24 == 0:
                print("PBOC", i, "/", len(unique), row["month"], flush=True)
    t = pd.DataFrame(results)
    t.to_csv(ARCHIVE / "raw_loans.csv", index=False, encoding="utf-8-sig")
    expected = pd.period_range(START, t.month.max(), freq="M").astype(str)
    report = dict(rows=len(t), first=t.month.min(), last=t.month.max(),
                  missing=sorted(set(expected) - set(t.month)),
                  basis=t.basis.value_counts().to_dict())
    print(json.dumps(report, ensure_ascii=True))
    assert not report["missing"] and set(t.basis) <= {"monthly", "year_to_date"}, report
    return t


def credit_series(reports):
    reports = reports.sort_values("month").set_index("month")
    months = pd.PeriodIndex(reports.index, freq="M")
    assert months.equals(pd.period_range(months[0], months[-1], freq="M"))
    out = reports[["released", "url", "basis", "household_reported_yi", "firm_reported_yi"]].copy()
    ytd = [0., 0.]
    for month, row in reports.iterrows():
        if month.endswith("-01"):
            ytd = [0., 0.]
        fields = [row.household_reported_yi, row.firm_reported_yi]
        if row.basis == "monthly":
            new = fields
            ytd = [a+b for a, b in zip(ytd, new)]
        else:
            assert row.basis == "year_to_date"
            new = [a-b for a, b in zip(fields, ytd)]
            ytd = fields
        out.loc[month, ["household_new_yi", "firm_new_yi"]] = new
        out.loc[month, ["household_ytd_yi", "firm_ytd_yi"]] = ytd
    out["middle_long_new_yi"] = out.household_new_yi + out.firm_new_yi
    out["ttm_yi"] = out.middle_long_new_yi.rolling(12).sum()
    out["pulse"] = out.ttm_yi.pct_change(12, fill_method=None)
    out["credit_change"] = out.pulse.diff(3)
    assert pd.to_datetime(out.released).is_monotonic_increasing
    assert (out.middle_long_new_yi.abs() < 90000).all()
    for month, expected in (("2025-02", -.29), ("2025-05", -.24)):
        assert abs(out.at[month, "pulse"] - expected) < .02, (month, out.at[month, "pulse"])
    return out


if __name__ == "__main__":
    loans = credit_series(collect())
    loans.to_csv(ARCHIVE / "credit_monthly.csv", encoding="utf-8-sig")
    print(loans.loc[["2025-02", "2025-05"], ["released", "pulse", "credit_change"]].to_string())
