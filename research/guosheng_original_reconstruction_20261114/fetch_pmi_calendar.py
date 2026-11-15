"""Find actual official PMI publication dates in the NBS news and analysis archive."""

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime
from html import unescape
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
BASE_URL = "https://www.stats.gov.cn/sj/"
LINK = re.compile(r'<a\b[^>]*href=[\'\"]([^\'\"]+)[\'\"][^>]*>(.*?)</a>', re.S | re.I)
PERIOD = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月(?:份)?")
STAMP = re.compile(r"\b(20\d{2}/\d{2}/\d{2})\s+\d{2}:\d{2}\b")
ANNOUNCED = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^。]{0,90}发布了中国(?:制造业和非制造业采购经理指数|采购经理指数)")


def get(url):
    for _ in range(3):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=7) as response:
                return response.read().decode("utf-8-sig")
        except (HTTPError, URLError, ConnectionError, TimeoutError, UnicodeDecodeError):
            continue
    return ""


def month_from_title(title):
    period = PERIOD.search(title)
    if period is None:
        return None
    year, month = map(int, period.groups())
    if not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}"


def list_articles(section_page):
    section, page = section_page
    root = urljoin(BASE_URL, f"{section}/")
    url = urljoin(root, f"index_{page}.html") if page else root
    html = get(url)
    articles = {}
    for href, text in LINK.findall(html):
        title = unescape(re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", text)))
        combined = section == "sjjd" and "解读" in title and (
            "采购经理指数" in title or "制造业和非制造业PMI" in title
        )
        release = section == "zxfb" and "中国采购经理指数运行情况" in title and "解读" not in title
        if combined or release:
            month = month_from_title(title)
            if month and "2012-01" <= month < "2025-09":
                articles[urljoin(root, href)] = (month, title, url)
    return articles


def dated_article(item):
    url, (month, title, archive_page) = item
    html = get(url)
    explicit = ANNOUNCED.search(re.sub(r"\s+", "", re.sub(r"<[^>]+>", " ", html)))
    if explicit:
        published = date(*map(int, explicit.groups()))
        how = "文章明确列出两项PMI的共同发布日期"
    elif "中国采购经理指数运行情况" in title:
        stamp = STAMP.search(html)
        if stamp is None or "非制造业商务活动指数" not in html or "制造业采购经理指数" not in html:
            return None
        published = datetime.strptime(stamp[1], "%Y/%m/%d").date()
        how = "含两项PMI的统计局原始发布时戳"
    else:
        return None
    year, number = map(int, month.split("-"))
    if not (0 <= 12 * (published.year - year) + published.month - number <= 1):
        return None
    return dict(month=month, official_released=published.isoformat(), source=url,
                evidence=how, title=title, archive_page=archive_page)


def main():
    pages = [("zxfb", i) for i in range(78)] + [("sjjd", i) for i in range(141)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        articles = {}
        for found in pool.map(list_articles, pages):
            articles.update(found)
        print(f"Found {len(articles)} potential PMI announcements", flush=True)
        results = [row for row in pool.map(dated_article, articles.items()) if row is not None]
    if not results:
        raise RuntimeError("The official archive returned no confirmed releases")
    fields = ("month", "official_released", "source", "evidence", "title", "archive_page")
    results.sort(key=lambda row: (row["month"], row["official_released"], row["source"]))
    with (HERE / "nbs_pmi_archive.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"{len(results)} articles covering {len({row['month'] for row in results})} months", flush=True)


if __name__ == "__main__":
    main()
