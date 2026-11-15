import urllib.request,re,html
from pathlib import Path
O=Path(__file__).resolve().parent/'input/official'
t=(O/'c749d614f901c30b7e9ccefe.html').read_text(encoding='utf8')
for m in list(re.finditer(r'2024[-/]\d\d[-/]\d\d|来源|发布时间',t))[:10]:print('DATE',t[max(0,m.start()-70):m.start()+90])
for u in ['https://www.stats.gov.cn/sj/zxfb/index_100.html','https://www.stats.gov.cn/sj/zxfb/index_200.html','https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/','https://www.stats.gov.cn/search/s?qt=%E5%B7%A5%E4%B8%9A%E7%94%9F%E4%BA%A7%E8%80%85%E5%87%BA%E5%8E%82%E4%BB%B7%E6%A0%BC']:
 try:
  text=urllib.request.urlopen(u,timeout=12).read().decode('utf8');print('URL',u,'LEN',len(text),'PAGING',re.findall(r'createPageHTML\([^\)]*\)',text)[-2:]);print('YEARS',re.findall(r'>\s*(20\d{2}年\d{1,2}月[^<]{0,50})',text)[:5]);print('JS',re.findall(r'<script[^>]*src=[\"\x27]([^\"\x27]+)',text)[-10:])
 except Exception as e:print('FAIL',u,str(e))
