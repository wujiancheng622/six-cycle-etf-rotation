from urllib.request import Request,urlopen
from urllib.parse import urlencode
import json
p={'siteCode':'bm36000002','qt':'2014年 工业生产者出厂价格','page':1,'pageSize':100,'tab':''}
try:
    req=Request('https://api.so-gov.cn/query/s',data=urlencode(p).encode(),headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/x-www-form-urlencoded','Origin':'https://www.stats.gov.cn','Referer':'https://www.stats.gov.cn/search/s'})
    raw=urlopen(req,timeout=20).read().decode('utf8');o=json.loads(raw);print(raw[:10000])
except Exception as e:print(type(e).__name__,str(e))
