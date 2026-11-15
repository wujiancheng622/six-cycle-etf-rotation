import urllib.request,re
u='https://www.stats.gov.cn/sj/zxfb/'
t=urllib.request.urlopen(u,timeout=25).read().decode('utf8')
print(t[:800])
for m in re.finditer('createPage|index_1|pageCount|PubDate|createPagination',t):print('PAGING',t[max(0,m.start()-160):m.start()+400])
for a,b in re.findall(r'<a[^>]*href=[\"\x27]([^\"\x27]+)[\"\x27][^>]*>(.*?)</a>',t,re.S):
    if '工业生产者' in b or '采购经理' in b:print('LINK',a,b[:300])
