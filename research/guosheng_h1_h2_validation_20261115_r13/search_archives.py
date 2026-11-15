import urllib.request,re
for url in ['https://p.so-gov.cn/web/api/api.js','https://p.so-gov.cn/web/api/handle.js','https://p.so-gov.cn/web/api/store.js']:
    try:
        t=urllib.request.urlopen(url,timeout=15).read().decode('utf8');print('URL',url,'SIZE',len(t))
        for pat in ['ajax','search-api','/api/','siteCode','search =','search:','api.search','getSearch','search/data','searchResult']:
            for m in list(re.finditer(re.escape(pat),t))[:3]:print('MATCH',pat,t[max(0,m.start()-100):m.start()+300])
    except Exception as e:print('ERROR',str(e))
