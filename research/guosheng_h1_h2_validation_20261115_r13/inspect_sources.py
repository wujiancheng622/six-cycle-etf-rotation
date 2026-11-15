from pathlib import Path
import json,re,html
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parent;O=H/'input/official'
rec=json.loads((O/'parsed_releases.json').read_text(encoding='utf8'))
for kind in ['ppi','pmi']:
    z=[r for r in rec if r['kind']==kind and r['status']!='parsed'];print('BAD',kind,len(z))
    for row in z[:2]:
        print(row)
        if 'file' not in row:continue
        t=(O/row['file']).read_text(encoding='utf8');print('META',re.findall('<meta[^>]+>',t,re.I)[:25]);print('VISIBLE_DATES',[t[max(0,m.start()-100):m.start()+150] for m in list(re.finditer(r'2024[-/]\d\d[-/]\d\d|2024年\d{1,2}月\d{1,2}日|文章来源|发布时间|来源：',t))[:12]])
        if kind=='ppi':
            for m in list(re.finditer('加工',t))[-3:]:print('RAW',t[m.start()-150:m.start()+1000])
        else:
            k=t.find('新订单');print(t[k-100:k+500])
f=pd.read_csv(R/'guosheng_original_reconstruction_20261114/results/pmi_dates_used.csv')
print('ORIGINAL_PMI',f.shape,list(f.columns));print(f.head(4).to_json(orient='records',force_ascii=False));print(f[['month','available','official_date_source']].iloc[[20,30,50,100,160]].to_json(orient='records',force_ascii=False))
