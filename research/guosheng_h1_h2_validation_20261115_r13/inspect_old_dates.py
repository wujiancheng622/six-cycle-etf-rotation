from pathlib import Path
import re,html,json
O=Path(__file__).resolve().parent/'input/official'
for name in ['dd6d71fb37447f399c976415.html','449c7501dd6f5f80084887be.html']:
    t=(O/name).read_text(encoding='utf8');flat=html.unescape(re.sub('<[^>]+>',' ',t));print('FILE',name)
    for pat in ['发布时间','发布日期','生成日期','2020年','2020','成文日期']:
        print('PATTERN',pat)
        for m in list(re.finditer(pat,flat))[:8]:print(re.sub(r'\s+',' ',flat[max(0,m.start()-70):m.start()+150]))
