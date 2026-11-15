"""Standalone figures from the saved research table."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                     'figure.dpi':160,'savefig.facecolor':'white','font.size':11})
with (OUT/'comparison.csv').open(encoding='utf-8-sig',newline='') as f:table=list(csv.DictReader(f))

def rows(period='full'):
    return {x['name']:x for x in table if x['period']==period and x['mode']=='amount' and x['account_start']=='2014-01-02'
            and x['role'] in ('main','factorial') and x['bp']=='9'}

base=rows()
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,key,title,cmap in zip(axes,['cagr','mdd'],['扣费后年化收益（%）','最大回撤（%，越低越好）'],['Blues','Oranges']):
    matrix=np.array([[100*float(base[b+r][key]) for r in 'MWD'] for b in 'MWD'])
    ax.imshow(matrix,cmap=cmap,vmin=matrix.min()-1,vmax=matrix.max()+1)
    for y in range(3):
        for x in range(3):ax.text(x,y,f'{matrix[y,x]:.2f}',ha='center',va='center',fontsize=16,
                                 color='white' if matrix[y,x]>(matrix.min()+matrix.max())/2 else '#17212b')
    ax.set_xticks(range(3),['月频','周频','日频'])
    ax.set_yticks(range(3),['月频','周频','日频'])
    ax.set_xlabel('篮子内重新配权频率')
    ax.set_ylabel('检查并接受新篮子的频率')
    ax.set_title(title)
fig.suptitle('换篮与配权的作用不同｜2014-01-02—2026-11-12\n含早期代理，50万元，9bp滑点另扣佣金',fontsize=14)
fig.savefig(OUT/'two_clocks.png')
plt.close(fig)

periods=['2014-2016','2017-2019','2020-2022','2023-2025']
names=['WW','DD','DM','DD_band']
labels=['纯周频','纯日频','月初＋换篮加调','日频＋5%相对缓冲']
fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
for j,(name,label) in enumerate(zip(names,labels)):
    vals=[100*(float(rows(p)[name]['cagr'])-float(rows(p)['MM']['cagr'])) for p in periods]
    ax.bar(np.arange(4)+(j-1.5)*.18,vals,width=.17,label=label)
ax.axhline(0,color='#455a64',lw=.8)
ax.set_xticks(range(4),periods)
ax.set_ylabel('相对纯月频的净年化差（百分点）')
ax.set_title('分段结果未显示持续的收益优势\n同一2014起账户，非重叠三年分段')
ax.legend(loc='upper right',fontsize=9)
ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'period_differences.png')
plt.close(fig)
print('Saved two standalone figures; matplotlib',matplotlib.__version__)
