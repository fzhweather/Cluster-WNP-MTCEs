
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:45:15 2022

@author: fuzhenghang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import seaborn as sns
import matplotlib as mpl



mpl.rcParams["font.family"] = 'Arial'  
mpl.rcParams["mathtext.fontset"] = 'cm' 
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["font.size"] = 12
sns.reset_orig()
data=[]
#ori = pd.read_excel('D:/TC/dataTC2.xlsx')
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/AMO_IPO.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data.append(table.row_values(i))
data1 = [data[i][j] for i in range(89,165) for j in range(1,13)]
data2 = [data[i][j] for i in range(89,165) for j in range(15,27)]
print(data1[-12:])

dataamo=[[] for i in range(76)]
amo=[]
dataipo=[[] for i in range(76)]
ipo=[]

amos = np.std(data1,ddof=1)
ipos = np.std(data2,ddof=1)
amoa = sum(data1)/len(data1)
ipoa = sum(data2)/len(data2)

for i in range(76):
    for j in range(12):
        dataamo[i].append((data1[12*i+j]))
        dataipo[i].append((data2[12*i+j]))
print(dataamo[0:12])
for i in range(76):
    amo.append(sum(dataamo[i])/12)
    ipo.append(sum(dataipo[i])/12)
samo=[]
sipo=[]
for i in range(68):
    samo.append(sum(amo[i:i+9])/9)
    sipo.append(sum(ipo[i:i+9])/9)
sipo = (np.array(sipo)-np.mean(sipo))/np.std(sipo)
samo = (np.array(samo)-np.mean(samo))/np.std(samo)
print(list(sipo))
print(list(samo))
mtce=[-0.610785504,-0.439804041,-0.397057441,-0.525292161,-0.568037068,-0.525292161,-0.439802348,-0.397057441,-0.482547254,
-0.397057441,-0.482547254,-0.012353278,0.201371257,0.372350884,0.671565233,0.757055047,0.671565233,0.71431014,0.842544861,
1.013524489,0.543330512,0.415095791,0.286861071,0.115881443,-0.012353278,0.15862635,0.244116164,0.073136536,-0.140587999,
0.115881443,-0.055098185,0.030391629,-0.055098185,0.030391629,0.073136536,-0.012353278,-0.097843092,0.201371257,0.115881443,
0.415095791,0.286861071,0.543330512,0.543330512,0.586075419,0.757055047,0.671565233,0.372350884,0.500585605,0.286861071,
0.329605977,0.030391629,0.201371257,0.030391629,-0.226077813,-0.26882272,-0.140587999,-0.311567627,-0.525292161,-0.439802348,
-0.354312534,-0.525292161,-0.610781975,-0.482547254,-0.311567627,-0.354312534,-0.26882272,-0.055098185,-0.055098185]#1949-2016
mt = [6.77777777777778,	7.44444444444444,	7.44444444444444,	7.88888888888889,	7.55555555555556,	8,	8,	8.55555555555556,	9.44444444444444,	9.77777777777778,	9.88888888888889,	10,	9.33333333333333,	9.44444444444444,	9.66666666666667,	9.88888888888889,	8.77777777777778,	8.22222222222222,	7.88888888888889,	7.44444444444444,	7.22222222222222,	8.11111111111111,	8.22222222222222,	7.77777777777778,	7.44444444444444,	8,	7.66666666666667,	8,	7.77777777777778,	8.33333333333333,	8,	7.88888888888889,	7.77777777777778,	8.55555555555556,	8.33333333333333,	9.11111111111111,	8.55555555555556,	9,	8.22222222222222,	8.22222222222222,	8.33333333333333,	7.55555555555556,	6.66666666666667,	7,	6.22222222222222,	6.33333333333333,	5.77777777777778,	6.77777777777778,	6.77777777777778,	6.55555555555556,	7.11111111111111,	7.11111111111111,	6.55555555555556,	6.11111111111111,	6.33333333333333,	6.55555555555556,	6,	5.77777777777778,	6.11111111111111,	6.22222222222222,	6.22222222222222,	6.44444444444444,	7.11111111111111,	7.11111111111111]
mt = (np.array(mt)-np.mean(mt))/np.std(mt)
mtce = [0,0,0,0]
for i in range(64):
  mtce.append(mt[i])  
print(mtce)
"""
mtce_na = [0.130657468,0.304867426,0.595217356,0.479077384,0.479077384,0.24679744,0.130657468,0.072587482,0.014517496,
           -0.101622475,-0.217762447,-0.450042391,-0.333902419,-0.275832433,-0.275832433,-0.391972405,-0.217762447,-0.217762447,
           -0.217762447,-0.275832433,-0.275832433,-0.391972405,-0.450042391,-0.508112377,-0.566182363,-0.624252349,-0.624252349,
           -0.624252349,-0.624252349,-0.682322335,-0.740392321,-0.682322335,-0.624252349,-0.624252349,-0.682322335,-0.740392321,
           -0.508112377,-0.275832433,-0.217762447,-0.101622475,-0.217762447,-0.391972405,-0.159692461,-0.15969246,	-0.217762447,
           -0.333902419,-0.391972405,-0.101622475,-0.043552489,0.072587482,0.304867426,0.362937412,0.595217356,0.769427314,
           0.769427314,0.653287342,0.362937412,0.421007398,0.595217356,0.8274973,0.711357328,0.421007398,0.24679744,
           0.304867426,0.421007398,0.595217356,0.653287342,0.711357328]"""
import statsmodels.api as sm
file = r'/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/regress.xlsx'
data = pd.read_excel(file)
data.columns = ['y', 'x1', 'x2']
x = sm.add_constant(data.iloc[:,1:]) #生成自变量
y = data['y'] #生成因变量
model = sm.OLS(y, x) #生成模型
result = model.fit() #模型拟合
fitted = [0,0,0,0]
for i in range(64):
  fitted.append(-0.0034+0.0691*sipo[i+4]-0.4759*samo[i+4])  
print(result.summary()) #模型描述

fig = plt.figure(figsize=(8,2.5),dpi=600)
ax=fig.add_axes([0,0,1,1])
x_data = [i for i in range(1949,2017)]

ax.plot(x_data,mtce,'-',linewidth = 2.5,color='royalblue',label='WNP-MTCEs',zorder=1)
#ax.plot(x_data,fitted,'-',linewidth = 2.5,color='tomato',label='NA-MTCEs',zorder=1)
ax.plot(x_data,samo,'-',linewidth = 2.5,color='limegreen',label='AMO',zorder=1)
ax.plot(x_data,sipo,'-',linewidth = 2.5,color='magenta',label='IPO',zorder=1)
ax.set_ylim(-2.1,2.1)
ax.set_xlim(1953,2016)
ax.axhline(y=0, linestyle='-',linewidth = 0.7,color='k',alpha=1,zorder=1)
ax.legend(frameon=False,fontsize=10,loc='upper right',ncol=2)
ax.fill_between([1949,1960], 3,-3, color='lightcyan',zorder=0)
ax.fill_between([1978,1996], 3,-3, color='papayawhip',zorder=0)
ax.fill_between([2002,2016], 3,-3, color='lightcyan',zorder=0)
ax.text(1956.5,-1.9,'AMO+&IPO-',ha='center')
ax.text(1987.5,-1.9,'AMO-&IPO+',ha='center')
ax.text(2009,-1.9,'AMO+&IPO-',ha='center')
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/fig2.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)
