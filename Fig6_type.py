
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:15:29 2022

@author: fuzhenghang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xlrd
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
sns.reset_orig()
from scipy import optimize
def f_1(x, A, B):
 return A * x + B
yt=[[0,0.5,1,1.5,2,2.5],[0,0.5,1,1.5,2],[0,0.5,1,1.5]]

mpl.rcParams["font.family"] = 'Arial'  
mpl.rcParams["mathtext.fontset"] = 'cm' 
mpl.rcParams["font.size"] = 12
data=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/month.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data.append(table.row_values(i))
data = [data[i] for i in range(0,len(data))]
jtwcf=[]
jmaf=[]
cmaf=[]
jtwc=[]
jma=[]
cma=[]

for i in range(12):
    jtwcf.append(data[i][1])
    jmaf.append(data[i][2])
    cmaf.append(data[i][3])
for i in range(12):
    jtwc.append(data[i][4])
    jma.append(data[i][5])
    cma.append(data[i][6])
print(jtwc,jma,cma)

fig = plt.figure(figsize=(16,8),dpi=600)

x1 = [0.05,0.05,0.5,0.5]
yy = [0.95,0.56,0.95,0.56]
dx = 0.4
dy = [0.34,0.34,0.34,0.34]
ax = []

label = ['(a) Monthly distribution','(b) Monthly ratio variation','(c) Frequency variation','(d) Ratio variation']
for i in range(4):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy[i]]))
x_data = [[i for i in range(1,13)],[i for i in range(1,13)]]
y=[[jtwcf,jmaf,cmaf],[jtwc,jma,cma]]
for num in range(1):
# 准备数据
    
    for i in range(len(x_data[num])):  
        x_data[num][i] = x_data[num][i]-0.25
    bb=ax[num].bar(x_data[num], y[num][0],width=0.25,color='#b5ffb9',label='Type EI',zorder=1)
    for i in range(len(x_data[num])):  
        x_data[num][i] = x_data[num][i]+0.25
    ax[num].bar(x_data[num], y[num][1],width=0.25,color='#f9bc86',label='Type WI-N',zorder=1)
    for i in range(len(x_data[num])):  
        x_data[num][i] = x_data[num][i]+0.25
    bb=ax[num].bar(x_data[num],y[num][2] ,width=0.25,color='#a3acff',label='Type WI-O',zorder=1)
    ax[num].set_yticks(yt[num])
    ax[num].tick_params(pad=2)
    ax[num].grid(ls="-",c='gray',alpha=0.5,linewidth=0.15)
    ax[0].text(0.1,40.8-num*45.7,label[num])
    #const1,p1 = pearsonr(y_data, basin[num])
    #const2,p2 = pearsonr(y_data1, basin[num])
    #ax[num].text(3.5,yle[num]*0.83,float(format(const1,'.3g')),c='royalblue')
    #ax[num].text(3.5,yle[num]*0.72,float(format(const2,'.3g')),c='tomato')
# 设置x轴标签名
#plt.ylabel("Level, m")
#plt.xlabel("Time")
#for a,b in zip(x_data,y_data):   #柱子上的数字显示
#    plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=10);
# 设置y轴标签名
xd=[1+i for i in range(12)]
ax[0].text(0.1,40.8-1*45.7,label[1])
raw_data = {'greenBars': jtwc, 'orangeBars': jma,'blueBars': cma}
ax[1].grid(ls="-",c='gray',alpha=0.5,linewidth=0.15)
greenBars=jtwc
orangeBars=jma
blueBars=cma
barWidth = 0.75
names = ('1','2','3','4','5','6','7','8','9','10','11','12')
ax[1].bar(xd, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
ax[1].bar(xd, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
ax[1].bar(xd, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)
for i in range(4,12):
    ax[1].text(xd[i],jtwc[i]/2,"%.1f" %jtwc[i],ha='center', va='center')
    ax[1].text(xd[i],jtwc[i]+jma[i]/2,"%.1f" %jma[i],ha='center', va='center')
    ax[1].text(xd[i],jtwc[i]+jma[i]+cma[i]/2,"%.1f" %cma[i],ha='center', va='center')

ax[1].text(xd[0],jma[0]/2,"%.1f" %jma[0],ha='center', va='center')
ax[1].text(xd[2],jma[2]/2,"%.1f" %jma[2],ha='center', va='center')
ax[1].text(xd[3],jma[3]/2,"%.1f" %jma[3],ha='center', va='center')
ax[1].text(xd[2],jma[2]+cma[2]/2,"%.1f" %cma[2],ha='center', va='center')
ax[1].text(xd[3],jma[3]+cma[3]/2,"%.1f" %cma[3],ha='center', va='center')
ax[1].set_xlabel('Month')
ax[0].set_ylabel('Frequency',labelpad=12)
ax[1].set_ylabel('Percentage, %')
ax[0].set_ylim(0,40)

ax[1].set_ylim(0,100)
#ax[2].set_xticklabels([1+i for i in range(12)])


ax[1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
ax[0].set_yticks([0,10,20,30,40])
ax[1].set_yticks([0,20,40,60,80,100])
ax[0].set_xticklabels([])
ax[0].legend(frameon=False,loc='upper left')
jtwcf=[]
jmaf=[]
cmaf=[]
jtwc=[]
jma=[]
cma=[]

for i in range(13,55):
    jtwcf.append(data[i][1])
    jmaf.append(data[i][2])
    cmaf.append(data[i][3])
for i in range(13,55):
    jtwc.append(data[i][4])
    jma.append(data[i][5])
    cma.append(data[i][6])
lon=[8.32,100]
x_data = [[i for i in range(1979,2021)],[i for i in range(1979,2021)]]
y=[[jtwcf,jmaf,cmaf],[jtwc,jma,cma]]

for num in range(2,3):
# 准备数据
    const1,p1 = pearsonr(y[0][0], [1+i for i in range(42)])
    const2,p2 = pearsonr(y[0][1], [1+i for i in range(42)])
    const3,p3 = pearsonr(y[0][2], [1+i for i in range(42)])
    pv=[p1,p2,p3]
    A1, B1 = optimize.curve_fit(f_1, x_data[num-2], y[0][0])[0]
    xn1 = np.arange(1979, 2021, 1)#30和75要对应x0的两个端点，0.01为步长
    yn1 = A1 * xn1 + B1
    ax[num].plot(xn1, yn1, '--',color='g',linewidth=1.5,zorder=3)
    A2, B2 = optimize.curve_fit(f_1, x_data[num-2], y[0][1])[0]
    xn2 = np.arange(1979, 2021, 1)#30和75要对应x0的两个端点，0.01为步长
    yn2 = A2 * xn2 + B2
    ax[num].plot(xn2, yn2, '--',color='#f9bc86',linewidth=1.5,zorder=3)
    A3, B3 = optimize.curve_fit(f_1, x_data[num-2], y[0][2])[0]
    xn3 = np.arange(1979, 2021, 1)#30和75要对应x0的两个端点，0.01为步长
    yn3 = A3 * xn3 + B3
    ax[num].plot(xn3, yn3, '--',color='#a3acff',linewidth=1.5,zorder=3)
    AA=[A1,A2,A3]

    ax[num].plot(x_data[num-2],y[0][0],'-',linewidth = 2,color='g',label='Type EI',zorder=1)
    ax[num].plot(x_data[num-2],y[0][1],'-',linewidth = 2,color='#f9bc86',label='Type WI-N',zorder=1)
    ax[num].plot(x_data[num-2],y[0][2],'-',linewidth = 2,color='#a3acff',label='Type WI-O',zorder=1)
    ax[num].set_xticks([1980,1990,2000,2010,2020])
    ax[num].set_xlim(1979,2020)
    ax[num].text(1979.2,lon[num-2]*1.03,label[num])
    fs=13
    ax[num].text(1992,lon[num-2]*0.92,'trend1 = ',color='g',fontsize=fs,fontweight='bold')
    ax[num].text(2001.5,lon[num-2]*0.92,'trend2 = ',color='#f9bc86',fontsize=fs,fontweight='bold')
    ax[num].text(2012.2,lon[num-2]*0.92,'trend3 = ',color='#a3acff',fontsize=fs,fontweight='bold')
    
    ax[num].text(1994.1,lon[num-2]*0.84,'p1 = ',color='g',fontsize=fs,fontweight='bold')
    ax[num].text(2003.7,lon[num-2]*0.84,'p2 = ',color='#f9bc86',fontsize=fs,fontweight='bold')
    ax[num].text(2014.4,lon[num-2]*0.84,'p3 = ',color='#a3acff',fontsize=fs,fontweight='bold')

    ax[num].text(1997.5,lon[num-2]*0.92,float(format(float(AA[0])*10,'.3g')),color='g',fontsize=fs,zorder=4,fontweight='bold')
    ax[num].text(2006.3,lon[num-2]*0.92,float(format(float(AA[1])*10,'.3g')),color='#f9bc86',fontsize=fs,zorder=4,fontweight='bold')
    ax[num].text(2017,lon[num-2]*0.92,float(format(float(AA[2])*10,'.3g')),color='#a3acff',fontsize=fs,zorder=4,fontweight='bold')
    
    ax[num].text(1997.5,lon[num-2]*0.84,float(format(pv[0],'.3g')),color='g',fontsize=fs,zorder=4,fontweight='bold')
    ax[num].text(2006.3,lon[num-2]*0.84,float(format(pv[1],'.3g')),color='#f9bc86',fontsize=fs,zorder=4,fontweight='bold')
    ax[num].text(2017,lon[num-2]*0.84,float(format(pv[2],'.3g')),color='#a3acff',fontsize=fs,zorder=4,fontweight='bold')
  
    ax[num].grid(ls="-",c='gray',alpha=0.5,linewidth=0.2)
# 设置y轴标签名
xd=[1979+i for i in range(42)]
ax[2].text(1979.2,-1.5,label[3])
raw_data = {'greenBars': jtwc, 'orangeBars': jma,'blueBars': cma}
ax[3].grid(ls="-",c='gray',alpha=0.5,linewidth=0.15)
greenBars=jtwc
orangeBars=jma
blueBars=cma

barWidth = 1
#names = ('1','2','3','4','5','6','7','8','9','10','11','12')
ax[3].bar(xd, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
ax[3].bar(xd, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
ax[3].bar(xd, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)

for i in range(42):
    if jtwc[i]!=0:
        ax[3].text(xd[i],jtwc[i]/2,"%.1f" %jtwc[i],ha='center', va='center',rotation=90,fontsize=8)
    if jma[i]!=0:
        ax[3].text(xd[i],jtwc[i]+jma[i]/2,"%.1f" %jma[i],ha='center', va='center',rotation=90,fontsize=8)
    if cma[i]!=0:
        ax[3].text(xd[i],jtwc[i]+jma[i]+cma[i]/2,"%.1f" %cma[i],ha='center', va='center',rotation=90,fontsize=8)



ax[3].set_xlabel('Year')
ax[2].set_ylabel('Frequency',labelpad=12)
ax[3].set_ylabel('Percentage, %')


ax[3].set_ylim(0,100)
ax[3].set_xlim(1979,2020)
#ax[2].set_xticklabels([1+i for i in range(12)])


ax[3].set_xticks([1980,1990,2000,2010,2020])

ax[3].set_yticks([0,20,40,60,80,100])
ax[2].set_xticklabels([])
ax[2].legend(frameon=False,loc='upper left',bbox_to_anchor=(0, 1.03))

plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/fig6.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)

