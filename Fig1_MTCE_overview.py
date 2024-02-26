
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:53:49 2022

@author: fuzhenghang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xlrd
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import pearsonr
from scipy import optimize
def f_1(x, A, B):
 return A * x + B
sns.reset_orig()
yt=[[0,0.5,1,1.5,2,2.5],[0,0.5,1,1.5,2],[0,0.5,1,1.5]]
loc=[0.132,0.235,0.48]
yle=[2.6,2.5,1.5]
co=['royalblue','limegreen','tomato']
mpl.rcParams["font.family"] = 'Arial'  
mpl.rcParams["mathtext.fontset"] = 'cm' 
mpl.rcParams["font.size"] = 12
data=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/FudanU/大三下/望道/数据结果/结果一：气候态.xlsx')
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
jtwcr=[]
jmar=[]
cmar=[]
for i in range(12):
    jtwcf.append(data[i][5])
    jmaf.append(data[i][6])
    cmaf.append(data[i][7])
for i in range(42):
    jtwc.append(data[i][1])
    jma.append(data[i][2])
    cma.append(data[i][3])
for i in range(44,86):
    jtwcr.append(data[i][3])
    jmar.append(data[i][7])
    cmar.append(data[i][11])
#print(jtwcr,jmar,cmar)


fig = plt.figure(figsize=(8,8),dpi=600)

x1 = [0.1,0.1,0.1]
yy = [0.955,0.60,0.295]
dx = 0.8
dy = [0.34,0.26,0.26]
ax = []
fuhao = [[] for i in range(3)]
label = ['(a) Monthly distribution','(b) Frequency','(c) MTCF/TTCF']
for i in range(3):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy[i]]))
x_data = [[i for i in range(1,13)],[i for i in range(1979,2021)],[i for i in range(1979,2021)]]
y=[[jtwcf,jmaf,cmaf],[jtwc,jma,cma],[jtwcr,jmar,cmar]]
for num in range(1):
# 准备数据
    
    for i in range(len(x_data[num])):  
        x_data[num][i] = x_data[num][i]-0.25
    bb=ax[num].bar(x_data[num], y[num][0],width=0.25,color='dimgray',label='JTWC',zorder=1)
    for i in range(len(x_data[num])):  
        x_data[num][i] = x_data[num][i]+0.5
    bb=ax[num].bar(x_data[num],y[num][2] ,width=0.25,color='royalblue',label='CMA',zorder=1)
    for i in range(len(x_data[num])):  
        x_data[num][i] = x_data[num][i]-0.25
    ax[num].bar(x_data[num], y[num][1],width=0.25,color='tomato',label='JMA',zorder=1)
    ax[num].set_yticks(yt[num])
    ax[num].tick_params(pad=2)
    ax[num].grid(ls="-",c='gray',alpha=0.5,linewidth=0.2)
    ax[0].text(0.3,2.66,label[num],fontweight='bold')
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
lon=[0,15.3,1.02]
fs=12
for num in range(1,3):
    x_data = [i for i in range(1979,2021)]
    y1=y[num][0]
    y2=y[num][1]
    y3=y[num][2]

    const1,p1 = pearsonr(y1, [1+i for i in range(42)])
    const2,p2 = pearsonr(y2, [1+i for i in range(42)])
    const3,p3 = pearsonr(y3, [1+i for i in range(42)])
    pv=[p1,p2,p3]
    A1, B1 = optimize.curve_fit(f_1, x_data, y1)[0]
    xn1 = np.arange(1979, 2021, 1)#30和75要对应x0的两个端点，0.01为步长
    yn1 = A1 * xn1 + B1
    ax[num].plot(xn1, yn1, '--',color='dimgray',linewidth=1.5,zorder=3)
    A2, B2 = optimize.curve_fit(f_1, x_data, y2)[0]
    xn2 = np.arange(1979, 2021, 1)#30和75要对应x0的两个端点，0.01为步长
    yn2 = A2 * xn2 + B2
    ax[num].plot(xn2, yn2, '--',color='tomato',linewidth=1.5,zorder=3)
    A3, B3 = optimize.curve_fit(f_1, x_data, y3)[0]
    xn3 = np.arange(1979, 2021, 1)#30和75要对应x0的两个端点，0.01为步长
    yn3 = A3 * xn3 + B3
    ax[num].plot(xn3, yn3, '--',color='royalblue',linewidth=1.5,zorder=3)
    AA=[A1,A2,A3]
    
    ax[num].plot(x_data,y1,'-',linewidth = 2,color='dimgray',label='JTWC',zorder=1)
    ax[num].plot(x_data,y2,'-',linewidth = 2,color='tomato',label='JMA',zorder=1)
    ax[num].plot(x_data,y3,'-',linewidth = 2,color='royalblue',label='CMA',zorder=1)
    ax[num].set_xticks([1980,1990,2000,2010,2020])
    ax[num].set_xlim(1979,2020)
    ax[num].text(1979.5,lon[num]*1.02,label[num],fontweight='bold')
    
    ax[num].text(1982,lon[num]*0.9,'trend1 = ',color='dimgray',fontsize=fs,fontweight='bold')
    ax[num].text(1996,lon[num]*0.9,'trend2 = ',color='tomato',fontsize=fs,fontweight='bold')
    ax[num].text(2009,lon[num]*0.9,'trend3 = ',color='royalblue',fontsize=fs,fontweight='bold')
    
    ax[num].text(1983.9,lon[num]*0.82,'p1 = ',color='dimgray',fontsize=fs,fontweight='bold')
    ax[num].text(1997.9,lon[num]*0.82,'p2 = ',color='tomato',fontsize=fs,fontweight='bold')
    ax[num].text(2010.9,lon[num]*0.82,'p3 = ',color='royalblue',fontsize=fs,fontweight='bold')

    ax[num].text(1986.5,lon[num]*0.9,float(format(float(AA[0])*10,'.3g')),color='dimgray',fontsize=fs,zorder=4,fontweight='bold')
    ax[num].text(2000.5,lon[num]*0.9,float(format(float(AA[1])*10,'.3g')),color='tomato',fontsize=fs,zorder=4,fontweight='bold')
    ax[num].text(2013.5,lon[num]*0.9,float(format(float(AA[2])*10,'.3g')),color='royalblue',fontsize=fs,zorder=4,fontweight='bold')
    
    ax[num].text(1986.5,lon[num]*0.82,float(format(pv[0],'.3g')),color='dimgray',fontsize=fs,zorder=4,fontweight='bold')
    ax[num].text(2000.5,lon[num]*0.82,float(format(pv[1],'.3g')),color='tomato',fontsize=fs,zorder=4,fontweight='bold')
    ax[num].text(2013.5,lon[num]*0.82,float(format(pv[2],'.3g')),color='royalblue',fontsize=fs,zorder=4,fontweight='bold')
  
    ax[num].grid(ls="-",c='gray',alpha=0.5,linewidth=1,zorder=0)
    
   

ax[0].set_xlabel('Month')
ax[2].set_xlabel('Year')
ax[0].set_ylim(0,2.6)
ax[1].set_ylim(0,15)
ax[2].set_ylim(0,1)
#ax[2].set_xticklabels([1+i for i in range(12)])
ax[0].grid(ls="-",c='gray',alpha=0.5,linewidth=0.4,zorder=0)

ax[0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
ax[1].set_yticks([0,4,8,12])
ax[2].set_yticks([0,0.2,0.4,0.6,0.8,1])
ax[1].set_xticklabels([])
ax[0].legend(frameon=False,loc='upper left')
ax[1].legend(frameon=False,bbox_to_anchor=(1,1,0,0.15),ncol=3,fontsize=12)
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/fig1.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)
