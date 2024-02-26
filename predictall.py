#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:05:38 2024

@author: fuzhenghang
"""

# In[0]
import matplotlib.ticker as mticker
import xlrd
import xarray as xr  
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import numpy as np
import netCDF4 as nc
import dateutil.parser  #引入这个库来将日期字符串转成统一的datatime时间格式
import matplotlib as mpl
from numpy import nan
import scipy
from scipy.stats import pearsonr
from cartopy.util import add_cyclic_point
import cmaps
import seaborn as sns
sns.reset_orig()


mpl.rcParams["font.family"] = 'Arial'  
mpl.rcParams["mathtext.fontset"] = 'cm' 
mpl.rcParams["font.size"] = 9
plt.rcParams['hatch.color'] = 'k' 

d1 = xr.open_dataset(r'/Users/fuzhenghang/Documents/ERA5/sst/ersst5.190101_202302.nc',use_cftime=True)
lon = d1.variables['lon'][:]
lat = d1.variables['lat'][:]
time = d1['time'][:]
sst = d1['sst'][(time.dt.year>=1901)&(time.dt.year<=2021)][:,0,:,:]
sst = np.array(sst)

for i in range(181):
    for j in range(360):
        for m in range(12):
            if not (np.isnan(sst[m::12,i,j]).any()):
                sst[m::12,i,j]=scipy.signal.detrend(sst[m::12,i,j], axis=0, type='linear', bp=0, overwrite_data=True)
# In[1]
nino = np.zeros((1452))

for i in range(1452):
    co2=0
    for l in range(10):
        for m in range(50):#5N-5S, 170W-120W
            if not np.isnan(sst[i,85+l,m+190]):
                nino[i]+=sst[i,85+l,m+190]
                co2+=1
    nino[i]=nino[i]/co2
nino34 = np.zeros((1452))    
for i in range(2,1450):
    nino34[i]=np.mean(nino[i-2:i+3],axis=0)
nino34[0] = np.mean(nino[0:3],axis=0)
nino34[1] = np.mean(nino[0:4],axis=0)
nino34[1450] = np.mean(nino[1448:1451],axis=0)
nino34[1451] = np.mean(nino[1448:],axis=0)
nino12 = nino34[11::12]
nino1 = nino34[12::12]
nino2 = nino34[13::12]
DJFnindex = (nino12[:-1]+nino1+nino2)/3 
nino6 = nino34[5::12]
nino7 = nino34[6::12]
nino8 = nino34[7::12]
nino9 = nino34[8::12]
nino10 = nino34[9::12]
nino11 = nino34[10::12]
JJAnindex = (nino7+nino8+nino9+nino10+nino11)/5
mtce = [10,	10,	7,	10,	5,	10,	9,	10,	8,	7,	9,	13,	8,	15,	5,	14,	6,	7,	8,	1,	6,	11,	7,	7,	5,	13,	6,	5,	8,	6,	8,	1,	9,	6,	8,	5,	8,	9,	6,	8,	8,	7]
mtce1 = [0,	2,	1,	2,	1,	5,	5,	4,	2,	4,	2,	8,	3,	4,	2,	3,	2,	3,	2,	1,	0,	2,	5,	2,	2,	3,	2,	3,	0,	2,	3,	0,	1,	1,	1,	1,	4,	3,	1,	2,	3,	1]
mtce2 = [8,	4,	0,	3,	3,	2,	3,	3,	4,	3,	4,	3,	1,	3,	2,	4,	2,	1,	3,	0,	5,	5,	1,	1,	3,	2,	2,	0,	5,	4,	5,	1,	5,	3,	5,	0,	0,	5,	4,	2,	3,	6]
mtce3 = [2,	4,	6,	5,	1,	3,	1,	3,	2,	0,	3,	2,	4,	8,	1,	7,	2,	3,	3,	0,	1,	4,	1,	4,	0,	8,	2,	2,	3,	0,	0,	0,	3,	2,	2,	4,	4,	1,	1,	4,	2,	0]
mtce = np.array(mtce)
print(list((mtce-np.mean(mtce))/np.std(mtce)))
#mtce1 = list((mtce1-np.mean(mtce1))/np.std(mtce1))
#mtce2 = list((mtce2-np.mean(mtce2))/np.std(mtce2))
#mtce3 = list((mtce3-np.mean(mtce3))/np.std(mtce3))
#b, a = scipy.signal.butter(N=3, Wn=1/5, btype='lowpass', analog=False, output='ba')
#mtce = scipy.signal.filtfilt(b, a, mtce)

corr1 = np.zeros((4,12))
#print(list(mtce))
for i in range(12):
    nino6 = nino34[8+i::12]
    nino7 = nino34[9+i::12]
    nino8 = nino34[10+i::12]
    JJAnindex = (nino6[:119]+nino7[:119]+nino8[:119])/3
    a, b = pearsonr(mtce, JJAnindex[77:119])
    corr1[0,i] = a
    if i == 9:
        print(a,b,'Nino')
        a1 = JJAnindex[77:119]
        #print(list(JJAnindex[77:119]))
    #print(a,b)
data1=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/corr.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data1.append(table.row_values(i))
data2 = []
for i in range(0,44):
    for  j in range(12):
        data2.append(data1[i][7+j])
data2 = np.array(data2)
for i in range(12):
    nino6 = data2[8+i::12]
    nino7 = data2[9+i::12]
    nino8 = data2[10+i::12]
    JJAnindex = (nino6[:42]+nino7[:42]+nino8[:42])/3
    a, b = pearsonr(mtce, JJAnindex[0:42])
    corr1[1,i] = a
    if i == 6:
        a2 = JJAnindex[0:42]
        #print(list(JJAnindex[0:42]))
        print(a,b,'PMM')
    #print(a,b)
    

data2 = []
for i in range(0,44):
    for  j in range(12):
        data2.append(data1[i][49+j])
#print(data2[0])
data2 = np.array(data2)
for i in range(12):
    nino6 = data2[8+i::12]
    nino7 = data2[9+i::12]
    nino8 = data2[10+i::12]
    JJAnindex = (nino6[:42]+nino7[:42]+nino8[:42])/3
    a, b = pearsonr(mtce, JJAnindex[0:42])
    corr1[2,i] = a
    if i == 9:
        a2 = JJAnindex[0:42]
        #print(list(JJAnindex[0:42]))
        print(a,b,'TNA')

data2 = []
for i in range(0,44):
    for  j in range(12):
        data2.append(data1[i][21+j])
       
#print(data2[0])

data2 = np.array(data2)
data2 = (data2-np.mean(data2))/np.std(data2)

for i in range(12):
    nino6 = data2[8+i::12]
    nino7 = data2[9+i::12]
    nino8 = data2[10+i::12]
    JJAnindex = (nino6[:42]+nino7[:42]+nino8[:42])/3
    a, b = pearsonr(mtce, JJAnindex[0:42])
    corr1[3,i] = a
    #print(a,b)
    if i == 3:
        a3 = JJAnindex[0:42]
        #print(list(JJAnindex[0:42]))
        print(a,b,'NPI')

#np.save('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/a3.npy',corr1)        
"""

data2 = []
for i in range(0,len(data1)):
    for  j in range(12):
        data2.append(data1[i][35+j])
#print(data2[:])

data2 = np.array(data2)

b, a = scipy.signal.butter(N=3, Wn=1/60, btype='lowpass', analog=False, output='ba')
data2 = scipy.signal.filtfilt(b, a, data2)
data2 = (data2-np.mean(data2))/np.std(data2)
#print(list(data2))
#mtce = [8,	4,	0,	3,	3,	2,	3,	3,	4,	3,	4,	3,	1,	3,	2,	4,	2,	1,	3,	0,	5,	5,	1,	1,	3,	2,	2,	0,	5,	4,	5,	1,	5,	3,	5,	0,	0,	5,	4,	2,	3,	6]
for i in range(12):
    nino7 = data2[9+i::12]
    nino8 = data2[10+i::12]
    nino9 = data2[11+i::12]
    nino10 = data2[12+i::12]
    nino11 = data2[13+i::12]
    JJAnindex = (nino7[:42]+nino8[:42]+nino9[:42]+nino10[:42]+nino11[:42])/5
    a, b = pearsonr(mtce, JJAnindex[0:42])
    if i == 9:
        print(a,b)
        a4 = JJAnindex[0:42]
        #print(list(JJAnindex[0:42]))
        """
import pandas as pd  
import statsmodels.api as sm
file = r'/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/regress4.xlsx'
data = pd.read_excel(file)
data.columns = ['y', 'x1', 'x2','x3','x4','x5','x6','x7']
x = sm.add_constant(data.iloc[:,1:]) #生成自变量
y = data['y'] #生成因变量
#print(x)
model = sm.OLS(y, x) #生成模型
result = model.fit() #模型拟合
#print(result.summary()) #模型描述
f1 = []
f2 = []
f3 = []
t1 = 0
t2 = 0
for i in range(42):
  f1.append(2.4803+0.1735*x['x1'][i]-1.3446*x['x2'][i])  
  f2.append(2.951+0.736*x['x3'][i]-0.1563*x['x4'][i])  
  f3.append(3.0365+0.113*x['x5'][i]+0.8403*x['x6'][i]-2.3483*x['x7'][i])  
  t1+=x['x7'][i]*(mtce3[i]-np.mean(mtce3))*2.3483
  t2+=(mtce3[i]-np.mean(mtce3))**2
print(t1/t2,'exp')
  
a, b = pearsonr(mtce1,f1)
print(a,b)
a, b = pearsonr(mtce2,f2)
print(a,b)
a, b = pearsonr(mtce3,f3)
print(a,b)
f4 = np.array(f1)+np.array(f2)+np.array(f3)
a, b = pearsonr(mtce,f4)
print(a,b)

fig = plt.figure(figsize=(8,8),dpi=1000)
ax=[]
x1 = [0,0.4,0,0.4]
yy = [1,1,0.62,0.62]
dx = 0.32
dy = 0.3
xl = ['Reconstructions']
yl = ['EI frequency','WI-N frequency','WI-O frequency','MTCE frequency']

for i in range(4):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))
tit=['(a) EI: MAM TNA + JJA PMM','(b) WI-N: DJF NP + JFM PMM','(c) WI-O: MAM PMM + JJA Niño3.4 + JJA TNA','(d) Total']
#mtce = [0.752578332003859,	0.752578332003859,	-0.289451601493894,	0.752578332003859,	-0.98413822382573,	0.752578332003859,	0.405235020837941,	0.752578332003859,	0.0578917096720235,	-0.289451601493894,	0.405235020837941,	1.79460826550161,	0.0578917096720235,	2.48929488783345,	-0.98413822382573,	2.14195157666753,	-0.636794912659812,	-0.289451601493894,	0.0578917096720235,	-2.3735114684894,	-0.636794912659812,	1.09992164316978,	-0.289451601493894,	-0.289451601493894,	-0.98413822382573,	1.79460826550161,	-0.636794912659812,	-0.98413822382573,	0.0578917096720235,	-0.636794912659812,	0.0578917096720235,	-2.3735114684894,	0.405235020837941,	-0.636794912659812,	0.0578917096720235,	-0.98413822382573,	0.0578917096720235,	0.405235020837941,	-0.636794912659812,	0.0578917096720235,	0.0578917096720235,-0.289451601]
x = [f1,f2,f3,f4]
y = [mtce1,mtce2,mtce3,mtce]
cor = ['k','k','k','k']
words = ['corr = 0.53**','corr = 0.50**','corr = 0.59**','corr = 0.66**']
xloc = [0.7,-1.6,35,-75]
yt = [9.45,2.1,6.3,2.1]
el = [1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009, 2014, 2015, 2018, 2019]
la = [1983, 1984, 1988, 1995, 1998, 1999, 2000, 2005, 2007, 2008, 2010, 2011, 2017, 2020]
c = ['g','#f9bc86','#a3acff','gray']
xt = [-2.9,-2.9,-132,-132]
from sklearn.linear_model import LinearRegression
for i in range(4):
    regressor = LinearRegression()
    regressor = regressor.fit(np.reshape(x[i],(-1, 1)),np.reshape(y[i],(-1, 1)))
    print(regressor.coef_)
    ax[i].plot(np.reshape(x[i],(-1,1)), regressor.predict(np.reshape(x[i],(-1,1))),color=cor[i],linewidth=1.8)
    ax[i].scatter(x[i],y[i],s=18,color=c[i],linewidth=0)
    """
    for k in range(42):
        if k+1979 in la:
            ax[i].scatter(x[i][k],y[i][k],s=18,color='b',linewidth=0)
        if k+1979 in el:
            ax[i].scatter(x[i][k],y[i][k],s=18,color='r',linewidth=0)
            """
    ax[i].grid('--',linewidth=0.3,alpha=0.5)
    ax[i].set_xlabel(xl[0])
    ax[i].set_ylabel(yl[i])
    if i<=2:
        ax[i].set_ylim(-0.2,8.2)
        ax[i].set_xlim(-0.2,8.2)
        ax[i].set_yticks([0,2,4,6,8])
        ax[i].text(-0.2,8.4,tit[i])
        ax[i].text(5,0.8,words[i])

ax[3].set_ylim(-0.4,15.4)
ax[3].set_xlim(-0.4,15.4)
ax[3].set_yticks([0,5,10,15])
ax[3].text(-0.4,15.8,tit[3])
ax[3].text(9.375,1.5,words[3])
ax[0].text(4.5,2,'MAM TNA: 14.4%')
ax[0].text(4.5,1.4,'JJA PMM: 14.2%')
ax[1].text(4.5,2,'DJF NP: 17.2%')
ax[1].text(4.5,1.4,'JFM PMM: 7.9%')
ax[2].text(4.5,2.6,'MAM PMM: 7.0%')
ax[2].text(4.5,2,'JJA Niño3.4: 10.0%')
ax[2].text(4.5,1.4,'JJA TNA: 18.2%')

plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/MLR.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)

 
