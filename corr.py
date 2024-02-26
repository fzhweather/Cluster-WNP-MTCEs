

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:39:16 2022

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
mpl.rcParams["font.size"] = 7
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
JJAnindex = (nino6+nino7+nino8+nino9+nino10)/5
"""
mtce = [10,	10,	7,	10,	5,	10,	9,	10,	8,	7,	9,	13,	8,	15,	5,	14,	6,	7,	8,	1,	6,	11,	7,	7,	5,	13,	6,	5,	8,	6,	8,	1,	9,	6,	8,	5,	8,	9,	6,	8,	8,	7]
for i in range(12):
    nino6 = nino34[8+i::12]
    nino7 = nino34[9+i::12]
    nino8 = nino34[10+i::12]
    JJAnindex = (nino6[:119]+nino7[:119]+nino8[:119])/3
    a, b = pearsonr(mtce, JJAnindex[77:119])
    if i == 9:
        a1 = JJAnindex[77:119]
        #print(list(JJAnindex[77:119]))
    print(a,b)
data1=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/corr.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data1.append(table.row_values(i))
data2 = []
for i in range(0,len(data1)):
    for  j in range(12):
        data2.append(data1[i][7+j])
data2 = np.array(data2)
for i in range(12):
    nino6 = data2[8+i::12]
    nino7 = data2[9+i::12]
    nino8 = data2[10+i::12]
    JJAnindex = (nino6[:43]+nino7[:43]+nino8[:43])/3
    if i == 9:
        a2 = JJAnindex[0:42]
        #print(list(JJAnindex[0:42]))
    a, b = pearsonr(mtce, JJAnindex[0:42])
    print(a,b)
data2 = []
for i in range(0,len(data1)):
    for  j in range(12):
        data2.append(data1[i][21+j])
#print(data2[0])
data2 = np.array(data2)
#mtce = [8,	4,	0,	3,	3,	2,	3,	3,	4,	3,	4,	3,	1,	3,	2,	4,	2,	1,	3,	0,	5,	5,	1,	1,	3,	2,	2,	0,	5,	4,	5,	1,	5,	3,	5,	0,	0,	5,	4,	2,	3,	6]
for i in range(12):
    nino6 = data2[8+i::12]
    nino7 = data2[9+i::12]
    nino8 = data2[10+i::12]
    JJAnindex = (nino6[:43]+nino7[:43]+nino8[:43])/3
    if i == 3:
        a3 = JJAnindex[0:42]
        #print(list(JJAnindex[0:42]))
    a, b = pearsonr(mtce, JJAnindex[0:42])
    print(a,b)
data2 = []
for i in range(0,len(data1)):
    for  j in range(12):
        data2.append(data1[i][35+j])
#print(data2[:])
data2 = np.array(data2)
#mtce = [8,	4,	0,	3,	3,	2,	3,	3,	4,	3,	4,	3,	1,	3,	2,	4,	2,	1,	3,	0,	5,	5,	1,	1,	3,	2,	2,	0,	5,	4,	5,	1,	5,	3,	5,	0,	0,	5,	4,	2,	3,	6]
for i in range(12):
    nino6 = data2[8+i::12]
    nino7 = data2[9+i::12]
    nino8 = data2[10+i::12]
    JJAnindex = (nino6[:42]+nino7[:42]+nino8[:42])/3
    a, b = pearsonr(mtce, JJAnindex[0:42])
    if i == 9:
        a4 = JJAnindex[0:42]
        #print(list(JJAnindex[0:42]))
    print(a,b)
import pandas as pd  
import statsmodels.api as sm
file = r'/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/regress1.xlsx'
data = pd.read_excel(file)
data.columns = ['y', 'x1', 'x2','x3','x4']
x = sm.add_constant(data.iloc[:,1:]) #生成自变量
y = data['y'] #生成因变量
model = sm.OLS(y, x) #生成模型
result = model.fit() #模型拟合
print(result.summary()) #模型描述
fitted = []
for i in range(42):
  fitted.append(-0.0675+0.1637*a1[i]+0.1189*a2[i]-0.3174*a3[i]-2.129*a4[i])  
a, b = pearsonr(mtce,fitted )
print(a,b)
"""
nino12 = sst[11::12]
nino1 = sst[12::12]
nino2 = sst[13::12]
DJFnindex = (nino12[:-1]+nino1+nino2)/3 
nino6 = sst[5::12]
nino7 = sst[6::12]
nino8 = sst[7::12]
nino9 = sst[8::12]
nino10 = sst[9::12]
nino11 = sst[10::12]
JJAnindex = (nino7+nino8+nino9+nino10+nino11)/5
SONnindex = (nino9+nino10+nino11)/3 

# In[1]

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

for i in range(13,55):
    jtwcf.append(data[i][1])
    jmaf.append(data[i][2])
    cmaf.append(data[i][3])
for i in range(13,55):
    jtwc.append(data[i][4])
    jma.append(data[i][5])
    cma.append(data[i][6])

xdata=[jtwcf,jmaf,cmaf,jtwcf,jmaf,cmaf]
print(jtwcf)
#m = DJFnindex[77:119]




fig = plt.figure(figsize=(8,4),dpi=600)
yt=[[0,0.5,1,1.5,2,2.5],[0,0.5,1,1.5,2],[0,0.5,1,1.5]]
x1 = [0.1,0.1,0.1,0.53,0.53,0.53]
yy = [0.99,0.645,0.3,0.99,0.645,0.3]
dx = 0.6
dy = 0.3
ax = []
lon1=[90,90,90]
label = ['(a) EI: Frequency & D(-1)JF SST','(b) WI-N: Frequency & D(-1)JF SST','(c) WI-O: Frequency & D(-1)JF SST','(d) EI: Frequency & JASON SST','(e) WI-N: Frequency & JASON SST','(f) WI-O: Frequency & JASON SST']
proj = ccrs.PlateCarree(central_longitude=180)  #中国为左
leftlon, rightlon, lowerlat, upperlat = (2,359.9,-65,65)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
for i in range(6):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))

for num in range(6):
    if num <= 2:
        m = DJFnindex[77:119]
    else:
        m = JJAnindex[78:120]
    sstr = np.zeros((2,181,360))
    x=xdata[num]
    for i in range(len(lat)):
        for j in range(len(lon)):
            if not (np.isnan(m[:,i,j]).any()):
                sstr[0,i,j],sstr[1,i,j] = pearsonr(x, m[:,i,j])
                
            else:
                sstr[0,i,j],sstr[1,i,j] = nan,nan
        for j in range(len(lon)):
            if not (np.isnan(m[:,i,j]).any()):
                sstr[0,i,j],sstr[1,i,j] = pearsonr(x, m[:,i,j])
            else:
                sstr[0,i,j],sstr[1,i,j] = nan,nan
    ll=[]
    for i in range(len(lat)):
        for j in range(len(lon)):
            if not np.isnan(sstr[1,i,j]):
                ll.append(sstr[1,i,j])

    sstr1, cycle_lon =add_cyclic_point(sstr[1,:,:], coord=lon)
    sstr0, cycle_lon =add_cyclic_point(sstr[0,:,:], coord=lon)

    c1 = ax[num].contourf(cycle_lon,lat,sstr0,zorder=1,levels=np.arange(-0.6 , 0.6 +0.01, 0.06),extend = 'both', transform=ccrs.PlateCarree(),cmap=cmaps.BlueDarkRed18)
    c1b = ax[num].contourf(cycle_lon,lat,sstr1,[min(ll),0.05,max(ll)], zorder=1,hatches=['....', None],colors="none", edgecolor='w',transform=ccrs.PlateCarree())

    ax[num].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())

    #ax[num].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5)
    gl = ax[num].gridlines(draw_labels=True, linewidth=0.1, color='k',  linestyle='--')
    ax[num].add_feature(cfeature.LAND.with_scale('110m'),color='lightgrey',zorder=0)
    gl.xlocator = mticker.FixedLocator([-120,-60,60,120,180])
    gl.ylocator = mticker.FixedLocator(np.arange(-40,59,20))
    gl.top_labels    = False 
    gl.right_labels  =  False
    ax[num].text(-175,68,label[num])
    if num in [0,1,3,4]:
        gl.bottom_labels  = False 
    if num in [3,4,5]:
        gl.left_labels  = False
    gl.xpadding = 2
    gl.ypadding = 2
position=fig.add_axes([0.365, 0.23, 0.5, 0.025])
cbar=fig.colorbar(c1,cax=position,orientation='horizontal',ticks=np.arange(-0.6 , 0.6 + 0.1, 0.2),
                  aspect=20,shrink=0.5,pad=0.04)
cbar.ax.tick_params(pad=0.2,length=0.5,width=0.7)
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/fig7.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)




