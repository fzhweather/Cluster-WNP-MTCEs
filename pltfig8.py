#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:39:20 2023

@author: fuzhenghang
"""
import matplotlib.pyplot as plt###引入库包
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import cartopy.mpl.ticker as cticker
import matplotlib.colors
import xarray as xr
import xlrd
import math
import cmaps
import matplotlib.ticker as mticker

mpl.rcParams["font.family"] = 'Arial'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 0.6

proj = ccrs.PlateCarree()  #中国为左
leftlon, rightlon, lowerlat, upperlat = (100,180,0,50)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
fig = plt.figure(figsize=(9,6),dpi=800)  
x1 = [0.05,0.05,0.05,0.05,0.32,0.32,0.32,0.32,0.59,0.59,0.59,0.59]
yy = [0.8,0.54,0.28,0.02,0.8,0.54,0.28,0.02,0.8,0.54,0.28,0.02]
dx = 0.4
dy = 0.23
ax = []
label = ['(a) EI: -72 h','(b) EI: -48 h','(c) EI: -24 h','(d) EI: 0 h',
         '(e) WI-N: -72 h','(f) WI-N: -48 h','(g) WI-N: -24 h','(h) WI-N: 0 h',
         '(i) WI-O: -72 h','(j) WI-O: -48 h','(k) WI-O: -24 h','(l) WI-O: 0 h']
for i in range(12):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
    
input_data = r'/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/850winduv11.nc'
d1 = Dataset(input_data)                                                             
lat=d1.variables['latitude'][:] 
lon=d1.variables['longitude'][:]
lon2d, lat2d = np.meshgrid(lon, lat)

u1 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/ua1.npy')
v1 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/va1.npy')
msl1 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/msla1.npy')
u2 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/ua2.npy')
v2 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/va2.npy')
msl2 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/msla2.npy')
u3 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/ua3.npy')
v3 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/va3.npy')
msl3 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/msla3.npy')
u_all=np.zeros((12,71,91))
v_all=np.zeros((12,71,91))
msl_all=np.zeros((12,71,91))
for i in range(4):
    u_all[0:4]=u1
    u_all[4:8]=u2
    u_all[8:12]=u3
    v_all[0:4]=v1
    v_all[4:8]=v2
    v_all[8:12]=v3
    msl_all[0:4]=msl1
    msl_all[4:8]=msl2
    msl_all[8:12]=msl3
    

a = ['/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/locationTCson10.xlsx',
     '/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/locationTCson11.xlsx',
     '/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/locationTCson12.xlsx']
b = ['/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m0.xlsx','/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m01.xlsx',
     '/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m11.xlsx','/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m21.xlsx',
     '/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m1.xlsx','/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m02.xlsx',
     '/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m12.xlsx','/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m22.xlsx',
     '/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m2.xlsx','/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m03.xlsx',
     '/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m13.xlsx','/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/m23.xlsx']
for i in range(12):
    data=[]
    table=xlrd.open_workbook(a[i//4])
    table=table.sheets()[0]
    nrows=table.nrows
    for m in range(nrows):
        if m ==0:
            continue
        data.append(table.row_values(m))
    data = [data[m] for m in range(0,len(data))]
    data1=[]
    table1=xlrd.open_workbook(b[i])
    table1=table1.sheets()[0]
    nrows=table1.nrows
    for m in range(nrows):
        if m ==0:
            continue
        data1.append(table1.row_values(m))
    data1 = [data1[m] for m in range(0,len(data1))]
    levels = np.arange(-2 , 2 + 0.1,0.2)
    cb = ax[i].contourf(lon2d,lat2d,msl_all[i],levels=levels,cmap=cmaps.BlueDarkRed18 ,transform=ccrs.PlateCarree(),extend='both')
    #plt.contour(lon2d,lat2d,msl_all,[575],colors='w')
    cq = ax[i].quiver(lon2d[::4,::4],lat2d[::4,::4],u_all[i][::4,::4],v_all[i][::4,::4],color='k',
                      transform=ccrs.PlateCarree(),scale=50,width=0.0065,edgecolor='w',linewidth=0.2)
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.4)
    gl = ax[i].gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,140,160,160,180])
    gl.ylocator = mticker.FixedLocator(np.arange(-15,60,15))
    gl.top_labels    = False
    gl.right_labels    = False
    if i not in [3,7,11]:
        gl.bottom_labels    = False
    if i > 3:
        gl.left_labels    = False
    if i == 0:
        ax[i].quiverkey(cq, X=0.83, Y = 1.05, U = 5,angle = 0,label='5m/s',labelpos='E', color = 'k',labelcolor = 'k')
        #ax[i].text(174.5,51.5,'5m/s',color='k',size=5)
    gl.ypadding=2
    gl.xpadding=2
    x=[]
    y=[]
    d = i%4
    if d==3:
        for m in range(len(data)):
            x.append(data[m][6])
        x=[m/10 for m in x]
        for m in range(len(data)):
            y.append(data[m][5])
        y=[m/10 for m in y]
        print(y)
        ax[i].scatter(x,y,s=4,c='red',marker='.')
        ax[i].scatter(np.mean(x),np.mean(y),s=25,c='yellow',marker='+',zorder =10)
    x1=[]
    y1=[]
    for m in range(len(data1)):
        x1.append(data1[m][6])
    x1=[m/10 for m in x1]
    for m in range(len(data1)):
        y1.append(data1[m][5])
    y1=[m/10 for m in y1]
    
    ax[i+3-2*d].scatter(x1,y1,s=4,c='blue',marker='.',zorder=10)
    ax[i+3-2*d].scatter(np.mean(x1),np.mean(y1),s=20,c='yellow',marker='*',zorder=10)
    if d==3:
        position=fig.add_axes([0.3, -0.018, 0.442, 0.018])
        cbar=fig.colorbar(cb,cax=position,orientation='horizontal',ticks=np.arange(-2 , 2 + 0.001, 1),
                          aspect=20,shrink=0.5,pad=0.04)
        cbar.ax.tick_params(pad=0.08,length=0.5,width=0.7)
    ax[i].text(102,51.5,label[i])


plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/circulation.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)

