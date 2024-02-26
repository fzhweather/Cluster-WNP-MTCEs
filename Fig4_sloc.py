#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 21:58:19 2023

@author: fuzhenghang
"""

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
import numpy as np
import xarray as xr
import matplotlib as mpl
import scipy.signal as signal
from numpy import nan
from matplotlib import pylab
import xlrd
import seaborn as sns
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
sns.reset_orig()

mpl.rcParams["font.family"] = 'Arial'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 9
data=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/locationTCmother1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data.append(table.row_values(i))
data = [data[i] for i in range(0,len(data))]

x=[]
y=[]
z=[]

for i in range(len(data)):
    x.append(data[i][6])
x=[i/10 for i in x]
for i in range(len(data)):
    y.append(data[i][5])
y=[i/10 for i in y]
for i in range(len(data)):
    z.append(data[i][0])

data1=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/locationTCson1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data1.append(table.row_values(i))
data1 = [data1[i] for i in range(0,len(data1))]

x11=[]
y11=[]

for i in range(len(data1)):
    x11.append(data1[i][6])
x11=[i/10 for i in x11]
for i in range(len(data1)):
    y11.append(data1[i][5])
y11=[i/10 for i in y11]

color = [plt.get_cmap("seismic", 42)(int(float(i-1979))) for i in z]

fig = plt.figure(figsize=(5,8),dpi=800)#设置比例
x1 = [0.2,0.2,0.2]
yy = [0.95,0.67,0.39]
dx = 0.7
dy = 0.25
ax = []
proj = ccrs.PlateCarree(central_longitude=180) 
label = ['(a) Mother-TC','(b) Son-TC','(c) Mother-TC intensity']
for i in range(3):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))

leftlon, rightlon, lowerlat, upperlat = (100,180,3,48)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
for i in range(3):
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    #ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5)
    ax[i].add_feature(cfeature.LAND.with_scale('50m'),color='lightgrey',zorder=0)
    gl = ax[i].gridlines(draw_labels=True, linewidth=0.1, color='k', linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,140,160,180])
    gl.ylocator = mticker.FixedLocator([0,15,30,45])
    gl.top_labels    = False    
    gl.right_labels  = False
    gl.bottom_labels  = False
    gl.xpadding = 2
    gl.ypadding = 2
    if i == 2:
        gl.bottom_labels  = True
    ax[i].text(-78,49,label[i])
    
plt.set_cmap(plt.get_cmap("seismic", 42))
a = ax[0].scatter(x,y,s=20,c=color,marker='.',transform=ccrs.PlateCarree())
b = ax[1].scatter(x11,y11,s=20,c=color,marker='.',transform=ccrs.PlateCarree())
position=fig.add_axes([0.93, 0.76, 0.02, 0.36])
plt.colorbar(a,cax=position, format=matplotlib.ticker.FuncFormatter(lambda i,pos:int(i*42+1979)),orientation='vertical',aspect=40,fraction=0.03,pad=0.08)
x=[]
y=[]
z=[]

for i in range(len(data)):
    x.append(data[i][6])
x=[i/10 for i in x]
for i in range(len(data)):
    y.append(data[i][5])
y=[i/10 for i in y]
for i in range(len(data)):
    z.append(data[i][8])

maxz=max(z)
minz=min(z)
color = [plt.get_cmap("seismic", 52)(int(float(i-minz)/(maxz-minz)*52)) for i in z]
c = ax[2].scatter(x,y,s=20,c=color,marker='.',transform=ccrs.PlateCarree())
position=fig.add_axes([0.93, 0.43, 0.02, 0.18])
plt.colorbar(a, cax=position,format=matplotlib.ticker.FuncFormatter(lambda i,pos:int(i*(maxz-minz)+minz)),orientation='vertical',aspect=40,fraction=0.03,pad=0.08)
ax[2].text(5,5,'m/s',fontsize=10)
ax[2].text(-77.7,44.5,'1979-1999: 31.75m/s',fontsize=7)
ax[2].text(-77.7,41.5,'2000-2020: 32.53m/s',fontsize=7)
ax[0].text(-77.7,44.5,'1979-1999: 20.87°N, 132.94°E',fontsize=7)
ax[0].text(-77.7,41.5,'2000-2020: 23.14°N, 131.71°E',fontsize=7)
ax[1].text(-77.7,44.5,'1979-1999: 16.98°N, 138.85°E',fontsize=7)
ax[1].text(-77.7,41.5,'2000-2020: 16.93°N, 138.87°E',fontsize=7)
loc1=[132.93616-180,20.86949,131.70855-180,23.14408]
lco2=[138.84576-180,16.98418,138.88618-180,16.92829]
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/fig4.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)



