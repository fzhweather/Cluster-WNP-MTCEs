
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:14:23 2022

@author: fuzhenghang
"""

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors
import numpy as np
import xarray as xr
import matplotlib as mpl
import scipy.signal as signal
from numpy import nan
from matplotlib import pylab
import xlrd



mpl.rcParams["font.family"] = 'Arial'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 10
data1=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/genesis_locationTCmother1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data1.append(table.row_values(i))
data1 = [data1[i] for i in range(0,len(data1))]


data2=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/locationTCson1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data2.append(table.row_values(i))
data2 = [data2[i] for i in range(0,len(data2))]

data3=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/locationTCmother1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data3.append(table.row_values(i))
data3 = [data3[i] for i in range(0,len(data3))]

lat=[]
lon=[]
for i in range(len(data1)):
    lat.append(data2[i][5]-data1[i][5])
for i in range(len(data1)):
    lon.append(data2[i][6]-data1[i][6])
lat=[i/10 for i in lat]
lon=[i/10 for i in lon]
a=sum(data1[i][5] for i in range(len(data1)))/len(data1)/10
b=sum(data1[i][6] for i in range(len(data1)))/len(data1)/10

lat1=[]
lon1=[]
for i in range(len(data1)):
    lat1.append(data2[i][5]-data3[i][5])
for i in range(len(data1)):
    lon1.append(data2[i][6]-data3[i][6])
lat1=[i/10 for i in lat1]
lon1=[i/10 for i in lon1]


fig = plt.figure(figsize=(6.25,5), dpi=600)
fig1 = fig.add_axes([0.1,0.95,0.8,0.4])
fig2 = fig.add_axes([0.1,0.47,0.8,0.4])
fig1.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False,labelcolor='k',length=2)

fig1.set_xlim([-50,50])
fig1.set_ylim([-25,25])
fig1.scatter(0,0,s=60,marker='*',color='r')
fig1.scatter(lon,lat,s=20,marker='.',color='coral')
plt.rcParams.update({'font.size':8})
fig1.legend(["Mother-TC location", "Son-TC location"],loc='upper left',frameon=False)
fig1.grid(color="gray", linestyle=":")
mpl.rcParams["font.family"] = 'Times New Roman'
fig1.text(40,18.3,'+', fontsize=40,horizontalalignment='center', verticalalignment='center')
mpl.rcParams["font.family"] = 'Arial'
fig1.text(44,22,'81', fontsize=9,horizontalalignment='center', verticalalignment='center')
fig1.text(36,22,'88', fontsize=9,horizontalalignment='center', verticalalignment='center')
fig1.text(36,17,'68', fontsize=9,horizontalalignment='center', verticalalignment='center')
fig1.text(44,17,'91', fontsize=9,horizontalalignment='center', verticalalignment='center')


fig2.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False,labelcolor='k',length=2)

fig2.set_xlim([-50,50])
fig2.set_ylim([-25,25])
fig2.scatter(0,0,s=60,marker='*',color='r')
fig2.scatter(lon1,lat1,s=20,marker='.',color='coral')
plt.rcParams.update({'font.size':8})

fig2.grid(color="gray", linestyle=":")
mpl.rcParams["font.family"] = 'Times New Roman'
fig2.text(40,18.6,'+', fontsize=40,horizontalalignment='center', verticalalignment='center')
mpl.rcParams["font.family"] = 'Arial'
fig2.text(44,22,'56', fontsize=9,horizontalalignment='center', verticalalignment='center')
fig2.text(36,22,'39', fontsize=9,horizontalalignment='center', verticalalignment='center')
fig2.text(36,17,'86', fontsize=9,horizontalalignment='center', verticalalignment='center')
fig2.text(44.5,17,'145', fontsize=9,horizontalalignment='center', verticalalignment='center')
l=0
m=0
n=0
o=0
for i in range(len(lat)):
    if lat[i]>0 and lon[i]>0:
        l+=1
    elif lat[i]>0 and lon[i]<0:
        m+=1
    elif lat[i]<0 and lon[i]<0:
        n+=1
    elif lat[i]<0 and lon[i]>0:
        o+=1
print(l,m,n,o)
l=0
m=0
n=0
o=0
for i in range(len(lat1)):
    if lat1[i]>0 and lon1[i]>0:
        l+=1
    elif lat1[i]>0 and lon1[i]<0:
        m+=1
    elif lat1[i]<0 and lon1[i]<0:
        n+=1
    elif lat1[i]<0 and lon1[i]>0:
        o+=1
print(l,m,n,o)
fig1.set_xticklabels([])
fig2.text(0,-35,'Latitudinal distance',horizontalalignment='center',fontsize=10)
fig1.text(-59.5,-15,'Longitudinal distance',horizontalalignment='center',fontsize=10,rotation=90)
fig2.text(-59.5,-15,'Longitudinal distance',horizontalalignment='center',fontsize=10,rotation=90)


fig1.text(-49,27,"(a) Genesis location of mother-TC",fontsize=10)
fig2.text(-49,27,"(b) Location of mother-TC",fontsize=10)
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/fig3.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)



