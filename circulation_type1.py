#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 20:01:27 2022

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

mpl.rcParams["font.family"] = 'Times New Roman'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 0.6

proj = ccrs.PlateCarree()  #中国为左
leftlon, rightlon, lowerlat, upperlat = (-100,-20,0,50)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
fig = plt.figure(figsize=(9,6),dpi=800)  
x1 = [0.05,0.05,0.33,0.33]
yy = [0.95,0.67,0.95,0.67]
dx = 0.4
dy = 0.25
ax = []
label = ['(a) -72 h','(c) -24 h','(b) -48 h','(d) 0 h']
for i in range(4):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))


def circulation(a,b,ax,d):
    data=[]
    table=xlrd.open_workbook(a)
    table=table.sheets()[0]
    nrows=table.nrows
    for i in range(nrows):
        if i ==0:
            continue
        data.append(table.row_values(i))
    data = [data[i] for i in range(0,len(data))]
    data1=[]
    table1=xlrd.open_workbook(b)
    table1=table1.sheets()[0]
    nrows=table1.nrows
    for i in range(nrows):
        if i ==0:
            continue
        data1.append(table1.row_values(i))
    data1 = [data1[i] for i in range(0,len(data1))]

            
    input_data = r'/mnt/data/xshome/fuzh19/nc/850wduv.nc'
    d1 = Dataset(input_data)
                                            
    d2 = Dataset(r'/mnt/data/xshome/fuzh19/nc/500gh.nc')                         
    msl=d2.variables["z"][:]/98.0665
    lat=d1.variables['latitude'][:]
    lon=d1.variables['longitude'][:]
    time=d1.variables['time'][:]
    u8501=d1.variables["u"][:]
    v8501=d1.variables["v"][:]
    lon2d, lat2d = np.meshgrid(lon, lat)
    u850=u8501
    v850=v8501


    d1=xr.open_dataset('/mnt/data/xshome/fuzh19/nc/850wduv.nc')
    d2=xr.open_dataset('/mnt/data/xshome/fuzh19/nc/500gh.nc')
    if d==3:
        u850sum=u850[[(time==(int(data[0][11])))]]
        v850sum=v850[[(time==(int(data[0][11])))]]
        mslsum=msl[[(time==(int(data[0][11])))]]
        d10=d1.u.loc[d1.time.dt.month.isin([int(data[0][1])])]
        d100=d10.loc[d10.time.dt.day.isin([int(data[0][2])])]
        dv10=d1.v.loc[d1.time.dt.month.isin([int(data[0][1])])]
        dv100=dv10.loc[dv10.time.dt.day.isin([int(data[0][2])])]
        hisu850sum=np.mean(d100,axis=0)
        hisv850sum=np.mean(dv100,axis=0)
        d20=d2.z.loc[d1.time.dt.month.isin([int(data[0][1])])]
        d200=d20.loc[d10.time.dt.day.isin([int(data[0][2])])]
        hismslsum=np.mean(d200/98.0665,axis=0)
        
        for i in range(1,len(data)):
            u850sum+=u850[[(time==(int(data[i][11])))]]
            v850sum+=v850[[(time==(int(data[i][11])))]]
            mslsum+=msl[[(time==(int(data[i][11])))]]
            d10=d1.u.loc[d1.time.dt.month.isin([int(data[i][1])])]
            d100=d10.loc[d10.time.dt.day.isin([int(data[i][2])])]
            dv10=d1.v.loc[d1.time.dt.month.isin([int(data[0][1])])]
            dv100=dv10.loc[dv10.time.dt.day.isin([int(data[0][2])])]
            hisu850sum+=np.mean(d100,axis=0)
            hisv850sum+=np.mean(dv100,axis=0)
            d20=d2.z.loc[d1.time.dt.month.isin([int(data[i][1])])]
            d200=d20.loc[d10.time.dt.day.isin([int(data[i][2])])]
            hismslsum+=np.mean(d200/98.0665,axis=0)
            
            
        hisu8501=hisu850sum/len(data)
        hisv8501=hisv850sum/len(data)
        hismsl1=hismslsum/len(data)
            
        u8501=u850sum/len(data)
        v8501=v850sum/len(data)
        msl1=mslsum/len(data)
        hisu850mean=hisu8501.copy()
        hisv850mean=hisv8501.copy()  
        hismslmean=hismsl1.copy()

        u850mean=u8501.copy()
        v850mean=v8501.copy()  
        mslmean=msl1.copy()
        
        m=[2,3,4,2,5,6,7,2,8,9,2]
        K=[]
        for i in range(len(m)):
            K.append(1/2*((1-math.cos(2*math.pi/m[i]))**(-1)))
        for k in range(len(K)):
            for y in range(len(lat)):
                for x in range(1,len(lon)-1):
                    u850mean[:,y,x]=(u8501[:,y,x]+K[k]*(u8501[:,y,x-1]+u8501[:,y,x+1]-2*u8501[:,y,x]))
                    v850mean[:,y,x]=(v8501[:,y,x]+K[k]*(v8501[:,y,x-1]+v8501[:,y,x+1]-2*v8501[:,y,x]))
                    mslmean[:,y,x]=msl1[:,y,x]+K[k]*(msl1[:,y,x-1]+msl1[:,y,x+1]-2*msl1[:,y,x])
                    hisu850mean[y,x]=(hisu8501[y,x]+K[k]*(hisu8501[y,x-1]+hisu8501[y,x+1]-2*hisu8501[y,x]))
                    hisv850mean[y,x]=(hisv8501[y,x]+K[k]*(hisv8501[y,x-1]+hisv8501[y,x+1]-2*hisv8501[y,x]))
                    hismslmean[y,x]=hismsl1[y,x]+K[k]*(hismsl1[y,x-1]+hismsl1[y,x+1]-2*hismsl1[y,x])
              
            u8501=u850mean.copy()
            v8501=v850mean.copy()  
            msl1=mslmean.copy()
            hisu8501=hisu850mean.copy()
            hisv8501=hisv850mean.copy()  
            hismsl1=hismslmean.copy()
            print(msl1.shape)        
            for x in range(len(lon)):
                for y in range(1,len(lat)-1):
                    u850mean[:,y,x]=u8501[:,y,x]+K[k]*(u8501[:,y-1,x]+u8501[:,y+1,x]-2*u8501[:,y,x])
                    v850mean[:,y,x]=v8501[:,y,x]+K[k]*(v8501[:,y-1,x]+v8501[:,y+1,x]-2*v8501[:,y,x])
                    mslmean[:,y,x]=msl1[:,y,x]+K[k]*(msl1[:,y-1,x]+msl1[:,y+1,x]-2*msl1[:,y,x])
                    hisu850mean[y,x]=(hisu8501[y,x]+K[k]*(hisu8501[y-1,x]+hisu8501[y+1,x]-2*hisu8501[y,x]))
                    hisv850mean[y,x]=(hisv8501[y,x]+K[k]*(hisv8501[y-1,x]+hisv8501[y+1,x]-2*hisv8501[y,x]))
                    hismslmean[y,x]=hismsl1[y,x]+K[k]*(hismsl1[y-1,x]+hismsl1[y+1,x]-2*hismsl1[y,x])

            u8501=u850mean.copy()
            v8501=v850mean.copy()
            msl1=mslmean.copy()
            hisu8501=hisu850mean.copy()
            hisv8501=hisv850mean.copy()  
            hismsl1=hismslmean.copy()
            
    if d==1:
        u850sum=u850[[(time==(int(data[0][14])))]]
        v850sum=v850[[(time==(int(data[0][14])))]]
        mslsum=msl[[(time==(int(data[0][14])))]]
        d10=d1.u.loc[d1.time.dt.month.isin([int(data[0][12])])]
        d100=d10.loc[d10.time.dt.day.isin([int(data[0][13])])]
        dv10=d1.v.loc[d1.time.dt.month.isin([int(data[0][12])])]
        dv100=dv10.loc[dv10.time.dt.day.isin([int(data[0][13])])]
        hisu850sum=np.mean(d100,axis=0)
        hisv850sum=np.mean(dv100,axis=0)
        d20=d2.z.loc[d1.time.dt.month.isin([int(data[0][12])])]
        d200=d20.loc[d10.time.dt.day.isin([int(data[0][13])])]
        hismslsum=np.mean(d200/98.0665,axis=0)
        #print(list(hismslsum))
        for i in range(1,len(data)):
            u850sum+=u850[[(time==(int(data[i][14])))]]
            v850sum+=v850[[(time==(int(data[i][14])))]]
            mslsum+=msl[[(time==(int(data[i][14])))]]
            d10=d1.u.loc[d1.time.dt.month.isin([int(data[i][12])])]
            d100=d10.loc[d10.time.dt.day.isin([int(data[i][13])])]
            dv10=d1.v.loc[d1.time.dt.month.isin([int(data[0][12])])]
            dv100=dv10.loc[dv10.time.dt.day.isin([int(data[0][13])])]
            hisu850sum+=np.mean(d100,axis=0)
            hisv850sum+=np.mean(dv100,axis=0)
            d20=d2.z.loc[d1.time.dt.month.isin([int(data[i][12])])]
            d200=d20.loc[d10.time.dt.day.isin([int(data[i][13])])]
            hismslsum+=np.mean(d200/98.0665,axis=0)
            
        hisu8501=hisu850sum/len(data)
        hisv8501=hisv850sum/len(data)
        hismsl1=hismslsum/len(data)
            
        u8501=u850sum/len(data)
        v8501=v850sum/len(data)
        msl1=mslsum/len(data)

        hisu850mean=hisu8501.copy()
        hisv850mean=hisv8501.copy()  
        hismslmean=hismsl1.copy()

        u850mean=u8501.copy()
        v850mean=v8501.copy()  
        mslmean=msl1.copy()
        
        
        m=[2,3,4,2,5,6,7,2,8,9,2]
        K=[]
        for i in range(len(m)):
            K.append(1/2*((1-math.cos(2*math.pi/m[i]))**(-1)))
        for k in range(len(K)):
            for y in range(len(lat)):
                for x in range(1,len(lon)-1):
                    u850mean[:,y,x]=(u8501[:,y,x]+K[k]*(u8501[:,y,x-1]+u8501[:,y,x+1]-2*u8501[:,y,x]))
                    v850mean[:,y,x]=(v8501[:,y,x]+K[k]*(v8501[:,y,x-1]+v8501[:,y,x+1]-2*v8501[:,y,x]))
                    mslmean[:,y,x]=msl1[:,y,x]+K[k]*(msl1[:,y,x-1]+msl1[:,y,x+1]-2*msl1[:,y,x])
                    hisu850mean[y,x]=(hisu8501[y,x]+K[k]*(hisu8501[y,x-1]+hisu8501[y,x+1]-2*hisu8501[y,x]))
                    hisv850mean[y,x]=(hisv8501[y,x]+K[k]*(hisv8501[y,x-1]+hisv8501[y,x+1]-2*hisv8501[y,x]))
                    hismslmean[y,x]=hismsl1[y,x]+K[k]*(hismsl1[y,x-1]+hismsl1[y,x+1]-2*hismsl1[y,x])
              
            u8501=u850mean.copy()
            v8501=v850mean.copy()  
            msl1=mslmean.copy()
            hisu8501=hisu850mean.copy()
            hisv8501=hisv850mean.copy()  
            hismsl1=hismslmean.copy()
            print(msl1.shape)        
            for x in range(len(lon)):
                for y in range(1,len(lat)-1):
                    u850mean[:,y,x]=u8501[:,y,x]+K[k]*(u8501[:,y-1,x]+u8501[:,y+1,x]-2*u8501[:,y,x])
                    v850mean[:,y,x]=v8501[:,y,x]+K[k]*(v8501[:,y-1,x]+v8501[:,y+1,x]-2*v8501[:,y,x])
                    mslmean[:,y,x]=msl1[:,y,x]+K[k]*(msl1[:,y-1,x]+msl1[:,y+1,x]-2*msl1[:,y,x])
                    hisu850mean[y,x]=(hisu8501[y,x]+K[k]*(hisu8501[y-1,x]+hisu8501[y+1,x]-2*hisu8501[y,x]))
                    hisv850mean[y,x]=(hisv8501[y,x]+K[k]*(hisv8501[y-1,x]+hisv8501[y+1,x]-2*hisv8501[y,x]))
                    hismslmean[y,x]=hismsl1[y,x]+K[k]*(hismsl1[y-1,x]+hismsl1[y+1,x]-2*hismsl1[y,x])
            u8501=u850mean.copy()
            v8501=v850mean.copy()
            msl1=mslmean.copy()
            hisu8501=hisu850mean.copy()
            hisv8501=hisv850mean.copy()  
            hismsl1=hismslmean.copy()
            
    if d==2:
        u850sum=u850[[(time==(int(data[0][17])))]]
        v850sum=v850[[(time==(int(data[0][17])))]]
        mslsum=msl[[(time==(int(data[0][17])))]]
        d10=d1.u.loc[d1.time.dt.month.isin([int(data[0][15])])]
        d100=d10.loc[d10.time.dt.day.isin([int(data[0][16])])]
        dv10=d1.v.loc[d1.time.dt.month.isin([int(data[0][15])])]
        dv100=dv10.loc[dv10.time.dt.day.isin([int(data[0][16])])]
        hisu850sum=np.mean(d100,axis=0)
        hisv850sum=np.mean(dv100,axis=0)
        d20=d2.z.loc[d1.time.dt.month.isin([int(data[0][15])])]
        d200=d20.loc[d10.time.dt.day.isin([int(data[0][16])])]
        hismslsum=np.mean(d200/98.0665,axis=0)
        #print(list(hismslsum))
        for i in range(1,len(data)):
            u850sum+=u850[[(time==(int(data[i][17])))]]
            v850sum+=v850[[(time==(int(data[i][17])))]]
            mslsum+=msl[[(time==(int(data[i][17])))]]
            d10=d1.u.loc[d1.time.dt.month.isin([int(data[i][15])])]
            d100=d10.loc[d10.time.dt.day.isin([int(data[i][16])])]
            dv10=d1.v.loc[d1.time.dt.month.isin([int(data[0][15])])]
            dv100=dv10.loc[dv10.time.dt.day.isin([int(data[0][16])])]
            hisu850sum+=np.mean(d100,axis=0)
            hisv850sum+=np.mean(dv100,axis=0)
            d20=d2.z.loc[d1.time.dt.month.isin([int(data[i][15])])]
            d200=d20.loc[d10.time.dt.day.isin([int(data[i][16])])]
            hismslsum+=np.mean(d200/98.0665,axis=0)
            
        hisu8501=hisu850sum/len(data)
        hisv8501=hisv850sum/len(data)
        hismsl1=hismslsum/len(data)
            
        u8501=u850sum/len(data)
        v8501=v850sum/len(data)
        msl1=mslsum/len(data)

        hisu850mean=hisu8501.copy()
        hisv850mean=hisv8501.copy()  
        hismslmean=hismsl1.copy()

        u850mean=u8501.copy()
        v850mean=v8501.copy()  
        mslmean=msl1.copy()
        
        
        m=[2,3,4,2,5,6,7,2,8,9,2]
        K=[]
        for i in range(len(m)):
            K.append(1/2*((1-math.cos(2*math.pi/m[i]))**(-1)))
        for k in range(len(K)):
            for y in range(len(lat)):
                for x in range(1,len(lon)-1):
                    u850mean[:,y,x]=(u8501[:,y,x]+K[k]*(u8501[:,y,x-1]+u8501[:,y,x+1]-2*u8501[:,y,x]))
                    v850mean[:,y,x]=(v8501[:,y,x]+K[k]*(v8501[:,y,x-1]+v8501[:,y,x+1]-2*v8501[:,y,x]))
                    mslmean[:,y,x]=msl1[:,y,x]+K[k]*(msl1[:,y,x-1]+msl1[:,y,x+1]-2*msl1[:,y,x])
                    hisu850mean[y,x]=(hisu8501[y,x]+K[k]*(hisu8501[y,x-1]+hisu8501[y,x+1]-2*hisu8501[y,x]))
                    hisv850mean[y,x]=(hisv8501[y,x]+K[k]*(hisv8501[y,x-1]+hisv8501[y,x+1]-2*hisv8501[y,x]))
                    hismslmean[y,x]=hismsl1[y,x]+K[k]*(hismsl1[y,x-1]+hismsl1[y,x+1]-2*hismsl1[y,x])
              
            u8501=u850mean.copy()
            v8501=v850mean.copy()  
            msl1=mslmean.copy()
            hisu8501=hisu850mean.copy()
            hisv8501=hisv850mean.copy()  
            hismsl1=hismslmean.copy()
            print(msl1.shape)        
            for x in range(len(lon)):
                for y in range(1,len(lat)-1):
                    u850mean[:,y,x]=u8501[:,y,x]+K[k]*(u8501[:,y-1,x]+u8501[:,y+1,x]-2*u8501[:,y,x])
                    v850mean[:,y,x]=v8501[:,y,x]+K[k]*(v8501[:,y-1,x]+v8501[:,y+1,x]-2*v8501[:,y,x])
                    mslmean[:,y,x]=msl1[:,y,x]+K[k]*(msl1[:,y-1,x]+msl1[:,y+1,x]-2*msl1[:,y,x])
                    hisu850mean[y,x]=(hisu8501[y,x]+K[k]*(hisu8501[y-1,x]+hisu8501[y+1,x]-2*hisu8501[y,x]))
                    hisv850mean[y,x]=(hisv8501[y,x]+K[k]*(hisv8501[y-1,x]+hisv8501[y+1,x]-2*hisv8501[y,x]))
                    hismslmean[y,x]=hismsl1[y,x]+K[k]*(hismsl1[y-1,x]+hismsl1[y+1,x]-2*hismsl1[y,x])
            u8501=u850mean.copy()
            v8501=v850mean.copy()
            msl1=mslmean.copy()
            hisu8501=hisu850mean.copy()
            hisv8501=hisv850mean.copy()  
            hismsl1=hismslmean.copy()
            
    if d==0:
        u850sum=u850[[(time==(int(data[0][20])))]]
        v850sum=v850[[(time==(int(data[0][20])))]]
        mslsum=msl[[(time==(int(data[0][20])))]]
        d10=d1.u.loc[d1.time.dt.month.isin([int(data[0][18])])]
        d100=d10.loc[d10.time.dt.day.isin([int(data[0][19])])]
        dv10=d1.v.loc[d1.time.dt.month.isin([int(data[0][18])])]
        dv100=dv10.loc[dv10.time.dt.day.isin([int(data[0][19])])]
        hisu850sum=np.mean(d100,axis=0)
        hisv850sum=np.mean(dv100,axis=0)
        d20=d2.z.loc[d1.time.dt.month.isin([int(data[0][18])])]
        d200=d20.loc[d10.time.dt.day.isin([int(data[0][19])])]
        hismslsum=np.mean(d200/98.0665,axis=0)
        #print(list(hismslsum))
        for i in range(1,len(data)):
            u850sum+=u850[[(time==(int(data[i][20])))]]
            v850sum+=v850[[(time==(int(data[i][20])))]]
            mslsum+=msl[[(time==(int(data[i][20])))]]
            d10=d1.u.loc[d1.time.dt.month.isin([int(data[i][18])])]
            d100=d10.loc[d10.time.dt.day.isin([int(data[i][19])])]
            dv10=d1.v.loc[d1.time.dt.month.isin([int(data[0][18])])]
            dv100=dv10.loc[dv10.time.dt.day.isin([int(data[0][19])])]
            hisu850sum+=np.mean(d100,axis=0)
            hisv850sum+=np.mean(dv100,axis=0)
            d20=d2.z.loc[d1.time.dt.month.isin([int(data[i][18])])]
            d200=d20.loc[d10.time.dt.day.isin([int(data[i][19])])]
            hismslsum+=np.mean(d200/98.0665,axis=0)
       
        hisu8501=hisu850sum/len(data)
        hisv8501=hisv850sum/len(data)
        hismsl1=hismslsum/len(data)
            
        u8501=u850sum/len(data)
        v8501=v850sum/len(data)
        msl1=mslsum/len(data)

        hisu850mean=hisu8501.copy()
        hisv850mean=hisv8501.copy()  
        hismslmean=hismsl1.copy()

        u850mean=u8501.copy()
        v850mean=v8501.copy()  
        mslmean=msl1.copy()

        
        m=[2,3,4,2,5,6,7,2,8,9,2]
        K=[]
        for i in range(len(m)):
            K.append(1/2*((1-math.cos(2*math.pi/m[i]))**(-1)))
        for k in range(len(K)):
            for y in range(len(lat)):
                for x in range(1,len(lon)-1):
                    u850mean[:,y,x]=(u8501[:,y,x]+K[k]*(u8501[:,y,x-1]+u8501[:,y,x+1]-2*u8501[:,y,x]))
                    v850mean[:,y,x]=(v8501[:,y,x]+K[k]*(v8501[:,y,x-1]+v8501[:,y,x+1]-2*v8501[:,y,x]))
                    mslmean[:,y,x]=msl1[:,y,x]+K[k]*(msl1[:,y,x-1]+msl1[:,y,x+1]-2*msl1[:,y,x])
                    hisu850mean[y,x]=(hisu8501[y,x]+K[k]*(hisu8501[y,x-1]+hisu8501[y,x+1]-2*hisu8501[y,x]))
                    hisv850mean[y,x]=(hisv8501[y,x]+K[k]*(hisv8501[y,x-1]+hisv8501[y,x+1]-2*hisv8501[y,x]))
                    hismslmean[y,x]=hismsl1[y,x]+K[k]*(hismsl1[y,x-1]+hismsl1[y,x+1]-2*hismsl1[y,x])
              
            u8501=u850mean.copy()
            v8501=v850mean.copy()  
            msl1=mslmean.copy()
            hisu8501=hisu850mean.copy()
            hisv8501=hisv850mean.copy()  
            hismsl1=hismslmean.copy()
            print(k)        
            for x in range(len(lon)):
                for y in range(1,len(lat)-1):
                    u850mean[:,y,x]=u8501[:,y,x]+K[k]*(u8501[:,y-1,x]+u8501[:,y+1,x]-2*u8501[:,y,x])
                    v850mean[:,y,x]=v8501[:,y,x]+K[k]*(v8501[:,y-1,x]+v8501[:,y+1,x]-2*v8501[:,y,x])
                    mslmean[:,y,x]=msl1[:,y,x]+K[k]*(msl1[:,y-1,x]+msl1[:,y+1,x]-2*msl1[:,y,x])
                    hisu850mean[y,x]=(hisu8501[y,x]+K[k]*(hisu8501[y-1,x]+hisu8501[y+1,x]-2*hisu8501[y,x]))
                    hisv850mean[y,x]=(hisv8501[y,x]+K[k]*(hisv8501[y-1,x]+hisv8501[y+1,x]-2*hisv8501[y,x]))
                    hismslmean[y,x]=hismsl1[y,x]+K[k]*(hismsl1[y-1,x]+hismsl1[y+1,x]-2*hismsl1[y,x])
            u8501=u850mean.copy()
            v8501=v850mean.copy()
            msl1=mslmean.copy()
            hisu8501=hisu850mean.copy()
            hisv8501=hisv850mean.copy()  
            hismsl1=hismslmean.copy()
            
    u_all=(np.mean(u850mean,axis=0)-hisu850mean)
    v_all=(np.mean(v850mean,axis=0)-hisv850mean)
    msl_all=(np.mean(mslmean,axis=0)-hismslmean)
    print(v_all)
    levels = np.arange(-2 , 2 + 0.1,0.2)
    cb = ax.contourf(lon2d,lat2d,msl_all,levels=levels,cmap=cmaps.BlueDarkRed18 ,transform=ccrs.PlateCarree(),extend='both')
    #plt.contour(lon2d,lat2d,msl_all,[575],colors='w')
    cq = ax.quiver(lon2d[::3,::3],lat2d[::3,::3],u_all[::3,::3],v_all[::3,::3],color='k',
                      transform=ccrs.PlateCarree(),scale=60,width=0.0045,edgecolor='w',linewidth=0.2)
    ax.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,160,-80,-40])
    gl.ylocator = mticker.FixedLocator(np.arange(-15,60,15))
    gl.top_labels    = False
    gl.right_labels    = False
    if d in [0]:
        gl.left_labels    = True
        gl.bottom_labels    = False
    if d in [1]:
        gl.left_labels    = True
        gl.bottom_labels    = True
    if d in [2]:
        gl.left_labels    = False
        gl.bottom_labels    = False
    if d in [3]:
        gl.left_labels    = False
        gl.bottom_labels    = True
    ax.quiver(174.5-200,5,3,0,color='red',transform=ccrs.PlateCarree(),width=0.0045,scale=60,edgecolor='w',linewidth=0.2)
    ax.text(174.5-200,2,'3m/s',color='red',weight="bold",size=5)
    gl.ypadding=8
    gl.xpadding=8
    x=[]
    y=[]
    if d==3:
        for i in range(len(data)):
            x.append(data[i][6])
        x=[i/10 for i in x]
        for i in range(len(data)):
            y.append(data[i][5])
        y=[i/10 for i in y]
        ax.scatter(x,y,s=4,c='red',marker='.')
    x1=[]
    y1=[]
    for i in range(len(data1)):
        x1.append(data1[i][6])
    x1=[i/10 for i in x1]
    for i in range(len(data1)):
        y1.append(data1[i][5])
    y1=[i/10 for i in y1]
    ax.scatter(x1,y1,s=4,c='blue',marker='.')
    if d==3:
        position=fig.add_axes([0.24, 0.634, 0.3, 0.015])
        cbar=fig.colorbar(cb,cax=position,orientation='horizontal',ticks=np.arange(-2 , 2 + 0.001, 1),
                          aspect=20,shrink=0.5,pad=0.04)
        cbar.ax.tick_params(pad=0.08,length=0.5,width=0.7)
    ax.text(102-200,51.5,label[d])
    return cb,cq


circulation('/Users/fuzhenghang/Documents/ERA5/mtce/NA_locationTCson10.xlsx','/Users/fuzhenghang/Documents/ERA5/mtce/NA_m0.xlsx',ax[3],3)

circulation('/Users/fuzhenghang/Documents/ERA5/mtce/NA_locationTCson10.xlsx','/Users/fuzhenghang/Documents/ERA5/mtce/NA_m01.xlsx',ax[1],1)

circulation('/Users/fuzhenghang/Documents/ERA5/mtce/NA_locationTCson10.xlsx','/Users/fuzhenghang/Documents/ERA5/mtce/NA_m11.xlsx',ax[2],2)

circulation('/Users/fuzhenghang/Documents/ERA5/mtce/NA_locationTCson10.xlsx','/Users/fuzhenghang/Documents/ERA5/mtce/NA_m21.xlsx',ax[0],0)


plt.show()
    