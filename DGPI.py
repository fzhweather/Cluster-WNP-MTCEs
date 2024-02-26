#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:09:24 2024

@author: fuzhenghang
"""


import matplotlib.pyplot as plt###引入库包
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import xlrd

input_data = r'/DS1/xshome/fuzh19/nc/500u.nc'
d1 = Dataset(input_data)
                                        
d2 = Dataset(r'/DS1/xshome/fuzh19/nc/500w.nc')   
input_data = r'/DS1/xshome/fuzh19/nc/850v.nc'
d3 = Dataset(input_data)
vor850=d3.variables["vo"][:]                      
msl = d2.variables["w"][:]
lat = d1.variables['latitude'][:]
lon = d1.variables['longitude'][:]
time = d1.variables['time'][:]
time2 = d3.variables['time'][:]
time4 = d2.variables['time'][:]
u500 = d1.variables["u"][:]
lon2d, lat2d = np.meshgrid(lon, lat)

input_data = r'/DS1/xshome/fuzh19/nc//850wdu.nc'
d4 = Dataset(input_data)
d5 = Dataset(r'/DS1/xshome/fuzh19/nc/850wdv.nc')   
d6 = Dataset(r'/DS1/xshome/fuzh19/nc/200wdu.nc')  
d7 = Dataset(r'/DS1/xshome/fuzh19/nc/200wdv.nc')  
d8 = Dataset(r'/DS1/xshome/fuzh19/nc/erasst.nc')  

u850=d4["u"][:]
v850=d5["v"][:]
u200=d6["u"][:]
v200=d7["v"][:]
time3 = d4.variables['time'][:]
time8 = d8.variables['time'][:]
sst=d8["sst"][:]

d1=xr.open_dataset('/DS1/xshome/fuzh19/nc/500u.nc')
d2=xr.open_dataset('/DS1/xshome/fuzh19/nc/500w.nc')
d3=xr.open_dataset('/DS1/xshome/fuzh19/nc/850v.nc')
d4 = xr.open_dataset('/DS1/xshome/fuzh19/nc//850wdu.nc')
d5 = xr.open_dataset('/DS1/xshome/fuzh19/nc/850wdv.nc')   
d6 = xr.open_dataset('/DS1/xshome/fuzh19/nc/200wdu.nc')  
d7 = xr.open_dataset('/DS1/xshome/fuzh19/nc/200wdv.nc')  
d8 = xr.open_dataset('/DS1/xshome/fuzh19/nc/erasst.nc') 
def dgpi(a,b,m):
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

    
    u500datac = np.zeros((len(data),181,360))
    w500datac = np.zeros((len(data),181,360))
    vor850datac = np.zeros((len(data),181,360))
    u850datac = np.zeros((len(data),181,360))
    u200datac = np.zeros((len(data),181,360))
    v850datac = np.zeros((len(data),181,360))
    v200datac = np.zeros((len(data),181,360))
    shdatac = np.zeros((len(data),181,360))
    sstdatac = np.zeros((len(data),181,360))
    
    u500data = np.zeros((len(data),181,360))
    w500data = np.zeros((len(data),181,360))
    vor850data = np.zeros((len(data),181,360))
    u850data = np.zeros((len(data),181,360))
    u200data = np.zeros((len(data),181,360))
    v850data = np.zeros((len(data),181,360))
    v200data = np.zeros((len(data),181,360))
    shdata = np.zeros((len(data),181,360))
    sstdata = np.zeros((len(data),181,360))
    
    for i in range(0,len(data)):
        u500data[i]=u500[time==(int(data[i][11]))]
        w500data[i]=msl[(time4==(int(data[i][11])))]
        vor850data[i]=vor850[time2==(int(data[i][11]))]
        u850data[i]=u850[time3==(int(data[i][11]))]
        u200data[i]=u200[time3==(int(data[i][11]))]
        v850data[i]=v850[time3==(int(data[i][11]))]
        v200data[i]=v200[time3==(int(data[i][11]))]
        temp =(u200data[i]**2+v200data[i]**2)**0.5-(u850data[i]**2+v850data[i]**2)**0.5
        shdata[i] = temp[0]
        sstdata[i]=sst[time8==(int(data[i][11]))]
        
        d10=d1.u.loc[d1.time.dt.month.isin([int(data[i][1])])]
        d100=d10.loc[d10.time.dt.day.isin([int(data[i][2])])]
        u500datac[i]=np.mean(d100,axis=0)
        
        d20=d2.w.loc[d2.time.dt.month.isin([int(data[i][1])])]
        d200=d20.loc[d20.time.dt.day.isin([int(data[i][2])])]
        w500datac[i] = np.mean(d200,axis=0)
        
        d30=d3.vo.loc[d3.time.dt.month.isin([int(data[i][1])])]
        d300=d30.loc[d30.time.dt.day.isin([int(data[i][2])])]
        vor850datac[i] = np.mean(d300,axis=0)
        
        d40=d4.u.loc[d4.time.dt.month.isin([int(data[i][1])])]
        d400=d40.loc[d40.time.dt.day.isin([int(data[i][2])])]
        u850datac[i] = np.mean(d400,axis=0)
        
        d50=d5.v.loc[d5.time.dt.month.isin([int(data[i][1])])]
        d500=d50.loc[d50.time.dt.day.isin([int(data[i][2])])]
        v850datac[i] = np.mean(d500,axis=0)
        
        d60=d6.u.loc[d6.time.dt.month.isin([int(data[i][1])])]
        d600=d60.loc[d60.time.dt.day.isin([int(data[i][2])])]
        u200datac[i] = np.mean(d600,axis=0)
        
        d70=d7.v.loc[d7.time.dt.month.isin([int(data[i][1])])]
        d700=d70.loc[d70.time.dt.day.isin([int(data[i][2])])]
        v200datac[i] = np.mean(d700,axis=0)
        temp =(u200datac[i]**2+v200datac[i]**2)**0.5-(u850datac[i]**2+v850datac[i]**2)**0.5
        shdatac[i] = temp[0]
        
        d80=d8.sst.loc[d8.time.dt.month.isin([int(data[i][1])])]
        d800=d80.loc[d80.time.dt.day.isin([int(data[i][2])])]
        sstdatac[i] = np.mean(d800,axis=0)
        

    return u500datac,u500data,w500datac,w500data,vor850datac,vor850data,shdatac,shdata


a = ['/DS1/xshome/fuzh19/mtce/locationTCson10.xlsx',
     '/DS1/xshome/fuzh19/mtce/locationTCson11.xlsx',
     '/DS1/xshome/fuzh19/mtce/locationTCson12.xlsx']
b = ['/DS1/xshome/fuzh19/mtce/m0.xlsx',
     '/DS1/xshome/fuzh19/mtce/m1.xlsx',
     '/DS1/xshome/fuzh19//mtce/m2.xlsx']
mtce1 = np.zeros((4*98,181,360))
mtce2 = np.zeros((4*123,181,360))
mtce3 = np.zeros((4*108,181,360))
mtce1c = np.zeros((4*98,181,360))
mtce2c = np.zeros((4*123,181,360))
mtce3c = np.zeros((4*108,181,360))
mtce1c[0:98],mtce1[0:98],mtce1c[98:98*2],mtce1[98:98*2],mtce1c[98*2:98*3],mtce1[98*2:98*3],mtce1c[98*3:98*4],mtce1[98*3:98*4] = dgpi(a[0],b[0],0)
mtce2c[0:123],mtce2[0:123],mtce2c[123:123*2],mtce2[123:123*2],mtce2c[123*2:123*3],mtce2[123*2:123*3],mtce2c[123*3:123*4],mtce2[123*3:123*4] = dgpi(a[1],b[1],1)
mtce3c[0:108],mtce3[0:108],mtce3c[108:108*2],mtce3[108:108*2],mtce3c[108*2:108*3],mtce3[108*2:108*3],mtce3c[108*3:108*4],mtce3[108*3:108*4] = dgpi(a[2],b[2],2)
np.save('./mtce1.npy',mtce1)
np.save('./mtce2.npy',mtce2)
np.save('./mtce3.npy',mtce3)
np.save('./mtce1c.npy',mtce1c)
np.save('./mtce2c.npy',mtce2c)
np.save('./mtce3c.npy',mtce3c)
    