#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:20:46 2024

@author: fuzhenghang
"""


import matplotlib.pyplot as plt###引入库包
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import xlrd


d8 = Dataset(r'/DS1/xshome/fuzh19/nc/erasst.nc')  
time8 = d8.variables['time'][:]
sst=d8["sst"][:]


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

    
   
    sstdatac = np.zeros((len(data),181,360))
    
  
    sstdata = np.zeros((len(data),181,360))
    
    for i in range(0,len(data)):
       
        sstdata[i]=sst[time8==(int(data[i][11]))]
      
        d80=d8.sst.loc[d8.time.dt.month.isin([int(data[i][1])])]
        d800=d80.loc[d80.time.dt.day.isin([int(data[i][2])])]
        sstdatac[i] = np.mean(d800,axis=0)
        

    return sstdatac,sstdata


a = ['/DS1/xshome/fuzh19/mtce/locationTCson10.xlsx',
     '/DS1/xshome/fuzh19/mtce/locationTCson11.xlsx',
     '/DS1/xshome/fuzh19/mtce/locationTCson12.xlsx']
b = ['/DS1/xshome/fuzh19/mtce/m0.xlsx',
     '/DS1/xshome/fuzh19/mtce/m1.xlsx',
     '/DS1/xshome/fuzh19//mtce/m2.xlsx']
mtce1 = np.zeros((98,181,360))
mtce2 = np.zeros((123,181,360))
mtce3 = np.zeros((108,181,360))
mtce1c = np.zeros((98,181,360))
mtce2c = np.zeros((123,181,360))
mtce3c = np.zeros((108,181,360))
mtce1c[0:98],mtce1[0:98] = dgpi(a[0],b[0],0)
mtce2c[0:123],mtce2[0:123] = dgpi(a[1],b[1],1)
mtce3c[0:108],mtce3[0:108] = dgpi(a[2],b[2],2)
np.save('./mtcesst1.npy',mtce1)
np.save('./mtcesst2.npy',mtce2)
np.save('./mtcesst3.npy',mtce3)
np.save('./mtce1sstc.npy',mtce1c)
np.save('./mtce2sstc.npy',mtce2c)
np.save('./mtce3sstc.npy',mtce3c)
    