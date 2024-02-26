#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 18:32:20 2024

@author: fuzhenghang
"""
# In[0]
import matplotlib.pyplot as plt###引入库包
import numpy as np
import matplotlib as mpl
import matplotlib.colors
from scipy.stats.mstats import ttest_ind
from netCDF4 import Dataset
import math

import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import cmaps

mtce1 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce1.npy')
mtce2 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce2.npy')
mtce3 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce3.npy')
mtce1c = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce1c.npy')
mtce2c = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce2c.npy')
mtce3c = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce3c.npy')

mtcesst1 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtcesst1.npy')
mtcesst2 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtcesst2.npy')
mtcesst3 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtcesst3.npy')
mtcesst1c = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce1sstc.npy')
mtcesst2c = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce2sstc.npy')
mtcesst3c = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/mtce3sstc.npy')

d1 = Dataset(r'/Users/fuzhenghang/Documents/ERA5/OLR_5month9_1979_2022.nc')
lat=d1.variables['latitude'][:] 
lon=d1.variables['longitude'][:]
#print(lat,lon)
mpl.rcParams["font.family"] = 'Arial'  
mpl.rcParams["mathtext.fontset"] = 'cm' 
mpl.rcParams["font.size"] = 8


f1 = 98
f2 = 123
f3 = 108

m1 = np.zeros((4*f1,181,360))
m2 = np.zeros((4*f2,181,360))
m3 = np.zeros((4*f3,181,360))
m1c = np.zeros((4*f1,181,360))
m2c = np.zeros((4*f2,181,360))
m3c = np.zeros((4*f3,181,360))

m=[2,3,4,2,5,6,7,2,8,9,2]
K=[]
for i in range(len(m)):
    K.append(1/2*((1-math.cos(2*math.pi/m[i]))**(-1)))
for k in range(len(K)):
    for y in range(181):
        for x in range(1,359):
            m1[:,y,x]=(mtce1[:,y,x]+K[k]*(mtce1[:,y,x-1]+mtce1[:,y,x+1]-2*mtce1[:,y,x]))
            m2[:,y,x]=(mtce2[:,y,x]+K[k]*(mtce2[:,y,x-1]+mtce2[:,y,x+1]-2*mtce2[:,y,x]))
            m3[:,y,x]=mtce3[:,y,x]+K[k]*(mtce3[:,y,x-1]+mtce3[:,y,x+1]-2*mtce3[:,y,x])
            m1c[:,y,x]=(mtce1c[:,y,x]+K[k]*(mtce1c[:,y,x-1]+mtce1c[:,y,x+1]-2*mtce1c[:,y,x]))
            m2c[:,y,x]=(mtce2c[:,y,x]+K[k]*(mtce2c[:,y,x-1]+mtce2c[:,y,x+1]-2*mtce2c[:,y,x]))
            m3c[:,y,x]=mtce3c[:,y,x]+K[k]*(mtce3c[:,y,x-1]+mtce3c[:,y,x+1]-2*mtce3c[:,y,x])
            
    mtce1=m1.copy()
    mtce2=m2.copy()
    mtce3=m3.copy()
    mtce1c=m1c.copy()
    mtce2c=m2c.copy()
    mtce3c=m3c.copy()
    print(mtce1[0,:,2])
    for x in range(360):
        for y in range(1,180):
            m1[:,y,x]=mtce1[:,y,x]+K[k]*(mtce1[:,y-1,x]+mtce1[:,y+1,x]-2*mtce1[:,y,x])
            m2[:,y,x]=mtce2[:,y,x]+K[k]*(mtce2[:,y-1,x]+mtce2[:,y+1,x]-2*mtce2[:,y,x])
            m3[:,y,x]=mtce3[:,y,x]+K[k]*(mtce3[:,y-1,x]+mtce3[:,y+1,x]-2*mtce3[:,y,x])
            m1c[:,y,x]=mtce1c[:,y,x]+K[k]*(mtce1c[:,y-1,x]+mtce1c[:,y+1,x]-2*mtce1c[:,y,x])
            m2c[:,y,x]=mtce2c[:,y,x]+K[k]*(mtce2c[:,y-1,x]+mtce2c[:,y+1,x]-2*mtce2c[:,y,x])
            m3c[:,y,x]=mtce3c[:,y,x]+K[k]*(mtce3c[:,y-1,x]+mtce3c[:,y+1,x]-2*mtce3c[:,y,x])
    mtce1=m1.copy()
    mtce2=m2.copy()
    mtce3=m3.copy()
    mtce1c=m1c.copy()
    mtce2c=m2c.copy()
    mtce3c=m3c.copy()    
    print(k)

# In[1]

uy1f=np.zeros((f1,181,360))
uy2f=np.zeros((f2,181,360))
uy3f=np.zeros((f3,181,360))

av1f=np.zeros((f1,181,360))
av2f=np.zeros((f2,181,360))
av3f=np.zeros((f3,181,360))


for i in range(1,180):
    for j in range(360):
        uy1f[:,i,j]=(mtce1[0:f1,i,j]-mtce1[0:f1,i+1,j])/110940
        uy2f[:,i,j]=(mtce2[0:f2,i,j]-mtce2[0:f2,i+1,j])/110940
        uy3f[:,i,j]=(mtce3[0:f3,i,j]-mtce3[0:f3,i+1,j])/110940
        av1f[:,i,j]=mtce1[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
        av2f[:,i,j]=mtce2[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
        av3f[:,i,j]=mtce3[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
    
DGPI1f=np.zeros((f1,181,360))
DGPI2f=np.zeros((f2,181,360))
DGPI3f=np.zeros((f3,181,360))

for i in range(1,180):
    for j in range(360):
        DGPI1f[:,i,j]=((2+0.1*mtce1[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1f[:,i,j]*100000)**(2.3))*((5.0-20*mtce1[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
        DGPI2f[:,i,j]=((2+0.1*mtce2[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2f[:,i,j]*100000)**(2.3))*((5.0-20*mtce2[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
        DGPI3f[:,i,j]=((2+0.1*mtce3[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3f[:,i,j]*100000)**(2.3))*((5.0-20*mtce3[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
        #print((5.5-uy1f[:,i,j]*100000)**(2.3))
        #print((5.5+abs(av1f[:,i,j]*100000))**(2.4))

uy1h=np.zeros((f1,181,360))
uy2h=np.zeros((f2,181,360))
uy3h=np.zeros((f3,181,360))

av1h=np.zeros((f1,181,360))
av2h=np.zeros((f2,181,360))
av3h=np.zeros((f3,181,360))


for i in range(1,180):
    for j in range(360):
        uy1h[:,i,j]=(mtce1c[0:f1,i,j]-mtce1c[0:f1,i+1,j])/110940
        uy2h[:,i,j]=(mtce2c[0:f2,i,j]-mtce2c[0:f2,i+1,j])/110940
        uy3h[:,i,j]=(mtce3c[0:f3,i,j]-mtce3c[0:f3,i+1,j])/110940
        av1h[:,i,j]=mtce1c[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
        av2h[:,i,j]=mtce2c[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
        av3h[:,i,j]=mtce3c[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)


DGPI1h=np.zeros((f1,181,360))
DGPI2h=np.zeros((f2,181,360))
DGPI3h=np.zeros((f3,181,360))

for i in range(1,180):
    for j in range(360):
        DGPI1h[:,i,j]=((2+0.1*mtce1c[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1h[:,i,j]*100000)**(2.3))*((5.0-20*mtce1c[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
        DGPI2h[:,i,j]=((2+0.1*mtce2c[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2h[:,i,j]*100000)**(2.3))*((5.0-20*mtce2c[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
        DGPI3h[:,i,j]=((2+0.1*mtce3c[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3h[:,i,j]*100000)**(2.3))*((5.0-20*mtce3c[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
   
for i in range(1,180):
    for j in range(360):
        for k1 in range(f1):
            if mtcesst1[k1,i,j]<26+273.15:
                DGPI1f[k1,i,j] = 0
            if mtcesst1c[k1,i,j]<26+273.15:
                DGPI1h[k1,i,j] = 0
        for k2 in range(f2):
            if mtcesst2[k2,i,j]<26+273.15:
                DGPI2f[k2,i,j] = 0
            if mtcesst2c[k2,i,j]<26+273.15:
                DGPI2h[k2,i,j] = 0
        for k3 in range(f3):
            if mtcesst3[k3,i,j]<26+273.15:
                DGPI3f[k3,i,j] = 0
            if mtcesst3c[k3,i,j]<26+273.15:
                DGPI3h[k3,i,j] = 0
                
def contris(mtce1,mtce1c,mtcesst1,mtce1sstc,mtce2,mtce2c,mtcesst2,mtce2sstc,mtce3,mtce3c,mtcesst3,mtce3sstc):
    uy1f=np.zeros((f1,181,360))
    uy2f=np.zeros((f2,181,360))
    uy3f=np.zeros((f3,181,360))

    av1f=np.zeros((f1,181,360))
    av2f=np.zeros((f2,181,360))
    av3f=np.zeros((f3,181,360))


    for i in range(1,180):
        for j in range(360):
            uy1f[:,i,j]=(mtce1c[0:f1,i,j]-mtce1c[0:f1,i+1,j])/110940
            uy2f[:,i,j]=(mtce2c[0:f2,i,j]-mtce2c[0:f2,i+1,j])/110940
            uy3f[:,i,j]=(mtce3c[0:f3,i,j]-mtce3c[0:f3,i+1,j])/110940
            av1f[:,i,j]=mtce1c[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av2f[:,i,j]=mtce2c[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av3f[:,i,j]=mtce3c[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
        
    DGPI1f=np.zeros((f1,181,360))
    DGPI2f=np.zeros((f2,181,360))
    DGPI3f=np.zeros((f3,181,360))

    for i in range(1,180):
        for j in range(360):
            DGPI1f[:,i,j]=((2+0.1*mtce1[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1f[:,i,j]*100000)**(2.3))*((5.0-20*mtce1c[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI2f[:,i,j]=((2+0.1*mtce2[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2f[:,i,j]*100000)**(2.3))*((5.0-20*mtce2c[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI3f[:,i,j]=((2+0.1*mtce3[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3f[:,i,j]*100000)**(2.3))*((5.0-20*mtce3c[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1


    uy1h=np.zeros((f1,181,360))
    uy2h=np.zeros((f2,181,360))
    uy3h=np.zeros((f3,181,360))

    av1h=np.zeros((f1,181,360))
    av2h=np.zeros((f2,181,360))
    av3h=np.zeros((f3,181,360))


    for i in range(1,180):
        for j in range(360):
            uy1h[:,i,j]=(mtce1c[0:f1,i,j]-mtce1c[0:f1,i+1,j])/110940
            uy2h[:,i,j]=(mtce2c[0:f2,i,j]-mtce2c[0:f2,i+1,j])/110940
            uy3h[:,i,j]=(mtce3c[0:f3,i,j]-mtce3c[0:f3,i+1,j])/110940
            av1h[:,i,j]=mtce1c[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av2h[:,i,j]=mtce2c[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av3h[:,i,j]=mtce3c[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)


    DGPI1h=np.zeros((f1,181,360))
    DGPI2h=np.zeros((f2,181,360))
    DGPI3h=np.zeros((f3,181,360))

    for i in range(1,180):
        for j in range(360):
            DGPI1h[:,i,j]=((2+0.1*mtce1c[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1h[:,i,j]*100000)**(2.3))*((5.0-20*mtce1c[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI2h[:,i,j]=((2+0.1*mtce2c[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2h[:,i,j]*100000)**(2.3))*((5.0-20*mtce2c[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI3h[:,i,j]=((2+0.1*mtce3c[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3h[:,i,j]*100000)**(2.3))*((5.0-20*mtce3c[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
       
    for i in range(1,180):
        for j in range(360):
            for k1 in range(f1):
                if mtcesst1[k1,i,j]<26+273.15:
                    DGPI1f[k1,i,j] = 0
                if mtcesst1c[k1,i,j]<26+273.15:
                    DGPI1h[k1,i,j] = 0
            for k2 in range(f2):
                if mtcesst2[k2,i,j]<26+273.15:
                    DGPI2f[k2,i,j] = 0
                if mtcesst2c[k2,i,j]<26+273.15:
                    DGPI2h[k2,i,j] = 0
            for k3 in range(f3):
                if mtcesst3[k3,i,j]<26+273.15:
                    DGPI3f[k3,i,j] = 0
                if mtcesst3c[k3,i,j]<26+273.15:
                    DGPI3h[k3,i,j] = 0
    return DGPI1h,DGPI1f,DGPI2h,DGPI2f,DGPI3h,DGPI3f
                
def contriu(mtce1,mtce1c,mtcesst1,mtce1sstc,mtce2,mtce2c,mtcesst2,mtce2sstc,mtce3,mtce3c,mtcesst3,mtce3sstc):
    uy1f=np.zeros((f1,181,360))
    uy2f=np.zeros((f2,181,360))
    uy3f=np.zeros((f3,181,360))

    av1f=np.zeros((f1,181,360))
    av2f=np.zeros((f2,181,360))
    av3f=np.zeros((f3,181,360))


    for i in range(1,180):
        for j in range(360):
            uy1f[:,i,j]=(mtce1[0:f1,i,j]-mtce1[0:f1,i+1,j])/110940
            uy2f[:,i,j]=(mtce2[0:f2,i,j]-mtce2[0:f2,i+1,j])/110940
            uy3f[:,i,j]=(mtce3[0:f3,i,j]-mtce3[0:f3,i+1,j])/110940
            av1f[:,i,j]=mtce1c[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av2f[:,i,j]=mtce2c[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av3f[:,i,j]=mtce3c[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
        
    DGPI1f=np.zeros((f1,181,360))
    DGPI2f=np.zeros((f2,181,360))
    DGPI3f=np.zeros((f3,181,360))

    for i in range(1,180):
        for j in range(360):
            DGPI1f[:,i,j]=((2+0.1*mtce1c[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1f[:,i,j]*100000)**(2.3))*((5.0-20*mtce1c[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI2f[:,i,j]=((2+0.1*mtce2c[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2f[:,i,j]*100000)**(2.3))*((5.0-20*mtce2c[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI3f[:,i,j]=((2+0.1*mtce3c[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3f[:,i,j]*100000)**(2.3))*((5.0-20*mtce3c[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1


    uy1h=np.zeros((f1,181,360))
    uy2h=np.zeros((f2,181,360))
    uy3h=np.zeros((f3,181,360))

    av1h=np.zeros((f1,181,360))
    av2h=np.zeros((f2,181,360))
    av3h=np.zeros((f3,181,360))


    for i in range(1,180):
        for j in range(360):
            uy1h[:,i,j]=(mtce1c[0:f1,i,j]-mtce1c[0:f1,i+1,j])/110940
            uy2h[:,i,j]=(mtce2c[0:f2,i,j]-mtce2c[0:f2,i+1,j])/110940
            uy3h[:,i,j]=(mtce3c[0:f3,i,j]-mtce3c[0:f3,i+1,j])/110940
            av1h[:,i,j]=mtce1c[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av2h[:,i,j]=mtce2c[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av3h[:,i,j]=mtce3c[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)


    DGPI1h=np.zeros((f1,181,360))
    DGPI2h=np.zeros((f2,181,360))
    DGPI3h=np.zeros((f3,181,360))

    for i in range(1,180):
        for j in range(360):
            DGPI1h[:,i,j]=((2+0.1*mtce1c[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1h[:,i,j]*100000)**(2.3))*((5.0-20*mtce1c[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI2h[:,i,j]=((2+0.1*mtce2c[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2h[:,i,j]*100000)**(2.3))*((5.0-20*mtce2c[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI3h[:,i,j]=((2+0.1*mtce3c[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3h[:,i,j]*100000)**(2.3))*((5.0-20*mtce3c[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
       
    for i in range(1,180):
        for j in range(360):
            for k1 in range(f1):
                if mtcesst1[k1,i,j]<26+273.15:
                    DGPI1f[k1,i,j] = 0
                if mtcesst1c[k1,i,j]<26+273.15:
                    DGPI1h[k1,i,j] = 0
            for k2 in range(f2):
                if mtcesst2[k2,i,j]<26+273.15:
                    DGPI2f[k2,i,j] = 0
                if mtcesst2c[k2,i,j]<26+273.15:
                    DGPI2h[k2,i,j] = 0
            for k3 in range(f3):
                if mtcesst3[k3,i,j]<26+273.15:
                    DGPI3f[k3,i,j] = 0
                if mtcesst3c[k3,i,j]<26+273.15:
                    DGPI3h[k3,i,j] = 0
    return DGPI1h,DGPI1f,DGPI2h,DGPI2f,DGPI3h,DGPI3f
                
def contriw(mtce1,mtce1c,mtcesst1,mtce1sstc,mtce2,mtce2c,mtcesst2,mtce2sstc,mtce3,mtce3c,mtcesst3,mtce3sstc):
    uy1f=np.zeros((f1,181,360))
    uy2f=np.zeros((f2,181,360))
    uy3f=np.zeros((f3,181,360))

    av1f=np.zeros((f1,181,360))
    av2f=np.zeros((f2,181,360))
    av3f=np.zeros((f3,181,360))


    for i in range(1,180):
        for j in range(360):
            uy1f[:,i,j]=(mtce1c[0:f1,i,j]-mtce1c[0:f1,i+1,j])/110940
            uy2f[:,i,j]=(mtce2c[0:f2,i,j]-mtce2c[0:f2,i+1,j])/110940
            uy3f[:,i,j]=(mtce3c[0:f3,i,j]-mtce3c[0:f3,i+1,j])/110940
            av1f[:,i,j]=mtce1c[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av2f[:,i,j]=mtce2c[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av3f[:,i,j]=mtce3c[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
        
    DGPI1f=np.zeros((f1,181,360))
    DGPI2f=np.zeros((f2,181,360))
    DGPI3f=np.zeros((f3,181,360))

    for i in range(1,180):
        for j in range(360):
            DGPI1f[:,i,j]=((2+0.1*mtce1c[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1f[:,i,j]*100000)**(2.3))*((5.0-20*mtce1[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI2f[:,i,j]=((2+0.1*mtce2c[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2f[:,i,j]*100000)**(2.3))*((5.0-20*mtce2[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI3f[:,i,j]=((2+0.1*mtce3c[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3f[:,i,j]*100000)**(2.3))*((5.0-20*mtce3[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1


    uy1h=np.zeros((f1,181,360))
    uy2h=np.zeros((f2,181,360))
    uy3h=np.zeros((f3,181,360))

    av1h=np.zeros((f1,181,360))
    av2h=np.zeros((f2,181,360))
    av3h=np.zeros((f3,181,360))


    for i in range(1,180):
        for j in range(360):
            uy1h[:,i,j]=(mtce1c[0:f1,i,j]-mtce1c[0:f1,i+1,j])/110940
            uy2h[:,i,j]=(mtce2c[0:f2,i,j]-mtce2c[0:f2,i+1,j])/110940
            uy3h[:,i,j]=(mtce3c[0:f3,i,j]-mtce3c[0:f3,i+1,j])/110940
            av1h[:,i,j]=mtce1c[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av2h[:,i,j]=mtce2c[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av3h[:,i,j]=mtce3c[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)


    DGPI1h=np.zeros((f1,181,360))
    DGPI2h=np.zeros((f2,181,360))
    DGPI3h=np.zeros((f3,181,360))

    for i in range(1,180):
        for j in range(360):
            DGPI1h[:,i,j]=((2+0.1*mtce1c[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1h[:,i,j]*100000)**(2.3))*((5.0-20*mtce1c[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI2h[:,i,j]=((2+0.1*mtce2c[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2h[:,i,j]*100000)**(2.3))*((5.0-20*mtce2c[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI3h[:,i,j]=((2+0.1*mtce3c[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3h[:,i,j]*100000)**(2.3))*((5.0-20*mtce3c[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
       
    for i in range(1,180):
        for j in range(360):
            for k1 in range(f1):
                if mtcesst1[k1,i,j]<26+273.15:
                    DGPI1f[k1,i,j] = 0
                if mtcesst1c[k1,i,j]<26+273.15:
                    DGPI1h[k1,i,j] = 0
            for k2 in range(f2):
                if mtcesst2[k2,i,j]<26+273.15:
                    DGPI2f[k2,i,j] = 0
                if mtcesst2c[k2,i,j]<26+273.15:
                    DGPI2h[k2,i,j] = 0
            for k3 in range(f3):
                if mtcesst3[k3,i,j]<26+273.15:
                    DGPI3f[k3,i,j] = 0
                if mtcesst3c[k3,i,j]<26+273.15:
                    DGPI3h[k3,i,j] = 0
    return DGPI1h,DGPI1f,DGPI2h,DGPI2f,DGPI3h,DGPI3f
                
def contriv(mtce1,mtce1c,mtcesst1,mtce1sstc,mtce2,mtce2c,mtcesst2,mtce2sstc,mtce3,mtce3c,mtcesst3,mtce3sstc):
    uy1f=np.zeros((f1,181,360))
    uy2f=np.zeros((f2,181,360))
    uy3f=np.zeros((f3,181,360))

    av1f=np.zeros((f1,181,360))
    av2f=np.zeros((f2,181,360))
    av3f=np.zeros((f3,181,360))


    for i in range(1,180):
        for j in range(360):
            uy1f[:,i,j]=(mtce1c[0:f1,i,j]-mtce1c[0:f1,i+1,j])/110940
            uy2f[:,i,j]=(mtce2c[0:f2,i,j]-mtce2c[0:f2,i+1,j])/110940
            uy3f[:,i,j]=(mtce3c[0:f3,i,j]-mtce3c[0:f3,i+1,j])/110940
            av1f[:,i,j]=mtce1[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av2f[:,i,j]=mtce2[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av3f[:,i,j]=mtce3[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
        
    DGPI1f=np.zeros((f1,181,360))
    DGPI2f=np.zeros((f2,181,360))
    DGPI3f=np.zeros((f3,181,360))

    for i in range(1,180):
        for j in range(360):
            DGPI1f[:,i,j]=((2+0.1*mtce1c[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1f[:,i,j]*100000)**(2.3))*((5.0-20*mtce1c[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI2f[:,i,j]=((2+0.1*mtce2c[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2f[:,i,j]*100000)**(2.3))*((5.0-20*mtce2c[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI3f[:,i,j]=((2+0.1*mtce3c[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3f[:,i,j]*100000)**(2.3))*((5.0-20*mtce3c[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3f[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1


    uy1h=np.zeros((f1,181,360))
    uy2h=np.zeros((f2,181,360))
    uy3h=np.zeros((f3,181,360))

    av1h=np.zeros((f1,181,360))
    av2h=np.zeros((f2,181,360))
    av3h=np.zeros((f3,181,360))


    for i in range(1,180):
        for j in range(360):
            uy1h[:,i,j]=(mtce1c[0:f1,i,j]-mtce1c[0:f1,i+1,j])/110940
            uy2h[:,i,j]=(mtce2c[0:f2,i,j]-mtce2c[0:f2,i+1,j])/110940
            uy3h[:,i,j]=(mtce3c[0:f3,i,j]-mtce3c[0:f3,i+1,j])/110940
            av1h[:,i,j]=mtce1c[f1*2:f1*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av2h[:,i,j]=mtce2c[f2*2:f2*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)
            av3h[:,i,j]=mtce3c[f3*2:f3*3,i,j]+2*7.292*0.00001*np.sin((90-i)/180*np.pi)


    DGPI1h=np.zeros((f1,181,360))
    DGPI2h=np.zeros((f2,181,360))
    DGPI3h=np.zeros((f3,181,360))

    for i in range(1,180):
        for j in range(360):
            DGPI1h[:,i,j]=((2+0.1*mtce1c[f1*3:f1*4,i,j])**(-1.7))*((5.5-uy1h[:,i,j]*100000)**(2.3))*((5.0-20*mtce1c[f1:f1*2,i,j])**(3.4))*((5.5+abs(av1h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI2h[:,i,j]=((2+0.1*mtce2c[f2*3:f2*4,i,j])**(-1.7))*((5.5-uy2h[:,i,j]*100000)**(2.3))*((5.0-20*mtce2c[f2:f2*2,i,j])**(3.4))*((5.5+abs(av2h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
            DGPI3h[:,i,j]=((2+0.1*mtce3c[f3*3:f3*4,i,j])**(-1.7))*((5.5-uy3h[:,i,j]*100000)**(2.3))*((5.0-20*mtce3c[f3:f3*2,i,j])**(3.4))*((5.5+abs(av3h[:,i,j]*100000))**(2.4))*np.exp(-11.8)-1
       
    for i in range(1,180):
        for j in range(360):
            for k1 in range(f1):
                if mtcesst1[k1,i,j]<26+273.15:
                    DGPI1f[k1,i,j] = 0
                if mtcesst1c[k1,i,j]<26+273.15:
                    DGPI1h[k1,i,j] = 0
            for k2 in range(f2):
                if mtcesst2[k2,i,j]<26+273.15:
                    DGPI2f[k2,i,j] = 0
                if mtcesst2c[k2,i,j]<26+273.15:
                    DGPI2h[k2,i,j] = 0
            for k3 in range(f3):
                if mtcesst3[k3,i,j]<26+273.15:
                    DGPI3f[k3,i,j] = 0
                if mtcesst3c[k3,i,j]<26+273.15:
                    DGPI3h[k3,i,j] = 0
    return DGPI1h,DGPI1f,DGPI2h,DGPI2f,DGPI3h,DGPI3f
sDGPI1h,sDGPI1f,sDGPI2h,sDGPI2f,sDGPI3h,sDGPI3f = contris(mtce1,mtce1c,mtcesst1,mtcesst1c,mtce2,mtce2c,mtcesst2,mtcesst1c,mtce3,mtce3c,mtcesst3,mtcesst3c)
uDGPI1h,uDGPI1f,uDGPI2h,uDGPI2f,uDGPI3h,uDGPI3f = contriu(mtce1,mtce1c,mtcesst1,mtcesst1c,mtce2,mtce2c,mtcesst2,mtcesst1c,mtce3,mtce3c,mtcesst3,mtcesst3c)
wDGPI1h,wDGPI1f,wDGPI2h,wDGPI2f,wDGPI3h,wDGPI3f = contriw(mtce1,mtce1c,mtcesst1,mtcesst1c,mtce2,mtce2c,mtcesst2,mtcesst1c,mtce3,mtce3c,mtcesst3,mtcesst3c)
vDGPI1h,vDGPI1f,vDGPI2h,vDGPI2f,vDGPI3h,vDGPI3f = contriv(mtce1,mtce1c,mtcesst1,mtcesst1c,mtce2,mtce2c,mtcesst2,mtcesst1c,mtce3,mtce3c,mtcesst3,mtcesst3c)
# In[2]     
proj = ccrs.PlateCarree(central_longitude=180)  #中国为左
leftlon, rightlon, lowerlat, upperlat = (100,180,5,40)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
fig = plt.figure(figsize=(10,6),dpi=800)  
#print(lon)
#print(lat)
ax=[]
xx=[0.05,0.35,0.65,0.05,0.35,0.65]
yy=[0.6,0.6,0.6,0.35,0.35,0.35]
dx=[0.28,0.28,0.28,0.28,0.28,0.28]
dy=0.2

for i in range(3):
    ax.append(fig.add_axes([xx[i],yy[i],dx[i],dy],projection = proj))
for i in range(3,6):
    ax.append(fig.add_axes([xx[i],yy[i],dx[i],dy]))        

label = ['(a) EI: DGPI','(b) WI-N: DGPI','(c) WI-O: DGPI','(d) EI','(e) WI-N','(f) WI-O']
DGPIh = [DGPI1h,DGPI2h,DGPI3h]
DGPIf = [DGPI1f,DGPI2f,DGPI3f]
data1 = [[110,8,170,25],[110,8,150,25],[120,15,170,30]]

sDGPIf = [sDGPI1f,sDGPI2f,sDGPI3f]
uDGPIf = [uDGPI1f,uDGPI2f,uDGPI3f]
wDGPIf = [wDGPI1f,wDGPI2f,wDGPI3f]
vDGPIf = [vDGPI1f,vDGPI2f,vDGPI3f]
data2 = [DGPIf,sDGPIf,wDGPIf,vDGPIf,uDGPIf]
contri1 = np.zeros((5,98))
contri2 = np.zeros((5,123))
contri3 = np.zeros((5,108))
for i in range(3):
    for j in range(5):
        contri1[j] = np.nanmean(data2[j][0][:,65:82,110:170],axis = (1,2))-np.nanmean(DGPIh[0][:,65:82,110:170],axis = (1,2))
        contri2[j] = np.nanmean(data2[j][1][:,65:82,110:150],axis = (1,2))-np.nanmean(DGPIh[1][:,65:82,110:150],axis = (1,2))
        contri3[j] = np.nanmean(data2[j][2][:,60:75,120:170],axis = (1,2))-np.nanmean(DGPIh[2][:,60:75,120:170],axis = (1,2))
for i in range(3):
    levels = np.arange(-18,18+0.1,2)
    a = np.nanmean(DGPIf[i]-DGPIh[i],axis = 0)
    cb=ax[i].contourf(lon,lat,a,levels=levels,cmap=cmaps.BlueDarkRed18,transform=ccrs.PlateCarree(),extend='both')
    ax[i].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax[i].add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.4,zorder = 3)
    ax[i].add_feature(cfeature.LAND.with_scale('50m'),color = 'lightgrey',zorder = 2)
    gl = ax[i].gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,140,160,180])
    gl.ylocator = mticker.FixedLocator(np.arange(-15,60,15))
    gl.top_labels    = False
    gl.right_labels    = False
    gl.bottom_labels    = True
    gl.left_labels    = True
    gl.xpadding = 2
    gl.ypadding = 2
    if i != 0:
        gl.left_labels    = False
    _,p1 = ttest_ind(DGPIh[i],DGPIf[i],equal_var=False)
    for la in range(50,85,2):
        for lo in range(100,180,2):
            if p1[la,lo]<=0.05:
                ax[i].text(lo-180,90-la,'.',zorder = 1)
    
    lon1 = np.empty(4)
    lat1 = np.empty(4)
    lon1[0],lat1[0] = data1[i][0], data1[i][1]  # lower left (ll)
    lon1[1],lat1[1] = data1[i][2], data1[i][1]  # lower right (lr)
    lon1[2],lat1[2] = data1[i][2], data1[i][3]  # upper right (ur)
    lon1[3],lat1[3] = data1[i][0], data1[i][3]  # upper left (ul)
    x, y =  lon1, lat1
    xy = list(zip(x,y))
    poly = plt.Polygon(xy,edgecolor="k",linestyle='-',fc="none", lw=0.5, alpha=1,transform=ccrs.PlateCarree(),zorder=7)
    ax[i].add_patch(poly)
    ax[i].text(-78.8,41.5,label[i])
    if i == 2:
        position=fig.add_axes([0.933, 0.6, 0.01, 0.2])
        cbar=plt.colorbar(cb,cax=position,orientation='vertical',ticks=np.arange(-18,18.001,6),
                         aspect=20,shrink=0.2,pad=0.06)
x_data = ['Total','VWS','Omega','Vor','-$\partial u $/$\partial y $']

con = [contri1,contri2,contri3]
for i in range(3,6):
    y_data = [np.nanmean(con[i-3][m]) for m in range(5)]
    bb=ax[i].bar(x_data, y_data,width=0.4)
    ax[i].set_ylim(-2,12)
    ax[i].set_yticks([0,3,6,9,12])
    for bar,height in zip(bb,y_data):
        if height<0:
            bar.set(color='royalblue')
        elif height>0:
            bar.set(color='tomato') 
    ymedian =np.zeros((3,5))
    spread = np.zeros((3,2,5))
    for c in range(3):
        for t in range(5):
            ymedian[c,t] = np.percentile(con[c][t],50)
            spread[c,0,t] = np.percentile(con[c][t],50) - np.percentile(con[c][t],25)
            spread[c,1,t] = np.percentile(con[c][t],75) - np.percentile(con[c][t],50)
    ax[i].axhline(y=0, color='black', linestyle='-',linewidth = 0.5)
    ax[i].text(-0.3,12.5,label[i])
    ax[i].errorbar(x_data, ymedian[i-3],fmt='.',mec='w',mew=0.6,ms=6,mfc='k',yerr = spread[i-3],c='k',linewidth=0.8,capsize=2.5)
    if i != 3:
        ax[i].set_yticklabels([])
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/dgpi.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)
   
        
        