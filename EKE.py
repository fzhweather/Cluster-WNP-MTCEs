#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:39:15 2023

@author: fuzhenghang
"""
# In[0]

import matplotlib.pyplot as plt###引入库包
import numpy as np
import matplotlib as mpl
import matplotlib.colors
from netCDF4 import Dataset
import xlrd
from scipy.stats.mstats import ttest_ind
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.reset_orig()

mpl.rcParams["font.family"] = 'Airal'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 1


cmap=plt.get_cmap(cmaps.GMT_wysiwyg)
newcolors=cmap(np.linspace(0, 1, 20))
newcmap = ListedColormap(newcolors[8:20])


d1 = Dataset(r'/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/850u_daily_mean.nc')
d2 = Dataset(r'/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/850v_daily_mean.nc') 
time=d1.variables['time'][:]
u850=d1.variables["u"][:]
v850=d2.variables["v"][:]
lat=d1.variables['latitude'][:] 
lon=d1.variables['longitude'][:]
# In[0]
mpl.rcParams["font.family"] = 'Airal' 
data1=[]
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKESTC.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data1.append(table.row_values(i))
data1 = [data1[i] for i in range(0,len(data1))]
proj = ccrs.PlateCarree(central_longitude=180)  #中国为左
leftlon, rightlon, lowerlat, upperlat = (100,180,0,50)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
fig = plt.figure(figsize=(10,6),dpi=800)  
x1 = [0.05,0.05,0.32,0.32,0.59,0.59]
yy = [0.9,0.6,0.9,0.6,0.9,0.6]
dx = 0.5
dy = 0.25
ax = []
label = ['(a) EI: EKE','(b) EI: KmKe','(c) WI-N: EKE','(d) WI-N: KmKe','(e) WI-O: EKE','(f) WI-O: KmKe']
for i in range(6):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
    ax[i].text(-78,51.5,label[i])


def eke(a,ax,b):
    data=[]
    table=xlrd.open_workbook(a)#'/Users/fuzhenghang/Documents/ERA5/mtce/EKEson.xlsx'
    table=table.sheets()[0]
    nrows=table.nrows
    for i in range(nrows):
        if i ==0:
            continue
        data.append(table.row_values(i))
    data = [data[i] for i in range(0,len(data))]
    
    umdr=np.zeros((len(data),181,360))
    vmdr=np.zeros((len(data),181,360))
    umr=np.zeros((11,len(data),181,360))
    vmr=np.zeros((11,len(data),181,360))

    eke1=np.zeros((len(data),181,360))


    for i in range(len(data)):
        for n in range(11):
            for r in range(11):
                umr[n,i,:,:]+=u850[time==(int(data[i][13])-(10-r-n)*24)][0,:,:]
                vmr[n,i,:,:]+=v850[time==(int(data[i][13])-(10-r-n)*24)][0,:,:]
            umr[n,i,:,:]=umr[n,i,:,:]/11
            vmr[n,i,:,:]=vmr[n,i,:,:]/11
            umdr[i,:,:]+=(u850[time==(int(data[i][13])-(5-n)*24)][0,:,:]-umr[n,i,:,:])**2
            vmdr[i,:,:]+=(v850[time==(int(data[i][13])-(5-n)*24)][0,:,:]-vmr[n,i,:,:])**2
        #print(vmdr[i,:,:])
        umdr[i,:,:]=umdr[i,:,:]/11
        vmdr[i,:,:]=vmdr[i,:,:]/11
        eke1[i,:,:]=(umdr[i,:,:]+vmdr[i,:,:])/2


    umdsr=np.zeros((len(data1),181,360))
    vmdsr=np.zeros((len(data1),181,360))
    umrs=np.zeros((11,len(data1),181,360))
    vmrs=np.zeros((11,len(data1),181,360))
    eke2=np.zeros((len(data1),181,360))

    for i in range(len(data1)):
        for n in range(11):
            for r in range(11):
                #print(i)
                umrs[n,i,:,:]+=u850[time==(int(data1[i][13])-(10-r-n)*24)][0,:,:]
                vmrs[n,i,:,:]+=v850[time==(int(data1[i][13])-(10-r-n)*24)][0,:,:]
            umrs[n,i,:,:]=umrs[n,i,:,:]/11
            vmrs[n,i,:,:]=vmrs[n,i,:,:]/11
            umdsr[i,:,:]+=(u850[time==(int(data1[i][13])-(5-n)*24)][0,:,:]-umrs[n,i,:,:])**2
            vmdsr[i,:,:]+=(v850[time==(int(data1[i][13])-(5-n)*24)][0,:,:]-vmrs[n,i,:,:])**2
        umdsr[i,:,:]=umdsr[i,:,:]/11
        vmdsr[i,:,:]=vmdsr[i,:,:]/11
        eke2[i,:,:]=(umdsr[i,:,:]+vmdsr[i,:,:])/2

    _,p1 = ttest_ind(eke1,eke2,equal_var=False)

    deke=np.mean(eke1,axis=0)-np.mean(eke2,axis=0)
    """
    for i in range(181):
        for j in range(360):
            if p1[i,j]>=0.05:
                deke[i,j]=np.nan
    """
    levels = np.arange(-6,18+0.1,2)
    cb=ax.contourf(lon,lat,deke,levels=levels,cmap=newcmap,transform=ccrs.PlateCarree(),extend='both')
    ax.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6)
    """
    cs=ax.contourf(lon,lat,p1,[0,0.05,1], zorder=1,hatches=['+++',None],colors="none",transform=ccrs.PlateCarree())
    colors = ['silver']
    # For each level, we set the color of its hatch 
    for i, collection in enumerate(cs.collections):
        collection.set_edgecolor(colors[i % len(colors)])
    # Doing this also colors in the box around each level
    # We can remove the colored line around the levels by setting the linewidth to 0
    for collection in cs.collections:
        collection.set_linewidth(0.)
        """
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,160,160,180])
    gl.ylocator = mticker.FixedLocator(np.arange(-15,60,15))
    gl.top_labels    = False
    gl.right_labels    = False
    gl.bottom_labels    = False
    gl.left_labels    = False
    gl.xpadding = 2
    gl.ypadding = 2
    if b==0:
        gl.left_labels    = True
        
        
def kmke(a,ax,b):
    data=[]
    table=xlrd.open_workbook(a)#'/Users/fuzhenghang/Documents/ERA5/mtce/EKEson.xlsx'
    table=table.sheets()[0]
    nrows=table.nrows
    for i in range(nrows):
        if i ==0:
            continue
        data.append(table.row_values(i))
    data = [data[i] for i in range(0,len(data))]
    
    umdr=np.zeros((len(data),181,360))
    vmdr=np.zeros((len(data),181,360))
    umr=np.zeros((11,len(data),181,360))
    vmr=np.zeros((11,len(data),181,360))

    uv1=np.zeros((len(data),181,360))
    um=np.zeros((len(data),181,360))
    vm=np.zeros((len(data),181,360))
    ux=np.zeros((len(data),181,360))
    vx=np.zeros((len(data),181,360))
    uy=np.zeros((len(data),181,360))
    vy=np.zeros((len(data),181,360))
    kekm1=np.zeros((len(data),181,360))
    for i in range(len(data)):
        for n in range(11):
            um[i,:,:]+=u850[time==(int(data[i][13])-(5-n)*24)][0,:,:]
            vm[i,:,:]+=v850[time==(int(data[i][13])-(5-n)*24)][0,:,:]
            for r in range(11):
                umr[n,i,:,:]+=u850[time==(int(data[i][13])-(10-r-n)*24)][0,:,:]
                vmr[n,i,:,:]+=v850[time==(int(data[i][13])-(10-r-n)*24)][0,:,:]
            umr[n,i,:,:]=umr[n,i,:,:]/11
            vmr[n,i,:,:]=vmr[n,i,:,:]/11
            umdr[i,:,:]+=(u850[time==(int(data[i][13])-(5-n)*24)][0,:,:]-umr[n,i,:,:])**2
            vmdr[i,:,:]+=(v850[time==(int(data[i][13])-(5-n)*24)][0,:,:]-vmr[n,i,:,:])**2
            uv1[i,:,:]+=(u850[time==(int(data[i][13])-(5-n)*24)][0,:,:]-umr[n,i,:,:])*(v850[time==(int(data[i][13])-(5-n)*24)][0,:,:]-vmr[n,i,:,:])
        #print(vmdr[i,:,:])
        umdr[i,:,:]=umdr[i,:,:]/11#u'2-
        vmdr[i,:,:]=vmdr[i,:,:]/11
        uv1[i,:,:]=uv1[i,:,:]/11
        
        um[i,:,:]=um[i,:,:]/11
        vm[i,:,:]=vm[i,:,:]/11
        for la in range(1,180):
            for lo in range(1,359):
                ux[i,la,lo]=(um[i,la,lo+1]-um[i,la,lo-1])/(2*110940*np.cos((90-la)/180*np.pi))
                vx[i,la,lo]=(vm[i,la,lo+1]-vm[i,la,lo-1])/(2*110940*np.cos((90-la)/180*np.pi))
                uy[i,la,lo]=(um[i,la-1,lo]-um[i,la+1,lo])/(2*110940)
                vy[i,la,lo]=(vm[i,la-1,lo]-vm[i,la+1,lo])/(2*110940)
        kekm1[i,:,:]=-umdr[i,:,:]*ux[i,:,:]-uv1[i,:,:]*uy[i,:,:]-uv1[i,:,:]*vx[i,:,:]-vmdr[i,:,:]*vy[i,:,:]

    umdsr=np.zeros((len(data1),181,360))
    vmdsr=np.zeros((len(data1),181,360))
    umrs=np.zeros((11,len(data1),181,360))
    vmrs=np.zeros((11,len(data1),181,360))
    uv2=np.zeros((len(data1),181,360))
    ums=np.zeros((len(data1),181,360))
    vms=np.zeros((len(data1),181,360))
    uxs=np.zeros((len(data1),181,360))
    vxs=np.zeros((len(data1),181,360))
    uys=np.zeros((len(data1),181,360))
    vys=np.zeros((len(data1),181,360))
    kekm2=np.zeros((len(data1),181,360))
    for i in range(len(data1)):
        for n in range(11):
            ums[i,:,:]+=u850[time==(int(data1[i][13])-(5-n)*24)][0,:,:]
            vms[i,:,:]+=v850[time==(int(data1[i][13])-(5-n)*24)][0,:,:]
            for r in range(11):
                #print(i)
                umrs[n,i,:,:]+=u850[time==(int(data1[i][13])-(10-r-n)*24)][0,:,:]
                vmrs[n,i,:,:]+=v850[time==(int(data1[i][13])-(10-r-n)*24)][0,:,:]
            umrs[n,i,:,:]=umrs[n,i,:,:]/11
            vmrs[n,i,:,:]=vmrs[n,i,:,:]/11
            umdsr[i,:,:]+=(u850[time==(int(data1[i][13])-(5-n)*24)][0,:,:]-umrs[n,i,:,:])**2
            vmdsr[i,:,:]+=(v850[time==(int(data1[i][13])-(5-n)*24)][0,:,:]-vmrs[n,i,:,:])**2
            uv2[i,:,:]+=(u850[time==(int(data1[i][13])-(5-n)*24)][0,:,:]-umrs[n,i,:,:])*(v850[time==(int(data1[i][13])-(5-n)*24)][0,:,:]-vmrs[n,i,:,:])
        umdsr[i,:,:]=umdsr[i,:,:]/11
        vmdsr[i,:,:]=vmdsr[i,:,:]/11
        uv2[i,:,:]=uv2[i,:,:]/11
        
        ums[i,:,:]=ums[i,:,:]/11
        vms[i,:,:]=vms[i,:,:]/11
        for la in range(1,180):
            for lo in range(1,359):
                uxs[i,la,lo]=(ums[i,la,lo+1]-ums[i,la,lo-1])/(2*110940*np.cos((90-la)/180*np.pi))
                vxs[i,la,lo]=(vms[i,la,lo+1]-vms[i,la,lo-1])/(2*110940*np.cos((90-la)/180*np.pi))
                uys[i,la,lo]=(ums[i,la-1,lo]-ums[i,la+1,lo])/(2*110940)
                vys[i,la,lo]=(vms[i,la-1,lo]-vms[i,la+1,lo])/(2*110940)
        kekm2[i,:,:]=-umdsr[i,:,:]*uxs[i,:,:]-uv2[i,:,:]*uys[i,:,:]-uv2[i,:,:]*vxs[i,:,:]-vmdsr[i,:,:]*vys[i,:,:]
        
    _,p1 = ttest_ind(kekm1,kekm2,equal_var=False)
        
    deke=np.mean(kekm1,axis=0)-np.mean(kekm2,axis=0)
    """
    for i in range(181):
        for j in range(360):
            if p1[i,j]>=0.05:
                deke[i,j]=np.nan
                """

    levels = np.arange(-6,18+0.1,2)
    cb=ax.contourf(lon,lat,deke*100000,levels=levels,cmap=newcmap,transform=ccrs.PlateCarree(),extend='both')
    ax.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6)
    """
    cs=ax.contourf(lon,lat,p1,[0,0.05,1], zorder=1,hatches=['+++',None],colors="none",transform=ccrs.PlateCarree())
    colors = ['silver']
    # For each level, we set the color of its hatch 
    for i, collection in enumerate(cs.collections):
        collection.set_edgecolor(colors[i % len(colors)])
    # Doing this also colors in the box around each level
    # We can remove the colored line around the levels by setting the linewidth to 0
    for collection in cs.collections:
        collection.set_linewidth(0.)
    """
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,160,160,180])
    gl.ylocator = mticker.FixedLocator(np.arange(-15,60,15))
    gl.top_labels    = False
    gl.right_labels    = False
    gl.bottom_labels    = True
    gl.left_labels    = True
    gl.xpadding = 2
    gl.ypadding = 2
    if b!=1:
        gl.left_labels    = False
    if b==1:
        position=fig.add_axes([0.36, 0.55, 0.4, 0.016])
        cbar=fig.colorbar(cb,cax=position,orientation='horizontal',ticks=np.arange(-6,18+0.001,6),
                          aspect=20,shrink=0.5,pad=0.04)
        cbar.ax.tick_params(pad=3,length=0.5,width=0.7) 
   


eke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson1.xlsx',ax[0],0)
print(1)
eke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson2.xlsx',ax[2],2)
print(1)
eke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson3.xlsx',ax[4],4)
print(1)
kmke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson1.xlsx',ax[1],1)
print(1)
kmke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson2.xlsx',ax[3],3)
print(1)
kmke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson3.xlsx',ax[5],5)
print(1)
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/EKE.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)

