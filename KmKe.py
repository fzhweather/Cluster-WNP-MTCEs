
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:57:28 2022

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

mpl.rcParams["font.family"] = 'Arial'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 1


cmap=plt.get_cmap(cmaps.temp_19lev)
newcolors=cmap(np.linspace(0, 1, 19))
newcmap = ListedColormap(newcolors[6:18])


d1 = Dataset(r'/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/850u_daily_mean.nc')
d2 = Dataset(r'/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/850v_daily_mean.nc') 
time=d1.variables['time'][:]
u850=d1.variables["u"][:]
v850=d2.variables["v"][:]
lat=d1.variables['latitude'][:] 
lon=d1.variables['longitude'][:]
# In[1]
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
x1 = [0.05,0.05,0.05,0.05,0.32,0.32,0.32,0.32,0.59,0.59,0.59,0.59]
yy = [0.9,0.6,0.3,0,0.9,0.6,0.3,0,0.9,0.6,0.3,0]
dx = 0.5
dy = 0.25
ax = []
label = ['(a) EI-Ux','(b) EI-Uy','(c) EI-Vx','(d) EI-Vy','(e) OWI-Ux','(f) OWI-Uy','(g) OWI-Vx','(h) OWI-Vy','(i) FWI-Ux','(j) FWI-Uy','(k) FWI-Vx','(l) FWI-Vy',]
for i in range(12):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy],projection = proj))
    ax[i].text(-78,51.5,label[i])

        
        
def kmke(a,ax,bx,cx,dx,num):
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
    
    kekma1=np.zeros((len(data),181,360))
    kekmb1=np.zeros((len(data),181,360))
    kekmc1=np.zeros((len(data),181,360))
    kekmd1=np.zeros((len(data),181,360))
    
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
        kekma1[i,:,:]=-umdr[i,:,:]*ux[i,:,:]
        kekmb1[i,:,:]=-uv1[i,:,:]*uy[i,:,:]
        kekmc1[i,:,:]=-uv1[i,:,:]*vx[i,:,:]
        kekmd1[i,:,:]=-vmdr[i,:,:]*vy[i,:,:]
        
        
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
    kekma2=np.zeros((len(data1),181,360))
    kekmb2=np.zeros((len(data1),181,360))
    kekmc2=np.zeros((len(data1),181,360))
    kekmd2=np.zeros((len(data1),181,360))
    
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
        kekma2[i,:,:]=-umdsr[i,:,:]*uxs[i,:,:]
        kekmb2[i,:,:]=-uv2[i,:,:]*uys[i,:,:]
        kekmc2[i,:,:]=-uv2[i,:,:]*vxs[i,:,:]
        kekmd2[i,:,:]=-vmdsr[i,:,:]*vys[i,:,:]
        
    _,p1 = ttest_ind(kekma1,kekma2,equal_var=False)
    _,p2 = ttest_ind(kekmb1,kekmb2,equal_var=False)
    _,p3 = ttest_ind(kekmc1,kekmc2,equal_var=False)
    _,p4 = ttest_ind(kekmd1,kekmd2,equal_var=False)
        
    dekea=np.mean(kekma1,axis=0)-np.mean(kekma2,axis=0)
    dekeb=np.mean(kekmb1,axis=0)-np.mean(kekmb2,axis=0)
    dekec=np.mean(kekmc1,axis=0)-np.mean(kekmc2,axis=0)
    deked=np.mean(kekmd1,axis=0)-np.mean(kekmd2,axis=0)
    """
    for i in range(181):
        for j in range(360):
            if p1[i,j]>=0.05:
                dekea[i,j]=np.nan
            if p2[i,j]>=0.05:
                dekeb[i,j]=np.nan
            if p3[i,j]>=0.05:
                dekec[i,j]=np.nan
            if p4[i,j]>=0.05:
                deked[i,j]=np.nan
    """
    levels = np.arange(-6,18+0.1,2)
    cb=ax.contourf(lon,lat,dekea*100000,levels=levels,cmap=newcmap,transform=ccrs.PlateCarree(),extend='both')
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
    gl.left_labels    = True
    gl.xpadding = 2
    gl.ypadding = 2
    if num!=1:
        gl.left_labels    = False
        
    cb=bx.contourf(lon,lat,dekeb*100000,levels=levels,cmap=newcmap,transform=ccrs.PlateCarree(),extend='both')
    bx.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    bx.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6)
    """
    cs=bx.contourf(lon,lat,p1,[0,0.05,1], zorder=1,hatches=['+++',None],colors="none",transform=ccrs.PlateCarree())
    colors = ['silver']
    # For each level, we set the color of its hatch 
    for i, collection in enumerate(cs.collections):
        collection.set_edgecolor(colors[i % len(colors)])
    # Doing this also colors in the box around each level
    # We can remove the colored line around the levels by setting the linewidth to 0
    for collection in cs.collections:
        collection.set_linewidth(0.)
    """
    gl = bx.gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,160,160,180])
    gl.ylocator = mticker.FixedLocator(np.arange(-15,60,15))
    gl.top_labels    = False
    gl.right_labels    = False
    gl.bottom_labels    = False
    gl.left_labels    = True
    gl.xpadding = 2
    gl.ypadding = 2
    if num!=1:
        gl.left_labels    = False    

    cb=cx.contourf(lon,lat,dekec*100000,levels=levels,cmap=newcmap,transform=ccrs.PlateCarree(),extend='both')
    cx.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    cx.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6)
    """
    cs=cx.contourf(lon,lat,p1,[0,0.05,1], zorder=1,hatches=['+++',None],colors="none",transform=ccrs.PlateCarree())
    colors = ['silver']
    # For each level, we set the color of its hatch 
    for i, collection in enumerate(cs.collections):
        collection.set_edgecolor(colors[i % len(colors)])
    # Doing this also colors in the box around each level
    # We can remove the colored line around the levels by setting the linewidth to 0
    for collection in cs.collections:
        collection.set_linewidth(0.)
    """
    gl = cx.gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,160,160,180])
    gl.ylocator = mticker.FixedLocator(np.arange(-15,60,15))
    gl.top_labels    = False
    gl.right_labels    = False
    gl.bottom_labels    = False
    gl.left_labels    = True
    gl.xpadding = 2
    gl.ypadding = 2
    if num!=1:
        gl.left_labels    = False
        
    cb=dx.contourf(lon,lat,deked*100000,levels=levels,cmap=newcmap,transform=ccrs.PlateCarree(),extend='both')
    dx.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    dx.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6)
    """
    cs=dx.contourf(lon,lat,p1,[0,0.05,1], zorder=1,hatches=['+++',None],colors="none",transform=ccrs.PlateCarree())
    colors = ['silver']
    # For each level, we set the color of its hatch 
    for i, collection in enumerate(cs.collections):
        collection.set_edgecolor(colors[i % len(colors)])
    # Doing this also colors in the box around each level
    # We can remove the colored line around the levels by setting the linewidth to 0
    for collection in cs.collections:
        collection.set_linewidth(0.)
    """
    gl = dx.gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([100,120,160,160,180])
    gl.ylocator = mticker.FixedLocator(np.arange(-15,60,15))
    gl.top_labels    = False
    gl.right_labels    = False
    gl.bottom_labels    = True
    gl.left_labels    = True
    gl.xpadding = 2
    gl.ypadding = 2
    if num!=1:
        gl.left_labels    = False    
    if num==3:
        position=fig.add_axes([0.36, -0.04, 0.4, 0.016])
        cbar=fig.colorbar(cb,cax=position,orientation='horizontal',ticks=np.arange(-6,18+0.001,6),
                          aspect=20,shrink=0.5,pad=0.04)
        cbar.ax.tick_params(pad=3,length=0.5,width=0.7) 
    



kmke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson1.xlsx',ax[0],ax[1],ax[2],ax[3],1)
print(1)」
kmke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson2.xlsx',ax[4],ax[5],ax[6],ax[7],2)
print(1)
kmke('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/EKEson3.xlsx',ax[8],ax[9],ax[10],ax[11],3)
print(1)
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/KmKe.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)


