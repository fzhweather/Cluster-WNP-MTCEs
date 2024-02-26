

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:05:13 2022

@author: fuzhenghang
"""


import matplotlib.pyplot as plt###引入库包
import numpy as np
import matplotlib as mpl
import netCDF4 as nc
import matplotlib.colors
import xlrd
import cmaps
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 1

cmap=plt.get_cmap(cmaps.BlueDarkRed18_r)
newcolors=cmap(np.linspace(0, 1, 18))
newcmap = ListedColormap(newcolors[7:18])

input_data = r'/DS1/xshome/fuzh19/nc//850wdu.nc'
d1 = nc.Dataset(input_data)
d2 = nc.Dataset(r'/DS1/xshome/fuzh19/nc/850wdv.nc')   
d3 = nc.Dataset(r'/DS1/xshome/fuzh19/nc/200wdu.nc')  
d4 = nc.Dataset(r'/DS1/xshome/fuzh19/nc/200wdv.nc')  

time=d1['time'][:]
u850=d1["u"][:]
v850=d2["v"][:]
u200=d3["u"][:]
v200=d4["v"][:]

fig = plt.figure(figsize=(8,8),dpi=600)
x1 = [0.05,0.05,0.05,0.05,0.37,0.37,0.37,0.37,0.69,0.69,0.69,0.69]
yy = [0.8,0.63,0.46,0.29,0.8,0.63,0.46,0.29,0.8,0.63,0.46,0.29]
dx = 0.3
dy = 0.15
ax = []
label = ['(a) EI: -72 h','(b) EI: -48 h','(c) EI: -24 h','(d) EI: 0 h',
         '(e) WI-N: -72 h','(f) WI-N: -48 h','(g) WI-N: -24 h','(h) WI-N: 0 h',
         '(i) WI-O: -72 h','(j) WI-O: -48 h','(k) WI-O: -24 h','(l) WI-O: 0 h']
for i in range(12):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))
    ax[i].text(0.01,1.025,label[i],transform=ax[i].transAxes)


def dyc1(t,ax):
    cal = []
    data=[]
    table=xlrd.open_workbook('/DS1/xshome/fuzh19/mtce/dyson1.xlsx')#'/dpvhome/dpv16/dyson1.xlsx'
    table=table.sheets()[0]
    nrows=table.nrows
    for i in range(nrows):
        if i ==0:
            continue
        data.append(table.row_values(i))
    data = [data[i] for i in range(0,len(data))]
    lon2d=np.arange(-20,61,1)
    lat2d=np.arange(-20,21,1)
    u850sum=u850[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-20:round(float(data[0][6]/10))+61]
    v850sum=v850[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-20:round(float(data[0][6]/10))+61]
    u200sum=u200[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-20:round(float(data[0][6]/10))+61]
    v200sum=v200[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-20:round(float(data[0][6]/10))+61]
    shearu =u200sum-u850sum
    shearv =v200sum-v850sum
    shearall=(u200sum**2+v200sum**2)**0.5-(u850sum**2+v850sum**2)**0.5
    cal.append(np.mean(np.array(shearall)[0,15:26,15:26],axis=(0,1)))
    print(shearall[0,15:26,15:26].shape)
    print(np.mean(shearall[0,15:26,15:26],axis=(0,1)))
    for i in range(1,len(data)):
        u8501=u850[(time==(int(data[i][12+t])))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-20:round(float(data[i][6]/10))+61]
        v8501=v850[(time==(int(data[i][12+t])))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-20:round(float(data[i][6]/10))+61]
        u2001=u200[(time==(int(data[i][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-20:round(float(data[0][6]/10))+61]
        v2001=v200[(time==(int(data[i][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-20:round(float(data[0][6]/10))+61]
        shearu += (u2001-u8501)
        shearv += (v2001-v8501)
        shearall+=(u2001**2+v2001**2)**0.5-(u8501**2+v8501**2)**0.5
        temp = np.array((u2001**2+v2001**2)**0.5-(u8501**2+v8501**2)**0.5)
        cal.append(np.mean(temp[0,15:26,15:26]))
        #print(temp)
    shearumean =shearu/len(data)
    shearvmean =shearv/len(data)
    shearmean  =shearall/len(data)
       
    shearu_all=(np.mean(shearumean,axis=0))
    shearv_all=(np.mean(shearvmean,axis=0))
    shear_all=(np.mean(shearmean,axis=0))
    
    
    levels = np.arange(-3,21+0.1,3)
    cb = ax.contourf(lon2d,lat2d,shear_all,levels=levels,cmap=newcmap,extend='both')
    #cb = ax.contourf(lon2d,lat2d,msl_all,levels=levels,cmap='coolwarm',extend='both')
    ax.contour(lon2d,lat2d,shear_all,[-12,12],alpha=1,colors=['white',"white"])
    #cbar=fig.colorbar(cb,orientation='vertical',ticks=np.arange(572,580+0.1,2),aspect=25,shrink=0.5,pad=0.07)
    hh=ax.quiver(lon2d[::2],lat2d[::2],shearu_all[::2,::2],shearv_all[::2,::2],color='k',scale=290,width=0.004)
   
    my_ticks = np.arange(-20, 41, 10)
    my_yticks= np.arange(-20,21,10)
    ax.set_xticks(my_ticks)
    ax.set_yticks(my_yticks)
    ax.axis([-20,40,-20,15])
    ax.tick_params(pad=2,length=1)
    ax.grid(linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    ax.quiverkey(hh, X=0.8, Y = 1.035, U = 15,angle = 0,label='15m/s',labelpos='E', color = 'k',labelcolor = 'k')
    #ax.quiver(-18,18.5,10,0,color='k',width=0.008,scale=100)
    #ax.text(0,0,'10m/s',color='k',weight="bold",size=26)
    ax.text(0,0,'+',color='r',weight="bold",size=15,horizontalalignment='center', verticalalignment='center')
    
    if t!=0:
        ax.set_xticklabels([])
    ax.set_ylabel('Lon, deg',labelpad=1)
    if t==0:
        ax.set_xlabel('Lat, deg',labelpad=1)
    return cal
        

def dyc2(t,ax):
    cal = []
    data=[]
    table=xlrd.open_workbook('/DS1/xshome/fuzh19/mtce/dyson2.xlsx')
    table=table.sheets()[0]
    nrows=table.nrows
    for i in range(nrows):
        if i ==0:
            continue
        data.append(table.row_values(i))
    data = [data[i] for i in range(0,len(data))]


    lon2d=np.arange(-40,21,1)
    lat2d=np.arange(-20,21,1)
    u850sum=u850[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    v850sum=v850[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    u200sum=u200[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    v200sum=v200[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    shearu =u200sum-u850sum
    shearv =v200sum-v850sum
    shearall=(u200sum**2+v200sum**2)**0.5-(u850sum**2+v850sum**2)**0.5
    cal.append(np.mean(np.array(shearall)[0,15:26,35:46]))
    for i in range(1,len(data)):
        u8501=u850[(time==(int(data[i][12+t])))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-40:round(float(data[i][6]/10))+21]
        v8501=v850[(time==(int(data[i][12+t])))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-40:round(float(data[i][6]/10))+21]
        u2001=u200[(time==(int(data[i][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
        v2001=v200[(time==(int(data[i][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
        shearu += (u2001-u8501)
        shearv += (v2001-v8501)
        shearall+=(u2001**2+v2001**2)**0.5-(u8501**2+v8501**2)**0.5
        temp = np.array((u2001**2+v2001**2)**0.5-(u8501**2+v8501**2)**0.5)
        cal.append(np.mean(temp[0,15:26,35:46]))
    shearumean =shearu/len(data)
    shearvmean =shearv/len(data)
    shearmean  =shearall/len(data)
       
    shearu_all=(np.mean(shearumean,axis=0))
    shearv_all=(np.mean(shearvmean,axis=0))
    shear_all=(np.mean(shearmean,axis=0))
    
    levels = np.arange(-3,21+0.1,3)
    cb = ax.contourf(lon2d,lat2d,shear_all,levels=levels,cmap=newcmap,extend='both')
    #cb = ax.contourf(lon2d,lat2d,msl_all,levels=levels,cmap='coolwarm',extend='both')
    ax.contour(lon2d,lat2d,shear_all,[-12,12],alpha=1,colors=['white',"white"])
    #cbar=fig.colorbar(cb,orientation='vertical',ticks=np.arange(572,580+0.1,2),aspect=25,shrink=0.5,pad=0.07)
    hh=ax.quiver(lon2d[::2],lat2d[::2],shearu_all[::2,::2],shearv_all[::2,::2],color='k',scale=290,width=0.004)

    my_ticks = np.arange(-40, 21, 10)
    my_yticks= np.arange(-20,21,10)
    ax.set_xticks(my_ticks)
    ax.set_yticks(my_yticks)
    ax.axis([-40,20,-20,15])
    ax.tick_params(pad=2,length=1)
    ax.grid(linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    ax.quiverkey(hh, X=0.8, Y = 1.035, U = 15,angle = 0,label='15m/s',labelpos='E', color = 'k',labelcolor = 'k')
    #ax.quiver(-18,18.5,10,0,color='k',width=0.008,scale=100)
    #ax.text(0,0,'10m/s',color='k',weight="bold",size=26)
    if t!=0:
        ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.text(0,0,'+',color='r',weight="bold",size=15,horizontalalignment='center', verticalalignment='center')
    if t==0:
        ax.set_xlabel('Lat, deg',labelpad=1)
    return cal

def dyc3(t,ax):
    cal = []
    data=[]
    table=xlrd.open_workbook('/DS1/xshome/fuzh19/mtce/dyson3.xlsx')
    table=table.sheets()[0]
    nrows=table.nrows
    for i in range(nrows):
        if i ==0:
            continue
        data.append(table.row_values(i))
    data = [data[i] for i in range(0,len(data))]


    lon2d=np.arange(-40,21,1)
    lat2d=np.arange(-20,21,1)
    u850sum=u850[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    v850sum=v850[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    u200sum=u200[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    v200sum=v200[(time==(int(data[0][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    shearu =u200sum-u850sum
    shearv =v200sum-v850sum
    shearall=(u200sum**2+v200sum**2)**0.5-(u850sum**2+v850sum**2)**0.5
    cal.append(np.mean(np.array(shearall)[0,15:26,35:46]))
    for i in range(1,len(data)):
        u8501=u850[(time==(int(data[i][12+t])))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-40:round(float(data[i][6]/10))+21]
        v8501=v850[(time==(int(data[i][12+t])))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-40:round(float(data[i][6]/10))+21]
        u2001=u200[(time==(int(data[i][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
        v2001=v200[(time==(int(data[i][12+t])))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
        shearu += (u2001-u8501)
        shearv += (v2001-v8501)
        shearall+=(u2001**2+v2001**2)**0.5-(u8501**2+v8501**2)**0.5
        temp = np.array((u2001**2+v2001**2)**0.5-(u8501**2+v8501**2)**0.5)
        cal.append(np.mean(temp[0,15:26,35:46]))
    shearumean =shearu/len(data)
    shearvmean =shearv/len(data)
    shearmean  =shearall/len(data)
       
    shearu_all=(np.mean(shearumean,axis=0))
    shearv_all=(np.mean(shearvmean,axis=0))
    shear_all=(np.mean(shearmean,axis=0))
    
    levels = np.arange(-3,21+0.1,3)
    cb = ax.contourf(lon2d,lat2d,shear_all,levels=levels,cmap=newcmap,extend='both')
    #cb = ax.contourf(lon2d,lat2d,msl_all,levels=levels,cmap='coolwarm',extend='both')
    ax.contour(lon2d,lat2d,shear_all,[-12,12],alpha=1,colors=['white',"white"])
    #cbar=fig.colorbar(cb,orientation='vertical',ticks=np.arange(572,580+0.1,2),aspect=25,shrink=0.5,pad=0.07)
    hh=ax.quiver(lon2d[::2],lat2d[::2],shearu_all[::2,::2],shearv_all[::2,::2],color='k',scale=290,width=0.004)

    my_ticks = np.arange(-40, 21, 10)
    my_yticks= np.arange(-20,21,10)
    ax.set_xticks(my_ticks)
    ax.set_yticks(my_yticks)
    ax.axis([-40,20,-20,15])
    ax.tick_params(pad=2,length=1)
    ax.grid(linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    ax.quiverkey(hh, X=0.8, Y = 1.035, U = 15,angle = 0,label='15m/s',labelpos='E', color = 'k',labelcolor = 'k')

    #ax.quiver(-18,18.5,10,0,color='k',width=0.008,scale=100)
    #ax.text(0,0,'10m/s',color='k',weight="bold",size=26)
    ax.text(0,0,'+',color='r',weight="bold",size=15,horizontalalignment='center', verticalalignment='center')
    if t==0:
        position=fig.add_axes([0.32, 0.238, 0.4, 0.016])
        cbar=fig.colorbar(cb,cax=position,orientation='horizontal',ticks=[0,6,12,18],
                          aspect=20,shrink=0.5,pad=0.04)
        cbar.ax.tick_params(pad=3,length=0.5,width=0.7)     
    if t!=0:
        ax.set_xticklabels([])
    ax.set_yticklabels([])
    if t==0:
        ax.set_xlabel('Lat, deg',labelpad=1)
    return cal
    
gh = [[] for i in range(3)]  
gh[0].append(dyc1(3,ax[0]))
gh[0].append(dyc1(2,ax[1]))
gh[0].append(dyc1(1,ax[2]))
gh[0].append(dyc1(0,ax[3]))
gh[1].append(dyc2(3,ax[4]))
gh[1].append(dyc2(2,ax[5]))
gh[1].append(dyc2(1,ax[6]))
gh[1].append(dyc2(0,ax[7]))
gh[2].append(dyc3(3,ax[8]))
gh[2].append(dyc3(2,ax[9]))
gh[2].append(dyc3(1,ax[10]))
gh[2].append(dyc3(0,ax[11]))
gh = np.asarray(gh, dtype = object)
np.save('./shear.npy',gh)
plt.savefig('dyn5.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)




