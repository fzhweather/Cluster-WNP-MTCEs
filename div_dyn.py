
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:42:06 2022

@author: fuzhenghang
"""


import matplotlib.pyplot as plt###引入库包
import numpy as np
import matplotlib as mpl
import netCDF4 as nc
import matplotlib.colors
import xlrd
import cmaps

mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 1



input_data = r'/DS1/xshome/fuzh19/nc/850d.nc'
d1 = nc.Dataset(input_data)
time=d1['time'][:]
u850=d1["d"][:]
print(u850.shape)

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
    lon2d=np.arange(-20,41,1)
    lat2d=np.arange(-20,21,1)
    
    u850sum=u850[time==(int(data[0][12+t]))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-20:round(float(data[0][6]/10))+41]
    cal.append(np.mean(u850[time==(int(data[0][12+t]))][:,85+round(float(data[0][5]/10)):96+round(float(data[0][5]/10)),round(float(data[0][6]/10))-5:round(float(data[0][6]/10))+6]))
    for i in range(1,len(data)):
        u850sum+=u850[time==(int(data[i][12+t]))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-20:round(float(data[i][6]/10))+41]
        cal.append(np.mean(u850[time==(int(data[i][12+t]))][:,85+round(float(data[i][5]/10)):96+round(float(data[i][5]/10)),round(float(data[i][6]/10))-5:round(float(data[i][6]/10))+6]))
    u850mean=u850sum/len(data)

    u_all=(np.mean(u850mean,axis=0))
    u_all=u_all*1000000
    
    levels = np.arange(-5,5+0.01,0.25)
    cb = ax.contourf(lon2d,lat2d,u_all,levels=levels,cmap='RdYlBu_r',extend='both')

    my_ticks = np.arange(-20, 41, 10)
    my_yticks= np.arange(-20,21,10)
    ax.set_xticks(my_ticks)
    ax.set_yticks(my_yticks)
    ax.axis([-20,40,-20,15])
    ax.tick_params(pad=2,length=1)
    ax.grid(linewidth=0.3, color='k', alpha=0.5, linestyle='--')
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
    u850sum=u850[time==(int(data[0][12+t]))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    cal.append(np.mean(u850[time==(int(data[0][12+t]))][:,85+round(float(data[0][5]/10)):96+round(float(data[0][5]/10)),round(float(data[0][6]/10))-5:round(float(data[0][6]/10))+6]))
    for i in range(1,len(data)):
        u850sum+=u850[time==(int(data[i][12+t]))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-40:round(float(data[i][6]/10))+21]
        cal.append(np.mean(u850[time==(int(data[i][12+t]))][:,85+round(float(data[i][5]/10)):96+round(float(data[i][5]/10)),round(float(data[i][6]/10))-5:round(float(data[i][6]/10))+6]))
    u850mean=u850sum/len(data)

    u_all=(np.mean(u850mean,axis=0))
    u_all=u_all*1000000
    
    levels = np.arange(-5,5+0.01,0.25)
    cb = ax.contourf(lon2d,lat2d,u_all,levels=levels,cmap='RdYlBu_r',extend='both')

    my_ticks = np.arange(-40, 21, 10)
    my_yticks= np.arange(-20,21,10)
    ax.set_xticks(my_ticks)
    ax.set_yticks(my_yticks)
    ax.axis([-40,20,-20,15])
    ax.tick_params(pad=2,length=1)
    ax.grid(linewidth=0.3, color='k', alpha=0.5, linestyle='--')
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
    u850sum=u850[time==(int(data[0][12+t]))][:,70+round(float(data[0][5]/10)):111+round(float(data[0][5]/10)),round(float(data[0][6]/10))-40:round(float(data[0][6]/10))+21]
    cal.append(np.mean(u850[time==(int(data[0][12+t]))][:,85+round(float(data[0][5]/10)):96+round(float(data[0][5]/10)),round(float(data[0][6]/10))-5:round(float(data[0][6]/10))+6]))
    for i in range(1,len(data)):
        u850sum+=u850[time==(int(data[i][12+t]))][:,70+round(float(data[i][5]/10)):111+round(float(data[i][5]/10)),round(float(data[i][6]/10))-40:round(float(data[i][6]/10))+21]
        cal.append(np.mean(u850[time==(int(data[i][12+t]))][:,85+round(float(data[i][5]/10)):96+round(float(data[i][5]/10)),round(float(data[i][6]/10))-5:round(float(data[i][6]/10))+6]))
    u850mean=u850sum/len(data)

    u_all=(np.mean(u850mean,axis=0))
    u_all=u_all*1000000

    levels = np.arange(-5,5+0.01,0.25)
    cb = ax.contourf(lon2d,lat2d,u_all,levels=levels,cmap='RdYlBu_r',extend='both')

    my_ticks = np.arange(-40, 21, 10)
    my_yticks= np.arange(-20,21,10)
    ax.set_xticks(my_ticks)
    ax.set_yticks(my_yticks)
    ax.axis([-40,20,-20,15])
    ax.tick_params(pad=2,length=1)
    ax.grid(linewidth=0.3, color='k', alpha=0.5, linestyle='--')
    #ax.quiver(-18,18.5,10,0,color='k',width=0.008,scale=100)
    #ax.text(0,0,'10m/s',color='k',weight="bold",size=26)
    ax.text(0,0,'+',color='r',weight="bold",size=15,horizontalalignment='center', verticalalignment='center')
    if t==0:
        position=fig.add_axes([0.32, 0.238, 0.4, 0.016])
        cbar=fig.colorbar(cb,cax=position,orientation='horizontal',ticks=np.arange(-4,4+0.001,2),
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
np.save('./div.npy',gh)
plt.savefig('dyn2.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)





