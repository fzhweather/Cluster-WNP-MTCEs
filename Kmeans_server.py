#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 16:32:49 2023

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

from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans

mpl.rcParams["font.family"] = 'Arial'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 4
mpl.rcParams["axes.linewidth"] = 0.5
data=[]
data1=[]
data2=[]
table=xlrd.open_workbook('/DS1/xshome/fuzh19/mtce/genesis_locationTCmother1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data.append(table.row_values(i))
data = [data[i] for i in range(0,len(data))]
print(data[0])
table=xlrd.open_workbook('/DS1/xshome/fuzh19/mtce/locationTCmother1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data1.append(table.row_values(i))
data1 = [data1[i] for i in range(0,len(data1))]

table=xlrd.open_workbook('/DS1/xshome/fuzh19/mtce/locationTCson1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data2.append(table.row_values(i))
data2 = [data2[i] for i in range(0,len(data2))]

x=[]
y=[]
z=[]

for i in range(len(data)):
    x.append(data[i][5:7])
for i in range(len(data1)):
    y.append(data1[i][5:7])
for i in range(len(data2)):
    z.append(data2[i][5:7])
for i in range(len(x)):
    x[i][0]=x[i][0]/10
    x[i][1]=x[i][1]/10
    y[i][0]=y[i][0]/10
    y[i][1]=y[i][1]/10
    z[i][0]=z[i][0]/10
    z[i][1]=z[i][1]/10
print(z[0:10])

dot1=np.zeros((329, 3),dtype=np.float32)
dot2=np.zeros((329, 3),dtype=np.float32)
#dot3=[]
#dot4=[]
for i in range(len(x)):
    dot1[i,0]=x[i][1]
    dot2[i,0]=x[i][0]
    dot1[i,1]=y[i][1]
    dot2[i,1]=y[i][0]
    dot1[i,2]=z[i][1]
    dot2[i,2]=z[i][0]

traj = np.zeros((329,3,2))
traj[:,:,0] = dot1
traj[:,:,1] = dot2
traj = traj.reshape((329,3*2))
km = KMeans(n_clusters=3)#构造聚类器
km.fit(traj)#聚类
label = km.labels_#获取聚类标签

"""
SSE = []
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(traj)
        #获取K-means算法的SSE
    SSE.append(km.inertia_)
S = [0]  # 存放轮廓系数
from sklearn.metrics import silhouette_score
for i in range(2,11):
    kmeans = KMeans(n_clusters=i)  # 构造聚类器
    kmeans.fit(traj)
    S.append(silhouette_score(traj,kmeans.labels_,metric='euclidean'))
print(S,SSE)

"""
labels = np.array(list(km.labels_))
label0=np.array(np.where(labels==0))
label1=np.array(np.where(labels==1))
label2=np.array(np.where(labels==2))

numlabel0=len(label0[0,:])
numlabel1=len(label1[0,:])
numlabel2=len(label2[0,:])


x0 = np.array([dot1[i,:]  for i in label0]).reshape((numlabel0,3))
y0 = np.array([dot2[i,:]  for i in label0]).reshape((numlabel0,3))
x1 = np.array([dot1[i,:]  for i in label1]).reshape((numlabel1,3))    
y1 = np.array([dot2[i,:]  for i in label1]).reshape((numlabel1,3))     
x2 = np.array([dot1[i,:]  for i in label2]).reshape((numlabel2,3))     
y2 = np.array([dot2[i,:]  for i in label2]).reshape((numlabel2,3))

#print(x0)
    
fig = plt.figure(figsize=(6,4),dpi=600)#设置比例
proj = ccrs.PlateCarree()
leftlon, rightlon, lowerlat, upperlat = (100,180,0,50)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()

fig_ax1 = fig.add_axes([0.2,0.55,0.25,0.4],projection = proj)
fig_ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
#fig_ax1.add_feature(cfeature.OCEAN.with_scale('50m'))
#fig_ax1.add_feature(cfeature.LAND.with_scale('50m'))
fig_ax1.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.3)
#fig_ax1.add_feature(cfeature.RIVERS.with_scale('50m'))
#fig_ax1.add_feature(cfeature.LAKES.with_scale('50m'))
gl = fig_ax1.gridlines(draw_labels=True, linewidth=0.1, color='k', linestyle='--')
gl.top_labels    = False  
gl.right_labels  = False
gl.bottom_labels  = False

#fig_ax1.set_title('(a) EOF1',loc='left',fontsize =10)
#for i in range(len(dot1)):
    #fig_ax1.quiver(dot1[i], dot2[i], dot3[i]-dot1[i], dot4[i]-dot2[i], angles='xy', scale=1, scale_units='xy',width=0.002)
#fig_ax1.arrow(dot1[0], dot2[0], dot3[0]-dot1[0], dot4[0]-dot2[0], head_width=0.05, head_length=0.1, fc='k', ec='k')
for i in range(numlabel0):
    fig_ax1.quiver(x0[i,0], y0[i,0], x0[i,1]-x0[i,0], y0[i,1]-y0[i,0], angles='xy', scale=1, scale_units='xy',width=0.0015,color='violet',headwidth=0,headlength=0)
    fig_ax1.scatter(x0[i,0], y0[i,0],marker='.',c='red',s=2)
    fig_ax1.scatter(x0[i,1], y0[i,1],marker='|',c='red',s=0.5)
    fig_ax1.quiver(x0[i,1], y0[i,1], x0[i,2]-x0[i,1], y0[i,2]-y0[i,1], angles='xy', scale=1, scale_units='xy',width=0.0015,color='darkorange',headwidth=20,headlength=20)

fig_ax2 = fig.add_axes([0.475,0.55,0.25,0.4],projection = proj)
fig_ax2.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
fig_ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.3)
#fig_ax2.add_feature(cfeature.RIVERS.with_scale('50m'))
#fig_ax2.add_feature(cfeature.LAKES.with_scale('50m'))
gl2=fig_ax2.gridlines(draw_labels=True, linewidth=0.1, color='k', alpha=0.5, linestyle='--')
gl2.top_labels    = False  
gl2.right_labels  = False
gl2.left_labels  = False
for i in range(numlabel1):
    fig_ax2.quiver(x1[i,0], y1[i,0], x1[i,1]-x1[i,0], y1[i,1]-y1[i,0], angles='xy', scale=1, scale_units='xy',width=0.0015,color='violet',headwidth=0,headlength=0)
    fig_ax2.scatter(x1[i,0], y1[i,0],marker='.',c='red',s=2)
    fig_ax2.scatter(x1[i,1], y1[i,1],marker='|',c='red',s=0.5)
    fig_ax2.quiver(x1[i,1], y1[i,1], x1[i,2]-x1[i,1], y1[i,2]-y1[i,1], angles='xy', scale=1, scale_units='xy',width=0.0015,color='darkorange',headwidth=20,headlength=20)

fig_ax3 = fig.add_axes([0.2,0.27,0.25,0.4],projection = proj)
fig_ax3.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
fig_ax3.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.3)
#fig_ax3.add_feature(cfeature.RIVERS.with_scale('50m'))
#ig_ax3.add_feature(cfeature.LAKES.with_scale('50m'))
gl3=fig_ax3.gridlines(draw_labels=True, linewidth=0.1, color='k', alpha=0.5, linestyle='--')
gl3.top_labels    = False  
gl3.right_labels  = False
for i in range(numlabel2):
    fig_ax3.quiver(x2[i,0], y2[i,0], x2[i,1]-x2[i,0], y2[i,1]-y2[i,0], angles='xy', scale=1, scale_units='xy',width=0.0015,color='violet',headwidth=0,headlength=0)
    fig_ax3.scatter(x2[i,0], y2[i,0],marker='.',c='red',s=2)
    fig_ax3.scatter(x2[i,1], y2[i,1],marker='|',c='red',s=0.5)
    fig_ax3.quiver(x2[i,1], y2[i,1], x2[i,2]-x2[i,1], y2[i,2]-y2[i,1], angles='xy', scale=1, scale_units='xy',width=0.0015,color='darkorange',headwidth=20,headlength=20)


print((numlabel0))
print((numlabel1))
print((numlabel2))

lab =list(labels)
print(list(labels))
fig1=[[],[],[],[],[],[]]
fig2=[[],[],[],[],[],[]]
fig3=[[],[],[],[],[],[]]
fig4=[[],[],[],[],[],[]]
for i in range(len(lab)):
    if lab[i]==0:
        fig1[0].append(data[i][6])
        fig1[1].append(data[i][5])
        fig1[2].append(data1[i][6])
        fig1[3].append(data1[i][5])
        fig1[4].append(data2[i][6])
        fig1[5].append(data2[i][5])
    elif lab[i]==1:
        fig2[0].append(data[i][6])
        fig2[1].append(data[i][5])
        fig2[2].append(data1[i][6])
        fig2[3].append(data1[i][5])
        fig2[4].append(data2[i][6])
        fig2[5].append(data2[i][5])
    elif lab[i]==2:
        fig3[0].append(data[i][6])
        fig3[1].append(data[i][5])
        fig3[2].append(data1[i][6])
        fig3[3].append(data1[i][5])
        fig3[4].append(data2[i][6])
        fig3[5].append(data2[i][5])


fig1=[0.1*sum(a)/numlabel0 for a in fig1]
fig2=[0.1*sum(a)/numlabel1 for a in fig2]
fig3=[0.1*sum(a)/numlabel2 for a in fig3]

print(fig3)
fig_ax1.quiver(fig1[0], fig1[1], fig1[2]-fig1[0], fig1[3]-fig1[1], angles='xy', scale=1, scale_units='xy',width=0.006,color='blue',headwidth=1,headlength=0)
fig_ax1.scatter(fig1[0], fig1[1],marker='.',c='blue',s=20)
fig_ax1.scatter(fig1[2], fig1[3],marker='_',c='blue',s=15)
fig_ax1.quiver(fig1[2], fig1[3], fig1[4]-fig1[2], fig1[5]-fig1[3], angles='xy', scale=1, scale_units='xy',width=0.006,color='blue',headwidth=4,headlength=4)

fig_ax2.quiver(fig2[0], fig2[1], fig2[2]-fig2[0], fig2[3]-fig2[1], angles='xy', scale=1, scale_units='xy',width=0.006,color='blue',headwidth=1,headlength=0)
fig_ax2.scatter(fig2[0], fig2[1],marker='.',c='blue',s=20)
fig_ax2.scatter(fig2[2], fig2[3],marker='_',c='blue',s=15)
fig_ax2.quiver(fig2[2], fig2[3], fig2[4]-fig2[2], fig2[5]-fig2[3], angles='xy', scale=1, scale_units='xy',width=0.006,color='blue',headwidth=4,headlength=4)

fig_ax3.quiver(fig3[0], fig3[1], fig3[2]-fig3[0], fig3[3]-fig3[1], angles='xy', scale=1, scale_units='xy',width=0.006,color='blue',headwidth=1,headlength=0)
fig_ax3.scatter(fig3[0], fig3[1],marker='.',c='blue',s=20)
fig_ax3.scatter(fig3[2], fig3[3],marker='_',c='blue',s=15)
fig_ax3.quiver(fig3[2], fig3[3], fig3[4]-fig3[2], fig3[5]-fig3[3], angles='xy', scale=1, scale_units='xy',width=0.006,color='blue',headwidth=4,headlength=4)
plt.text(111.5,52,'(c) Num : 108',horizontalalignment='center', verticalalignment='center')
plt.text(110.5,112,'(a) Num : 98',horizontalalignment='center', verticalalignment='center')
plt.text(199.7,112,'(b) Num : 123',horizontalalignment='center', verticalalignment='center')
plt.text(189.5,51,'(d) ')
gl.ypadding=1
gl2.xpadding=1
gl3.xpadding=1
gl3.ypadding=1

fig_ax4 = fig.add_axes([0.475,0.356,0.25,0.23])
SSE = []
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(traj)
        #获取K-means算法的SSE
    SSE.append(km.inertia_)
S = [0]  # 存放轮廓系数
from sklearn.metrics import silhouette_score
for i in range(2,11):
    kmeans = KMeans(n_clusters=i)  # 构造聚类器
    kmeans.fit(traj)
    S.append(silhouette_score(traj,kmeans.labels_,metric='euclidean'))
x=[1+i for i in range(10)]
fig_ax4.tick_params(length=1.8,width=0.4,pad=0.5)
fig_ax4.set_xticks(x)
fig_ax4.set_yticks([0.0,0.1,0.2,0.3])
fig_ax4.set_yticklabels([0.0,0.1,0.2,0.3],color='r')

ax2 = fig_ax4.twinx()
fig_ax4.axvline(x=3,  linestyle='-',linewidth = 3,color='gainsboro',zorder=0)
fig_ax4.grid(ls="--",c='k',linewidth=0.05,zorder=10)
fig_ax4.plot(x,S,marker='^',markersize=1,linewidth=0.5,color='r',label='ASW',zorder=1)
ax2.set_yticks([100000,150000,200000,250000,300000])
ax2.tick_params(length=1.8,width=0.4,pad=0.5)
ax2.set_yticklabels([100000,150000,200000,250000,300000],color='b')
def formatnum(x, pos):
    return '$%.1f$x$10^{5}$' % (x/100000)
formatter = FuncFormatter(formatnum)
ax2.yaxis.set_major_formatter(formatter)
ax2.plot(x,SSE,marker='o',markersize=1,linewidth=0.5,color='b',label='SSE',zorder=1)
fig_ax4.legend(frameon=False,loc='upper right', bbox_to_anchor=(0.75, 1))
ax2.legend(frameon=False,loc=0)
fig_ax4.set_xlabel('Number of clusters, k',labelpad=1)
plt.savefig('/DS1/xshome/fuzh19/code/202211/fig5.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)




