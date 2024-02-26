#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:49:35 2024

@author: fuzhenghang
"""

import matplotlib.ticker as mticker
import xlrd
import xarray as xr  
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import numpy as np
import netCDF4 as nc
import dateutil.parser  #引入这个库来将日期字符串转成统一的datatime时间格式
import matplotlib as mpl
from numpy import nan
import scipy
from scipy.stats import pearsonr
from cartopy.util import add_cyclic_point
import cmaps
import seaborn as sns
sns.reset_orig()


mpl.rcParams["font.family"] = 'Arial'  
mpl.rcParams["mathtext.fontset"] = 'cm' 
mpl.rcParams["font.size"] = 10
plt.rcParams['hatch.color'] = 'k' 

fig = plt.figure(figsize=(5,8),dpi=1000)
ax=[]
x1 = [0,0,0]
yy = [1,0.68,0.36]
dx = 0.8
dy = 0.25
a1 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/a1.npy')
a2 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/a2.npy')
a3 = np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/a3.npy')
data = [a1,a2,a3]
for i in range(3):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))

co = ["#cd2626", "#ffa500", "#0000ff","#66cd00"]
sst = ['Nino3.4','PMM','TNA','NP']
la = ['o-','*-','^-','v-']
xl = ['SON','OND','NDJ','DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO']
ti = ['(a) EI','(b) WI-N','(c) WI-O']
for i in range(3):
    for j in range(4):
        y1 = np.zeros((12))-0.304
        y2 = y1+0.608
        x1 = [0+i for i in range(12)]
        ax[i].fill_between(x1, y1, y2, color='#DCDCDC',alpha=0.2,linewidths=0,zorder=0)
        ax[i].plot(data[i][j],la[j],color = co[j],ms=5,label=sst[j])
        ax[i].set_ylim(-0.5,0.5)
        ax[i].set_xlim(0,11)
        ax[i].set_xticks([i for i in range(12)])
        ax[i].set_xticklabels(xl)
        ax[i].text(0.1,0.53,ti[i])
        ax[i].axvline(x=2.5,  linestyle='--',linewidth = 0.6,color='gray',alpha=1,zorder=1)
    if i == 0:
        ax[i].legend(frameon=False,loc='upper left',ncol=2,fontsize=9)
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/leadlag.pdf', format="pdf", bbox_inches = 'tight',pad_inches=0,dpi=600)
