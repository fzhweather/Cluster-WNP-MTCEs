#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:28:40 2024

@author: fuzhenghang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = 'Airal'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.linewidth"] = 1


gh = list(np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/gh.npy',allow_pickle=True))
div = list(np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/div.npy',allow_pickle=True))
vor = list(np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/vor.npy',allow_pickle=True))
rh = list(np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/rh.npy',allow_pickle=True))
shear = list(np.load('/Users/fuzhenghang/Documents/python/WNP-MTCE/data/shear.npy',allow_pickle=True))
#print(shear[0][0])
data = [gh,rh,div,vor,shear]
x1 = [0,0.2,0.4,0.6,0.8,0,0.2,0.4,0.6,0.8,0,0.2,0.4,0.6,0.8]
yy = [1,1,1,1,1,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6]
dx = 0.16
dy = 0.16
ax = []
fig = plt.figure(figsize=(10,10),dpi=600)
for i in range(15):
    ax.append(fig.add_axes([x1[i],yy[i],dx,dy]))

la = ['(a) EI: 500H','(b) EI: 600SH','(c) EI: 850Div.','(d) EI: 850Vor.','(e) EI: 200-850VWS',
      '(f) WI-N: 500H','(g) WI-N: 600SH','(h) WI-N: 850Div.','(i) WI-N: 850Vor.','(j) WI-N: 200-850VWS',
      '(k) WI-O: 500H','(l) WI-O: 600SH','(m) WI-O: 850Div.','(n) WI-O: 850Vor.','(o) WI-O: 200-850VWS']
co = ['tomato','#66cd00','royalblue','orange','#00BFFF']
yt = [[580,582,584,586,588,590],[0.002,0.004,0.006,0.008],[-0.000009,-0.000006,-0.000003,0,0.000003],
      [-0.00002,0,0.00002,0.00004],[-10,0,10,20]]
yl = [[579.5,591],[0.001,0.008],[-0.000009,0.000003],[-0.00002,0.00005],[-12,20]]
for i in range(3):
    for j in range(5):
        f = ax[i*5+j].boxplot(data[j][i],showfliers=False,showmeans=True,patch_artist=True,whis=1.2,meanprops={'markerfacecolor':co[j],"markeredgecolor": "k","linewidth": 0.05,'markersize':4})
        colors = [co[j],co[j],co[j],co[j]]
        for box, c in zip(f['boxes'], colors):  
            box.set(color='k', linewidth=1)
            box.set(facecolor=c)
        for median in f['medians']:
            median.set(color='k', linewidth=1)
        for whisker, cap in zip(f['whiskers'], f['caps']):
            whisker.set(color='black', linestyle='--', linewidth=1)
            cap.set(color='black', linewidth=1)

        ax[i*5+j].text(2.5,0.03*(yl[j][1]-yl[j][0])+yl[j][1],la[i*5+j],horizontalalignment ='center')
        ax[i*5+j].set_yticks(yt[j])
        ax[i*5+j].set_ylim(yl[j])
        if j == 1:
            ax[i*5+j].set_yticklabels([2,4,6,8])
            ax[i*5+j].text(0.46,0.00815,'1e-3')
        if i != 2:
            ax[i*5+j].set_xticklabels([])
        if i == 2:
            ax[i*5+j].set_xticklabels(['-72h','-48h','-24h','+0h'])
plt.savefig('/Users/fuzhenghang/Documents/python/WNP-MTCE/epsfig/Equa_cir.eps', format="eps", bbox_inches = 'tight',pad_inches=0,dpi=600)
        
        