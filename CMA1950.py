#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:31:58 2023

@author: fuzhenghang
"""

import xlrd

data=[]
#ori = pd.read_excel('D:/TC/dataTC2.xlsx')
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/CMA1223.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    if i ==0:
        continue
    data.append(table.row_values(i))
data = [data[i][:] for i in range(1,len(data))]
#print(data[0:5])
#data.columns=["名字","日期","小时","强度","纬度","经度","USA_WIND"]


a=1
b=[1]
c=[[[] for i in range(12)]for i in range(74)]
d=[[] for i in range(74)]
mtce=[[]for i in range(74)]
tce=[[]for i in range(74)]
mdayn=[[]for i in range(74)]
#data = data.values
#print(float(data[4][1]))
for i in range(len(data)-2):
    if int(data[i][11])==int(data[i+1][11]):
        if data[i][3]==data[i+1][3]:
            a+=1
            mtce[int(data[i][0])-1949].append(data[i][9])
            mtce[int(data[i][0])-1949].append(data[i+1][9])
            mdayn[int(data[i][0])-1949].append(int(data[i][11]))
            
        else:
            b.append(a)
            a=1
            
    else:
        if int(data[i][11])==int(data[i+1][11])-1:
            b.append(a)
            c[int(data[i][0]-1949)][int(data[i][1]-1)].append(max(b))
            b=[1]
            a=1
        else:
            b.append(a)
            c[int(data[i][0]-1949)][int(data[i][1]-1)].append(max(b))
            c[int(data[i][0]-1949)][int(data[i][1]-1)].append(0)
            b=[1]
            a=1
#print(c)    

for e in range(74):
    for f in range(11):
        if c[e][f+1]!=[] and c[e][f+1][0]>=2 and (c[e][f]==[] or c[e][f][-1]<=1):
            c[e][f+1].insert(0,1)


for r in range(74):
    for g in range(12):
        n=0
        for k in range(len(c[r][g])-1):
            if c[r][g][k]<=1 and c[r][g][k+1]>=2:
                n+=1
        d[r].append(n) 
           
print(d)
print('1979-2020总群发事件')
print(sum(d[i][j]for j in range(12) for i in range(74)))
yn=[]
for i in range(74):
    yn.append(sum(d[i]))
print('逐年')
print(yn)
#print(sum(d[i]) for i in range(42))
mon=[[] for i in range(12)]
for i in range(12):
    for j in range(74):
        mon[i].append(d[j][i])
print('逐月气候态')
mon=[sum(mon[i])/74 for i in range(12)]
print(mon)
print('年均')
print(sum(mon))

#data = data.values
#print(float(data[4][1]))
for i in range(len(data)-2):
    tce[int(data[i][0])-1949].append(data[i][0])
tcen=[]
for i in tce:
    i=set(i)
    tcen.append(len(i))
print('逐年总台风数')
print(tcen)

mtcen=[]
for i in mtce:
    i=set(i)
    mtcen.append(len(i))
print('逐年群发事件中的台风数')
print(mtcen)
print(sum(mtcen))
print('逐年群发天数')
mday=[]
for i in range(74):
    k=set(mdayn[i])
    mday.append(len(k))
print(mday)
r=[]
for i in range(74):
    r.append(mtcen[i]/tcen[i])
print(r)

