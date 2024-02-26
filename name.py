#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:38:15 2023

@author: fuzhenghang
"""

import xlrd
data=[]
#ori = pd.read_excel('D:/TC/dataTC2.xlsx')
table=xlrd.open_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/1.xlsx')
table=table.sheets()[0]
nrows=table.nrows
for i in range(nrows):
    data.append(table.row_values(i))
# In[0]
print(data[0])
name = ''
for i in range(len(data)):
    if int(data[i][0])==66666:
        name = data[i][7]
    if int(data[i][0])!=66666:
        data[i][6]=name
# In[1]

import openpyxl
    

workbook = openpyxl.load_workbook('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/2.xlsx')
    

# 创建一个工作表
worksheet = workbook['Sheet1']
row = 1
for i in data:
    n=1
    for item in i:
        worksheet.cell(row, n).value = item
        n+=1
    row += 1
    
    # 保存工作簿
workbook.save('/Users/fuzhenghang/Documents/ERA5/MTCE/mtce/2.xlsx')