#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 21:22:41 2018

@author: qiangwenxu
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression

## 设一个分隔线的function，方便阅读结果
def next():
    print("")
    print("----------")
    print("")

## 导入 raw data
df = pd.DataFrame(pd.read_csv('stats_10000.csv',header=0))

## 看一下 raw data 的信息
df.info()
next()

##combine KDA 

## 按队伍分组，计算队伍方差 calculate the variance of each team.
df_team = df.groupby(['match','win']) #categorize df
df_var = df_team.agg('var').reset_index()

## 去掉不需要的列（不作为x变量输入）
## 可取变量输入数量为10个
df_var = df_var.drop(
        ['match',
         'id',
         'item1',
         'item2',
         'item3',
         'item4',
         'item5',
         'item6',
         'trinket',
         'largestkillingspree',
         'largestmultikill',
         'killingsprees',
         'longesttimespentliving',
         'doublekills',
         'triplekills',
         'quadrakills',
         'pentakills',
         'legendarykills',
         'magicdmgdealt',
         'physicaldmgdealt',
         'truedmgdealt',
         'largestcrit',
         'magicdmgtochamp', 
         'physdmgtaken',
         'truedmgtaken',
         'goldspent',
         'turretkills',
         'inhibkills',
         'neutralminionskilled',
         'ownjunglekills',
         'enemyjunglekills',
         'champlvl',
         'pinksbought',
         'wardsbought',
         'wardskilled',
         'firstblood',
         
         axis=1)

print("### Confirm the data structure ### \n \n",df_var.tail(6))
next()
df_var.to_csv('rank_var_test.csv')  #保存一下结果表格


  
