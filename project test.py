# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:17:26 2018

@author: jason
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

## 去掉不需要的列（不作为x变量输入）
## 可取变量输入数量为10个
## 看一下 raw data 的信息
def DataModify(df):
    df['deaths_p']=df['deaths']

    for i in range(df.shape[0]):
        if df.at[i, 'deaths'] == 0:
            df.set_value(i, 'deaths_p', 1)

    df['KDA'] = (df['kills']+df['assists'])/df['deaths_p']

    df = df.drop(
            ['item1',
             'item2',
             'item3',
             'item4',
             'item5',
             'item6',
             'trinket',
             'kills',
             'assists',
             'deaths',
             'deaths_p',
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
             'firstblood'], axis=1)
    
    ## 按队伍分组，计算队伍方差 calculate the variance of each team.
    df_team = df.groupby(['match','win']) #categorize df
    df_var = df_team.agg('var').reset_index()
    
    df_var = df_var.drop(['match',
                          'id'], axis=1)
    
    print("### Confirm the data structure ### \n \n",df_var.tail(6))
          
    return(df_var)
          
df_var = DataModify(df)

print("### Confirm the data structure ### \n \n",df_var.tail(6))
next()
df_var.to_csv('rank_var_test.csv')  #保存一下结果表格










