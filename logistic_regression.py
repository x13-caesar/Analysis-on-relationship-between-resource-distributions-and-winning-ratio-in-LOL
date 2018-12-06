#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:28:11 2018

@author: qiangwenxu
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


## 设一个分隔线的function，方便阅读结果
def line():
    print("")
    print("----------")
    print("")

## 优化数据
def GetKDA(df):
    # 把 deaths 项的 0 都变成 1
    df['deaths_p']=df['deaths'] 
    for i in range(df.shape[0]):
        if df.at[i, 'deaths'] == 0:
            df.set_value(i, 'deaths_p', 1)

    # 计算 KDA
    df['KDA'] = (df['kills']+df['assists'])/df['deaths_p']
    return df

## 得到建模需要的特征数据
def GetFactor(df, unwanted):
    # 筛选自变量
    df = df.drop(unwanted, axis=1)
    ## 按队伍分组，计算队伍方差 calculate the variance of each team.
    df_team = df.groupby(['match','win']) #categorize df
    df_var = df_team.agg('var').reset_index()
    # match和id用完了，也拿掉
    df_var = df_var.drop(['match',
                          'id'], axis=1)
    # 删掉 NaN的行
    df_var = df_var.dropna()
    return(df_var)

## 以 logistic Regression 进行建模
def GetModel(df_var):
    # 获得 Y
    Y = df_var['win'].values
    # 预测数据
    df_var = df_var.drop(['win'], axis=1)
    X = df_var.values
    # 将训练数据进行分割以便于进行交叉验证
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size = 0.3, 
                                                        random_state = 21)
    # 初始化logistics模型
    LRCls = LogisticRegression(penalty="l1")
    # 训练logistics模型
    LRCls.fit(X_train, Y_train.ravel())
    # 测试logistics模型
    model_score = LRCls.score(X_test, Y_test.ravel())
    print('The accuracy of the Logistic Regression model is', model_score)
    return LRCls

def GetImportance(LRCls, factors):
    imp = LRCls.coef_[0].tolist()
    imp_dic = dict(zip(factors, imp))
    return imp_dic

## 以下为使用真实数据进行分析的代码过程
    
## 确认样本数据
fileName = 'stats2.csv'

## 导入 raw data
df = pd.DataFrame(pd.read_csv(fileName, header=0))

df = GetKDA(df)

unwanted = ['item1',
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
             'timecc',
             'pinksbought',
             'wardsbought',
             'wardskilled',
             'firstblood']


df_var = GetFactor(df, unwanted)

LRCls = GetModel(df_var)

factors = df_var.columns.tolist()[1:]
importance = GetImportance(LRCls, factors)

print('The coefficient of factors:')
for key,value in importance.items():
    print('{key}:{value}'.format(key = key, value = value))
