import pandas as pd
from sklearn.linear_model import LogisticRegression

'''
我们的因变量y其实是个(0/1) true or false 的值，
不能用线性回归或者ridge之类的变体，
否则预测结果出来个 0.2238，我们也不知道咋办……
看了下，只能用 logistic regression，
'''

## 读数据
df_var = pd.DataFrame(pd.read_csv('rank_var_test.csv',header=0))

## 定义 X 和 y
X = df_var.iloc[:2000, 1:(df_var.shape[0])]
y = df_var.win

## 定义 logistic regression model
# random_state 是个随机种子，不用管，写什么都行
# solver是用来选择优化算法
# multi_class用来定义多类，“one v.s. rest” - ovr
clf = LogisticRegression(random_state=0, solver='liblinear',
                         multi_class='ovr').fit(X, y)

## 看一下根据测试数据出来的准确度评分，0.8以上算能用了
print("regression model score: ", clf.score(X, y))
print("")
## 扔几个样本数据进去测一下，然后发现模型特别蠢 so bad
print("predict of first 10: ", clf.predict(X[:10]))
print("")
print("predict confidence scores for beginning 10: \n", 
      clf.decision_function(X[:10]))

'''

付从大神那儿拿来的代码，用的是 RidgeCV，
本质上是个自带“留一法”交叉验证的 岭回归 ridge regression，
我们研究的问题并不适用，
先把原本code扔这儿供参考。

def train_model(train_df, test_df):  #输入 train 和 test
    reg_target = train_df['y']
    del train_df['x']
    reg_data = train_df
    rcv = RidgeCV()
    rcv.fit(reg_data, reg_target)
    data1t_y = test_df['x']
    del test_df['x']
    data1t_x = test_df
    y_pred = rcv.predict(X=data1t_x)
    mse = mean_squared_error(data1t_y, y_pred)
    return rcv.alpha_, mse #输出 lambda 和 error

'''
    
