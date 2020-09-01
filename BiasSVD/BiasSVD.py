# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:06:02 2020

@author: Dragonfly
@email: liuxiangmin@tom.com
"""

# 带用户和物品打分偏差的矩阵分解算法做评分预测
# 使用小型MovieLens数据集,10万数据量
# MAE：0.68
# time: 迭代5个epoch, 用时436s

import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.preprocessing import LabelEncoder

class BiasSVD:
    def __init__(self, n_factors=30, init_mean=0, init_std_dev=0.1, lr=0.01, reg=0.02, n_epochs=5):
        self.n_factors = n_factors    # 隐向量维度
        self.init_mean = init_mean    # 隐向量初始化高斯分布均值
        self.init_std_dev = init_std_dev  # 隐向量初始化高斯分布标准差
        self.lr = lr       # 定义学习率。此处为统一的学习率，也可以给各参数分别定义。
        self.reg = reg      # 定义正则系数。此处为统一的正则系数，也可以给各参数分别定义。
        self.n_epochs = n_epochs    # 最大迭代次数

    def fit(self, train):
        # 定义模型参数
        self.mu = train['rating'].mean()        # 全局偏差常数
        self.bu = np.zeros(train['userId'].nunique())  # 定义用户偏差系数bu
        self.bi = np.zeros(train['movieId'].nunique())  # 定义物品偏差系数bi
        self.pu = np.random.normal(self.init_mean, self.init_std_dev, (train['userId'].nunique(), self.n_factors))    # 定义用户矩阵
        self.qi = np.random.normal(self.init_mean, self.init_std_dev, (train['movieId'].nunique(), self.n_factors))    # 定义物品矩阵

        # 使用随机梯度下降求解各模型参数
        for epoch in tqdm(range(self.n_epochs)):
            for row in train.itertuples():
                u = row[1]
                i = row[2]
                r = row[3]
                if i >= train['movieId'].nunique():                 # 当物品编号大于行数时，跳过
                    continue

                # 计算拟合值
                r_hat = self.mu + self.bi[i] + self.bu[u] + np.dot(self.qi[i], self.pu[u])
                error = r - r_hat

                # 更新参数
                self.bi[i] += self.lr * (error - self.reg * self.bi[i])
                self.bu[u] += self.lr * (error - self.reg * self.bu[u])
                self.qi[i] += self.lr * (error * self.pu[u] - self.reg * self.qi[i])
                self.pu[u] += self.lr * (error * self.qi[i] - self.reg * self.pu[u])

    def predict(self, u, i):
        try:
            qi_ = self.qi[i]
        except IndexError:
            print("unknown item: ", i)  # 未识别的物品id
            return
        try:
            pu_ = self.pu[u]
        except IndexError:
            print("unknown user: ", u)  # 未识别的用户id
            return

        r_hat = self.mu + self.bi[i] + self.bu[u] + np.dot(qi_, pu_)

        return r_hat


if __name__ == "__main__":
    # 获取开始时间
    start = time.time()

    # 读取数据
    rating_file = '../ml-latest-small/ratings.csv'
    data = pd.read_csv(rating_file)

    # 将Id 转换为用户矩阵/物品矩阵的行号
    data['userId'] = data['userId'] - 1
    data['movieId'] = LabelEncoder().fit_transform(data['movieId'])

    data.drop("timestamp", axis=1, inplace=True)  # 去除不用的列
    data = data.sample(frac=1, random_state=2020)  # 打乱顺序
    data.reset_index(drop=True, inplace=True)

    train_test_ratio = 0.7
    train = data.iloc[0:int(len(data) * train_test_ratio), :]  # 分割训练集合测试集
    test = data.iloc[int(len(data) * train_test_ratio):, :]

    # 调用BiasSVD类
    model = BiasSVD()

    # 模型训练
    model.fit(train)

    # 模型预测
    rating = model.predict(20, 100)
    print("The predict rating for user 20 to movie 100 is: ", rating)
    model.predict(100, 10000)   # 测试未识别的物品id
    model.predict(1000, 100)    # 测试未识别的用户id

    # 模型评估
    count = 0
    error_sum = 0
    for row in test.itertuples():
        u = row[1]
        i = row[2]
        r = row[3]
        # 跳过未识别的物品或用户
        if i < train['movieId'].nunique() and u < train['userId'].nunique():
            r_hat = model.predict(u, i)
            error_sum += abs(r - r_hat)
            count += 1
    print("MAE: ", error_sum / count)

    # 获取结束时间
    end = time.time()
    print("run time: ", end - start)
