# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:38:23 2020

@author: Dragonfly
@email: liuxiangmin@tom.com
"""

# 项亮《推荐系统实践》UserCF的实现
# 使用小型MovieLens数据集,10万数据量

# 选取k=10个最相似用户，推荐n=20个物品
# precision=25%, recall=10%, coverage=10%, popularity=4.4
# precision、recall和K不是正相关或负相关，需选取合适的K值。
# K值越大，越倾向于推荐热门物品，发现长尾物品能力越低，覆盖率越低。
# time: 74s

import pandas as pd
import random
import math
from operator import itemgetter
from tqdm import tqdm
import time

def split_data(data, train_test_ratio):
    """
    将数据集切分为训练集和测试集
    :param data: pandas DataFrame数据
    :param train_test_ratio: 训练集样本占总样本的比例
    :return: (train,词典 test词典)
    """
    train = {}  # 数据集是词典，key为用户，value为用户有过评分的电影列表
    test = {}
    random.seed(2020)
    for row in data.itertuples():
        if random.random() < train_test_ratio:
            if row[1] not in train.keys():  # userId的在tuple里的序号为1
                train[row[1]] = []
            train[row[1]].append(row[2])  # movieId在tuple里的序号为2
        else:
            if row[1] not in test.keys():
                test[row[1]] = []
            test[row[1]].append(row[2])

    return train, test

def user_similarity(train):
    """
    计算用户间的相似度
    :param train: 训练数据
    :return: 用户间相似度的词典
    """
    # 建立物品到用户之间的倒查表，降低计算用户相似度的时间复杂性
    item_users = {}
    for user, items in train.items():
        for item in items:
            if item not in item_users.keys():
                item_users[item] = set()
            item_users[item].add(user)

    # 计算用户间共同评论过的电影数量。相似度公式的分子
    C = {}      # 用户间共同评论过的电影数量
    N = {}      # 各用户评论过的电影数量
    for item, users in tqdm(item_users.items()):
        for u in users:
            if u not in N.keys():
                N[u] = 0
            N[u] += 1
            if u not in C.keys():
                C[u] = {}
            for v in users:
                if u == v:
                    continue
                if v not in C[u].keys():
                    C[u][v] = 0
                C[u][v] += 1 / math.log(1 + len(users))  # 惩罚热门物品对相似度的影响

    # 计算最终的用户间相似度。在上步的基础上除以分母
    W = {}
    for u, related_users in C.items():
        if u not in W.keys():
            W[u] = {}
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])

    return W

def get_recommend(user, train, W, K, n):
    """
    获取推荐结果
    :param uesr: 待推荐用户编号
    :param train: 训练集
    :param W: 计算得到的相似度词典
    :param K: 选取最相似的用户数
    :param n: 推荐物品的数量
    :return: 推荐的物品及用户对其的感兴趣程度（使用累积相似度度量）
    """
    rank = {}
    for sim_user, similarity in sorted(W[user].items(), key=itemgetter(1), reverse=True)[0:K]:
        for item in train[sim_user]:
            if item in train[user]:
                continue
            if item not in rank.keys():
                rank[item] = 0
            rank[item] += similarity
    rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]

    return rank

def precision(train, test, W, K, n):
    """
    计算推荐结果的精度
    :param train:  训练集
    :param test: 测试集
    :param W: 计算得到的相似度词典
    :param K: 选取最相似的用户数
    :param n: 推荐物品的数量
    :return: 精度
    """
    hit = 0
    total = 0
    for user in train.keys():
        if user in test.keys():
            rank = get_recommend(user,train, W, K, n)
            for item, _ in rank:
                if item in test[user]:
                    hit += 1
            total += n

    return hit / total

def recall(train, test, W, K, n):
    """
    计算推荐结果的召回率
    :param train:  训练集
    :param test: 测试集
    :param W: 计算得到的相似度词典
    :param K: 选取最相似的用户数
    :param n: 推荐物品的数量
    :return: 召回率
    """
    hit = 0
    total = 0
    for user in train.keys():
        if user in test.keys():
            rank = get_recommend(user,train, W, K, n)
            for item, _ in rank:
                if item in test[user]:
                    hit += 1
            total += len(test[user])

    return hit / total

def coverage(train, W, K, n):
    """
    计算推荐结果的覆盖率--所有的物品中，最终有多大比例被推荐
    :param train:  训练集
    :param W: 计算得到的相似度词典
    :param K: 选取最相似的用户数
    :param n: 推荐物品的数量
    :return: 覆盖率
    """
    recommend_items = set()
    total_items = set()
    for user in train.keys():
        for item in train[user]:
            total_items.add(item)
        rank = get_recommend(user, train, W, K, n)
        for item, _ in rank:
            recommend_items.add(item)

    return len(recommend_items) / len(total_items)

def popularity(train, W, K, n):
    """
    计算推荐结果的流行度--所有推荐物品流行度的均值
    :param train:  训练集
    :param W: 计算得到的相似度词典
    :param K: 选取最相似的用户数
    :param n: 推荐物品的数量
    :return: 流行度
    """
    # 计算各物品流行度，以出现次数度量
    item_popularity = {}
    for user in train.keys():
        for item in train[user]:
            if item not in item_popularity.keys():
                item_popularity[item] = 0
            item_popularity[item] += 1

    # 计算推荐结果的平均流行度
    pu = 0
    count = 0
    for user in train.keys():
        rank = get_recommend(user, train, W, K, n)
        for item, _ in rank:
            pu += math.log(1 + item_popularity[item])    # 物品的流行度满足长尾分布，取对数后，流行度均值更加稳定
            count += 1

    return pu / count

if __name__ == "__main__":
    # 获取开始时间
    start = time.time()

    # 读取数据
    rating_file = '../ml-latest-small/ratings.csv'
    data = pd.read_csv(rating_file)

    # 设定K,n值
    K = 10
    n = 20

    # 划分训练集测试集
    train, test = split_data(data, train_test_ratio=0.7)

    # 计算相似度
    W = user_similarity(train)

    # 推荐的精度
    re_precision = precision(train, test, W, K, n)
    print("precision: ", re_precision)

    # 推荐的召回率
    re_recall = recall(train, test, W, K, n)
    print("recall: ", re_recall)

    # 推荐的覆盖率
    re_coverage = coverage(train, W, K, n)
    print("coverage: ", re_coverage)

    # 推荐的平均流行度
    re_popularity = popularity(train, W, K, n)
    print("popularity: ", re_popularity)

    # 获取结束时间
    end = time.time()
    print("run time: ", end - start)
