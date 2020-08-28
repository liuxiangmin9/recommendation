# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:54:29 2020

@author: Dragonfly
@email: liuxiangmin@tom.com
"""

# 项亮《推荐系统实践》ItemCF的实现
# 使用小型MovieLens数据集,10万数据量

# 给用户评论过的电影选取k=10个最相似的电影，最终推荐n=20部电影
# precision=26%, recall=10%, coverage=15%, popularity=4.3
# precision、recall和K不是正相关或负相关，需选取合适的K值。
# K值越大，越倾向于推荐热门物品，发现长尾物品能力越低，覆盖率越低。
# time: 347s

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

def item_similarity(train):
    """
    计算物品间的相似度
    :param train: 训练数据
    :return: 物品间相似度的词典
    """
    # 已经有了用户到物品之间的倒查表，不需要再建立
    # 计算二部电影被用户都评论过的用户数量。N(i)∩N(j),评论过i电影和评论过j电影的用户的交集。相似度公式的分子

    C = {}  # 计算二部电影被用户都评论过的用户数量
    N = {}  # 统计各电影被多少用户评论过
    for user, items in tqdm(train.items()):
        for i in items:
            if i not in N.keys():
                N[i] = 0
            N[i] += 1
            if i not in C.keys():
                C[i] = {}
            for j in items:
                if i == j:
                    continue
                if j not in C[i].keys():
                    C[i][j] = 0
                C[i][j] += 1 / math.log(1 + len(items))  # 惩罚活跃用户对相似度的影响

    # 计算最终的物品间相似度
    # 除以分母
    W = {}
    for i, related_items in C.items():
        if i not in W.keys():
            W[i] = {}
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])

    return W

def get_recommend(user, train, W, K, n):
    """
    获取推荐结果
    :param uesr: 待推荐用户编号
    :param train: 训练集
    :param W: 计算得到的相似度词典
    :param K: 选取最相似的物品数
    :param n: 推荐物品的数量
    :return: 推荐的物品及用户对其的感兴趣程度（使用累积相似度度量）
    """
    rank = {}
    for item in train[user]:
        for sim_item, similarity in sorted(W[item].items(), key=itemgetter(1), reverse=True)[0:K]:
            if sim_item in train[user]:
                continue
            if sim_item not in rank:
                rank[sim_item] = 0
            rank[sim_item] += similarity
    rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]

    return rank

def precision(train, test, W, K, n):
    """
    计算推荐结果的精度
    :param train:  训练集
    :param test: 测试集
    :param W: 计算得到的相似度词典
    :param K: 选取最相似的物品数
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
    :param K: 选取最相似的物品数
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
    :param K: 选取最相似的物品数
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
    :param K: 选取最相似的物品数
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
    W = item_similarity(train)

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
