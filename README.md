# recommendation
推荐算法的实现，python、jupyter双版本，详尽注释。
正在努力更新中...

#### 对MovieLens电影评分小型数据集做Top N推荐：
0. baseline--推荐最热门的N=20部电影，precision=11%, recall=4%, coverage=0.2%, popularity=169。这部分内容包含在UserCF.py
1. UserCF，precision=25%, recall=10%, coverage=10%, popularity=4.4
2. ItemCF，precision=26%, recall=10%, coverage=15%, popularity=4.3

#### 对MovieLens电影评分小型数据集做评分预测：
0. baseline--使用所有评分的均值做预测，MAE:0.83，这部分内容包含在LinearRegression_Rating.ipynb
1. BiasSVD，MAE:0.68
2. LinearRegression，MAE:0.61
