#!/usr/bin/python

import pandas as pd
import numpy as np

goodsdaily = pd.read_csv("../data/goodsdaily.csv", sep=",")

# 用户表现数据数据分析
# print("-------商品用户行为数据goodsdaily----------")
# print("是否有空值：", goodsdaily.apply(lambda x: True if x.isna().unique().shape[0] == 2 else False, axis=0))
# print("数据量：",goodsdaily.shape)
# print(goodsdaily.info())
# print(goodsdaily.describe())
# print("起止日期：%d---%d"%(goodsdaily["data_date"].max(),goodsdaily["data_date"].min()))
# print("goods_id是否唯一：",goodsdaily["goods_id"].is_unique)
# print("所有商品的最大最小在售天数：%d-%d"%(goodsdaily["onsale_days"].max(),goodsdaily["onsale_days"].min()))

# id=goodsdaily["data_date"].apply(lambda x:str(x))+goodsdaily["goods_id"]
# print("data_date与goods_id是否唯一：",id.is_unique)
