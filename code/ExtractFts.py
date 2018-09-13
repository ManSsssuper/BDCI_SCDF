#!/usr/bin/python

import pandas as pd
import datetime as dt


def extractFts(ftsDelta=80):
    trainTimeSlice = [[20170826, 20171114, 20171230, 20180202],
                      [20170819, 20171107, 20171223, 20180126],
                      [20170812, 20171031, 20171216, 20180119],
                      [20170805, 20171024, 20171209, 20180112],
                      [20170729, 20171017, 20171202, 20180105],
                      [20170401, 20170620, 20170805, 20170908],
                      [20170325, 20170613, 20170729, 20170901],
                      [20170318, 20170606, 20170722, 20170825],
                      [20170311, 20170530, 20170715, 20170818],
                      [20170304, 20170523, 20170708, 20170811]]
    testTimeSlice = [20171226, 20180316, 20180501, 20180604]
    saleData = pd.read_csv("../features/saledata_sub_presku.csv", sep=",", index_col=None, low_memory=False)
    goodsdailyData = pd.read_csv("../features/goodsdaily_sub_presku.csv", sep=",", index_col=None)
    goodsPromoteData = pd.read_csv("../features/goodspromote_sub_presku.csv", sep=",", index_col=None)
    goodsInfoData = pd.read_csv("../data/goodsinfo.csv", sep=",", index_col=None)
    # 删除无用列
    goodsInfoData = goodsInfoData.drop(["cat_level6_id", "cat_level7_id"], axis=1)

    trainFts = pd.DataFrame()
    for timeSlice in trainTimeSlice:
        labelData = saleData[(saleData["data_date"] >= timeSlice[2]) & (saleData["data_date"] <= timeSlice[3])]
        ftsData = saleData[(saleData["data_date"] >= timeSlice[0]) & (saleData["data_date"] <= timeSlice[1])]
        ftsData = ftsData[ftsData["sku_id"].isin(labelData["sku_id"])]
        trainFts = pd.concat(
            [trainFts, extract(goodsdailyData, goodsPromoteData, goodsInfoData, ftsData, labelData, timeSlice)], axis=0)

    ftsData = saleData[(saleData["data_date"] >= testTimeSlice[0]) & (saleData["data_date"] <= testTimeSlice[1])]
    trainFts.to_csv("../features/trainFts.csv", sep=",", index=None)

    testFts = extract(goodsdailyData, goodsPromoteData, goodsInfoData, ftsData, None, testTimeSlice, train=False)
    testFts.to_csv("../features/testFts.csv", sep=",", index=None)
    return trainFts, testFts


def extract(goodsdailyData, goodsPromoteData, goodsInfoData, ftsData, labelData, timeSlice, train=True):
    # 格式化数据
    ftsData["goods_price"] = ftsData["goods_price"].astype(float)
    ftsData["orginal_shop_price"] = ftsData["orginal_shop_price"].astype(float)
    if train:
        fts = labelData[["sku_id", "goods_id"]].drop_duplicates().reset_index(drop=True)
    else:
        testData = pd.read_csv("../data/submit_example.csv", sep=",", index_col=None)
        relation = pd.read_csv("../data/goods_sku_relation.csv", sep=",", index_col=None)
        fts = relation[relation["sku_id"].isin(testData["sku_id"])][["sku_id", "goods_id"]].reset_index(drop=True)
    goods_id = pd.DataFrame(labelData["goods_id"].unique())
    goods_id.columns = ["goods_id"]
    if train:
        # 获取标记
        week1 = timeSlice[2]
        week2 = int(dt.datetime.strftime((dt.datetime.strptime(str(week1), "%Y%m%d") + dt.timedelta(7)), "%Y%m%d"))
        week3 = int(dt.datetime.strftime((dt.datetime.strptime(str(week2), "%Y%m%d") + dt.timedelta(7)), "%Y%m%d"))
        week4 = int(dt.datetime.strftime((dt.datetime.strptime(str(week3), "%Y%m%d") + dt.timedelta(7)), "%Y%m%d"))
        week5 = int(dt.datetime.strftime((dt.datetime.strptime(str(week4), "%Y%m%d") + dt.timedelta(7)), "%Y%m%d"))
        # 获取连接键值

        labels = pd.DataFrame(fts["sku_id"].unique())

        labels.columns = ["sku_id"]

        label1 = pd.DataFrame(labelData[(labelData["data_date"] >= week1) & (labelData["data_date"] < week2)][
                                  ["sku_id", "goods_num"]].groupby("sku_id").sum()).reset_index(drop=False)
        label2 = pd.DataFrame(labelData[(labelData["data_date"] >= week2) & (labelData["data_date"] < week3)][
                                  ["sku_id", "goods_num"]].groupby("sku_id").sum()).reset_index(drop=False)
        label3 = pd.DataFrame(labelData[(labelData["data_date"] >= week3) & (labelData["data_date"] < week4)][
                                  ["sku_id", "goods_num"]].groupby("sku_id").sum()).reset_index(drop=False)
        label4 = pd.DataFrame(labelData[(labelData["data_date"] >= week4) & (labelData["data_date"] < week5)][
                                  ["sku_id", "goods_num"]].groupby("sku_id").sum()).reset_index(drop=False)
        label5 = pd.DataFrame(
            labelData[labelData["data_date"] >= week5][["sku_id", "goods_num"]].groupby("sku_id").sum()).reset_index(
            drop=False)

        labels = pd.merge(labels, label1, how="left", on="sku_id")
        labels = pd.merge(labels, label2, how="left", on="sku_id")
        labels = pd.merge(labels, label3, how="left", on="sku_id")
        labels = pd.merge(labels, label4, how="left", on="sku_id")
        labels = pd.merge(labels, label5, how="left", on="sku_id")
        labels.fillna(0, inplace=True)
        labels.columns = ["sku_id", "week1", "week2", "week3", "week4", "week5"]

    # 首先要初始化时间区间
    days = [timeSlice[1], ]
    tmpDate = dt.datetime.strptime(str(timeSlice[1]), "%Y%m%d")
    for delta in [2, 3, 5, 7, 10, 20, 30, 40, 60]:
        day = int(dt.datetime.strftime((tmpDate - dt.timedelta(delta - 1)), "%Y%m%d"))
        days.append(day)
    days.append(timeSlice[0])

    # 自定义函数
    def getSkuFts(column):
        saleSum = column["goods_num"].sum()
        saleMean = saleSum / len(column)
        saleAvgPrice = (column["goods_num"].mul(column["goods_price"], axis=0)).sum() / saleSum
        saleAvgOriginPrice = (column["goods_num"].mul(column["orginal_shop_price"], axis=0)).sum() / saleSum
        return saleSum, saleMean, saleAvgPrice, saleAvgOriginPrice

    def getGoodsDailyFts(column):
        click = column["goods_click"].sum()
        cart = column["cart_click"].sum()
        favor = column["favorites_click"].sum()
        buy = column["sales_uv"].sum()
        return click, cart, favor, buy

    def getGoodsSaleFts(column):
        saleSum = column["goods_num"].sum()
        saleMean = saleSum / len(column)
        return saleSum, saleMean

    def getSkuRank(column):
        saleSum = column
        return saleSum

    # 生成时间序列特征
    for timeStart in days:
        # 获取sku特征
        # 前1、2、3、5、7、10、20、30、40、60、80天总销量
        # 前1、2、3、5、7、10、20、30、40、60、80天平均销量
        # 前1、2、3、5、7、10、20、30、40、60、80天平均价格（总价 / 销量）
        # 前1、2、3、5、7、10、20、30、40、60、80天吊牌平均价格（总价 / 销量）
        skuFts = ftsData[(ftsData["data_date"] >= timeStart)].groupby("sku_id").apply(getSkuFts)

        skuFts0 = skuFts.apply(lambda x: x[0])
        skuFts1 = skuFts.apply(lambda x: x[1])
        skuFts2 = skuFts.apply(lambda x: x[2])
        skuFts3 = skuFts.apply(lambda x: x[3])

        skuFts = pd.concat([skuFts0, skuFts1, skuFts2, skuFts3], axis=1)
        skuFts = skuFts.reset_index(drop=False)
        skuFts.columns = ["sku_id", "sf0", "sf1", "sf2", "sf3"]
        fts = pd.merge(fts, skuFts, how="left", on="sku_id")
        print(fts.shape)
        # ------------------------------------------------------------------------
        # goods特征：

        # goods在1、2、3、5、7、10、20、30、40、60、80天的点击、加购、收藏、购买数
        goodsdailyFts = goodsdailyData[
            (goodsdailyData["data_date"] >= timeStart) & (goodsdailyData["data_date"] <= timeSlice[1])].groupby(
            "goods_id").apply(getGoodsDailyFts)

        goodsdailyFts0 = goodsdailyFts.apply(lambda x: x[0])
        goodsdailyFts1 = goodsdailyFts.apply(lambda x: x[1])
        goodsdailyFts2 = goodsdailyFts.apply(lambda x: x[2])
        goodsdailyFts3 = goodsdailyFts.apply(lambda x: x[3])

        goodsdailyFts = pd.concat([goodsdailyFts0, goodsdailyFts1, goodsdailyFts2, goodsdailyFts3], axis=1)
        goodsdailyFts = goodsdailyFts.reset_index(drop=False)
        goodsdailyFts.columns = ["goods_id", "gf0", "gf1", "gf2", "gf3"]
        fts = pd.merge(fts, goodsdailyFts, how="left", on="goods_id")
        print("haha")
        print(fts.shape)

        # goods在1、2、3、5、7、10、20、30、40、60、80天销售量
        # goods前1、2、3、5、7、10、20、30、40、60、80天平均销量
        goodsSaleFts = ftsData[ftsData["data_date"] >= timeStart].groupby("goods_id").apply(getGoodsSaleFts)

        goodsSaleFts0 = goodsSaleFts.apply(lambda x: x[0])
        goodsSaleFts1 = goodsSaleFts.apply(lambda x: x[1])

        goodsSaleFts = pd.concat([goodsSaleFts0, goodsSaleFts1], axis=1)
        goodsSaleFts = goodsSaleFts.reset_index(drop=False)
        goodsSaleFts.columns = ["goods_id", "gsf0", "gsf1"]
        fts = pd.merge(fts, goodsSaleFts, how="left", on="goods_id")
        print(fts.shape)

        # # goods在1、2、3、5、7、10、20、30、40、60、80天的促销天数
        # goodsPromoteDays = goodsPromoteData[
        #     (goodsPromoteData["data_date"] >= timeStart) & (goodsPromoteData["data_date"] <= timeSlice[1])].groupby(
        #     "goods_id").count()
        # print(goodsPromoteDays)
        # goodsPromoteDays = pd.DataFrame(goodsPromoteDays).reset_index(drop=False)
        # print(goodsPromoteDays)
        # goodsPromoteDays.columns = ["goods_id", "goodsPromoteDays"]
        # fts = pd.merge(fts, goodsPromoteDays, how="left", on="goods_id")

        # 在1、2、3、5、7、10、20、30、40、60、80天，sku在goods里面的销量排名
        skuInGoodsSaleRank = ftsData[ftsData["data_date"] >= timeStart].groupby("goods_id").apply(
            lambda x: x.groupby("sku_id").apply(lambda y: y["goods_num"].sum()).rank())
        skuInGoodsSaleRank = skuInGoodsSaleRank.reset_index(drop=False)
        skuInGoodsSaleRank.columns = ["goods_id", "sku_id", "rank"]
        skuInGoodsSaleRank = skuInGoodsSaleRank.drop(["goods_id"], axis=1)
        fts = pd.merge(fts, skuInGoodsSaleRank, how="left", on="sku_id")
        print(fts.columns)
        print(fts.shape)

    # 填充缺失值
    fts.fillna(0, inplace=True)

    # goods信息特征
    # goods所属的一至五级类目id
    # goods季节属性
    # goods品牌id
    fts = pd.merge(fts, goodsInfoData, how="left", on="goods_id")
    print(fts.shape)
    if train:
        labels = labels.drop(["sku_id"], axis=1)
        fts = pd.concat([fts, labels], axis=1)
    print("生成特征：", fts.shape)
    return fts


extractFts()

# import datetime as dt
# dates = []
# labelEndDate = dt.datetime(2018, 6, 4)
# labelStartDate = labelEndDate - dt.timedelta(34)
# ftsEndDate = labelStartDate - dt.timedelta(46)
# ftsStartDate = ftsEndDate - dt.timedelta(80)
# dates.append(
#     [int(dt.datetime.strftime(day, "%Y%m%d")) for day in [ftsStartDate, ftsEndDate, labelStartDate, labelEndDate]])
# while int(dt.datetime.strftime(ftsStartDate, "%Y%m%d")) >= 20170301:
#     labelEndDate = labelEndDate-dt.timedelta(7)
#     labelStartDate = labelEndDate - dt.timedelta(34)
#     ftsEndDate = labelStartDate - dt.timedelta(46)
#     ftsStartDate = ftsEndDate - dt.timedelta(80)
#     if int(dt.datetime.strftime(ftsStartDate, "%Y%m%d")) >= 20170301:
#         dates.append([int(dt.datetime.strftime(day, "%Y%m%d")) for day in
#                   [ftsStartDate, ftsEndDate, labelStartDate, labelEndDate]])
# print(dates)
