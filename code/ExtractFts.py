#!/usr/bin/python

import pandas as pd
import numpy as np
import datetime as dt

lastDate = dt.datetime.strptime("20180316", "%Y%m%d")
# 训练标记日
trainLabelDates = [str(x) for x in range(20180307, 20180317, 1)]


def initData(tofile):
    saleData = pd.read_csv("../data/goodsale.csv", sep=",", index_col=None)
    submitSkuData = pd.read_csv("../data/submit_example.csv", sep=",", index_col=None)
    # 对数据进行筛选，只取预测sku相关的数据
    data = saleData[saleData["sku_id"].isin(submitSkuData["sku_id"])].iloc[:10000, :]
    submitSku = submitSkuData["sku_id"]
    if tofile:
        data.to_csv("../features/data_sub_sku.csv", sep=",", index=None)
    return data, submitSku


def extractFts(testLabelDate, dataSubSkuToFile=False, saveFts=True):
    # testLabelDate例如：20180501
    # 为每天提取一次特征，所以总共需要训练35次，提取35次特征

    data, submitSku = initData(tofile=dataSubSkuToFile)

    testDate = dt.datetime.strptime(str(testLabelDate), "%Y%m%d")
    interval = testDate - lastDate

    # 得到训练数据
    trainData = pd.DataFrame()
    for trainLabelDate in trainLabelDates:
        ftsLastDate = int(dt.datetime.strftime(dt.datetime.strptime(trainLabelDate, "%Y%m%d") - interval, "%Y%m%d"))
        ftsData = data[data["data_date"] <= ftsLastDate]
        labelData = data[data["data_date"] == testLabelDate]
        trainData = pd.concat([trainData, extract(ftsData, labelData, ftsLastDate, trainLabelDate, submitSku)], axis=0)
    # 得到预测用X
    testData = extract(data, None, 20180316, testLabelDate, submitSku)
    if saveFts:
        trainData.to_csv("../features/" + str(testLabelDate) + "_train.csv", sep=",", index=None)
        testData.to_csv("../features/" + str(testLabelDate) + "_test.csv", sep=",", index=None)
    return trainData, testData


def extract(ftsData, labelData, ftsLastDate, labelDate, submitSku):
    fts = pd.DataFrame(labelData["sku_id"]).set_index("sku_id", drop=False)

    lastyear_today = int(labelDate) - 10000

    ly_today_date = dt.datetime.strptime(str(lastyear_today), "%Y%m%d")
    fts_last_date = dt.datetime.strptime(str(ftsLastDate), "%Y%m%d")
    lasyear_beforetoday_2_day = int(dt.datetime.strftime(ly_today_date - dt.timedelta(1), "%Y%m%d"))
    lasyear_beforetoday_3_day = int(dt.datetime.strftime(ly_today_date - dt.timedelta(2), "%Y%m%d"))
    lasyear_beforetoday_4_day = int(dt.datetime.strftime(ly_today_date - dt.timedelta(3), "%Y%m%d"))
    lasyear_beforetoday_1_week = int(dt.datetime.strftime(ly_today_date - dt.timedelta(6), "%Y%m%d"))
    lasyear_aftertoday_2_day = int(dt.datetime.strftime(ly_today_date + dt.timedelta(1), "%Y%m%d"))
    lasyear_aftertoday_3_day = int(dt.datetime.strftime(ly_today_date + dt.timedelta(2), "%Y%m%d"))
    lasyear_aftertoday_4_day = int(dt.datetime.strftime(ly_today_date + dt.timedelta(3), "%Y%m%d"))
    lasyear_aftertoday_1_week = int(dt.datetime.strftime(ly_today_date + dt.timedelta(6), "%Y%m%d"))
    lasyear_aftertoday_1_month = int(dt.datetime.strftime(ly_today_date + dt.timedelta(29), "%Y%m%d"))
    lasyear_aftertoday_2_month = int(dt.datetime.strftime(ly_today_date + dt.timedelta(59), "%Y%m%d"))
    lasyear_aftertoday_3_month = int(dt.datetime.strftime(ly_today_date + dt.timedelta(89), "%Y%m%d"))
    beforelastday_1_month = int(dt.datetime.strftime(fts_last_date - dt.timedelta(29), "%Y%m%d"))
    beforelastday_2_month = int(dt.datetime.strftime(fts_last_date - dt.timedelta(59), "%Y%m%d"))
    beforelastday_3_month = int(dt.datetime.strftime(fts_last_date - dt.timedelta(89), "%Y%m%d"))

    # sku去年今日的销量
    fts = pd.merge(fts,
                   ftsData[ftsData["data_date"] == lastyear_today][["sku_id", "goods_num"]],
                   how="left", on='sku_id')
    # sku去年今日前两天的销量

    s1 = ftsData[
        (ftsData["data_date"] >= lasyear_beforetoday_2_day) & (
                ftsData["data_date"] <= lastyear_today)][["sku_id", "goods_num"]].groupby("sku_id").apply(
        lambda x: sum(x["goods_num"]))

    # sku去年今日前三天的销量
    s2 = ftsData[
        (ftsData["data_date"] >= lasyear_beforetoday_3_day) & (
                ftsData["data_date"] <= lastyear_today)][["sku_id", "goods_num"]].groupby("sku_id").apply(
        lambda x: sum(x["goods_num"]))
    # sku去年今日前一周的销量
    fts = pd.merge(fts, ftsData[
        (ftsData["data_date"] >= lasyear_beforetoday_1_week) & (
                ftsData["data_date"] <= lastyear_today)][["sku_id", "goods_num"]].groupby("sku_id").apply(
        lambda x: sum(x["goods_num"])), how="left", left_index=True, right_index=True)
    # sku去年今日后两天的销量
    fts = pd.merge(fts, ftsData[
        (ftsData["data_date"] >= lastyear_today) & (
                ftsData["data_date"] <= lasyear_aftertoday_2_day)][["sku_id", "goods_num"]].groupby("sku_id").apply(
        lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku去年今日后三天的销量
    fts = pd.merge(fts, ftsData[
        (ftsData["data_date"] >= lastyear_today) & (
                ftsData["data_date"] <= lasyear_aftertoday_3_day)][["sku_id", "goods_num"]].groupby("sku_id").apply(
        lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku去年今日后一周的销量
    fts = pd.merge(fts, ftsData[
        (ftsData["data_date"] >= lastyear_today) & (
                ftsData["data_date"] <= lasyear_aftertoday_1_week)][["sku_id", "goods_num"]].groupby("sku_id").apply(
        lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku去年今日作为一周中第四天的一周销量
    fts = pd.merge(fts, ftsData[
        (ftsData["data_date"] >= lasyear_beforetoday_4_day) & (
                ftsData["data_date"] <= lasyear_aftertoday_4_day)][["sku_id", "goods_num"]].groupby("sku_id").apply(
        lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku去年今日后一个月的销量
    fts = pd.merge(fts, ftsData[
        (ftsData["data_date"] >= lastyear_today) & (
                ftsData["data_date"] <= lasyear_aftertoday_1_month)][["sku_id", "goods_num"]].groupby(
        "sku_id").apply(
        lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku去年今日后两个月的销量
    fts = pd.merge(fts, ftsData[
        (ftsData["data_date"] >= lastyear_today) & (
                ftsData["data_date"] <= lasyear_aftertoday_2_month)][["sku_id", "goods_num"]].groupby(
        "sku_id").apply(
        lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku去年今日后三个月的销量
    fts = pd.merge(fts, ftsData[
        (ftsData["data_date"] >= lastyear_today) & (
                ftsData["data_date"] <= lasyear_aftertoday_3_month)][["sku_id", "goods_num"]].groupby(
        "sku_id").apply(
        lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku最后一天前一个月的销量
    fts = pd.merge(fts, ftsData[
        ftsData["data_date"] >= beforelastday_1_month][["sku_id", "goods_num"]].groupby(
        "sku_id").apply(lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku最后一天前两个月的销量
    fts = pd.merge(fts, ftsData[
        ftsData["data_date"] >= beforelastday_2_month][["sku_id", "goods_num"]].groupby(
        "sku_id").apply(lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku最后一天前三个月的销量
    fts = pd.merge(fts, ftsData[
        ftsData["data_date"] >= beforelastday_3_month][["sku_id", "goods_num"]].groupby(
        "sku_id").apply(lambda x: sum(x["goods_num"])), how="left", on='sku_id')
    # sku去年今日的平均价格
    fts = pd.merge(fts, ftsData[ftsData["data_date"] == lastyear_today][
        ["sku_id", "goods_price"]], how="left", on='sku_id')
    # sku去年今日的吊牌价格
    fts = pd.merge(fts, ftsData[ftsData["data_date"] == lastyear_today][
        ["sku_id", "orginal_shop_price"]], how="left", on='sku_id')
    fts = fts.join(s1, how="left")
    fts.columns = ["sku_id", "sku_sale_lyear_today", "sku_sale_lyear_beforetoday_2_day",
                   "sku_sale_lyear_beforetoday_3_day",
                   "sku_sale_lyear_beforetoday_1_week", "sku_sale_lyear_aftertoday_2_day",
                   "sku_sale_lyear_aftertoday_3_day",
                   "sku_sale_lyear_aftertoday_3_day", "sku_sale_lyear_today_1_week",
                   "sku_sale_lyear_aftertoday_1_month",
                   "sku_sale_lyear_aftertoday_2_month", "sku_sale_lyear_aftertoday_3_month",
                   "sku_sale_before_lastday_1_month",
                   "sku_sale_before_lastday_2_month", "sku_sale_before_lastday_3_month", "sku_lyear_today_goods_price",
                   "sku_lyear_today_orginal_shop_price"]

    # 说明是训练数据
    if ftsLastDate != 20180316:
        fts["label"] = labelData["goods_num"]
    return fts
