#!/usr/bin/python

import pandas as pd
import numpy as np
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

    for timeSlice in trainTimeSlice:
        labelData = saleData[(saleData["data_date"] >= timeSlice[2]) & (saleData["data_date"] <= timeSlice[3])]
        ftsData = saleData[(saleData["data_date"] >= timeSlice[0]) & (saleData["data_date"] <= timeSlice[1])]
        ftsData = ftsData[ftsData["sku_id"].isin(labelData["sku_id"])]
        extract(ftsData, labelData, timeSlice)


def extract(ftsData, labelData, timeSlice):
    # 获取标记
    week1 = timeSlice[2]
    week2 = int(dt.datetime.strftime((dt.datetime.strptime(str(week1), "%Y%m%d") + dt.timedelta(7)), "%Y%m%d"))
    week3 = int(dt.datetime.strftime((dt.datetime.strptime(str(week2), "%Y%m%d") + dt.timedelta(7)), "%Y%m%d"))
    week4 = int(dt.datetime.strftime((dt.datetime.strptime(str(week3), "%Y%m%d") + dt.timedelta(7)), "%Y%m%d"))
    week5 = int(dt.datetime.strftime((dt.datetime.strptime(str(week4), "%Y%m%d") + dt.timedelta(7)), "%Y%m%d"))
    labels = pd.DataFrame(labelData["sku_id"].unique())
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
