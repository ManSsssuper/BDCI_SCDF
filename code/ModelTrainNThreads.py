#!/usr/bin/python

from code import ExtractFts
from code import ModelTrain

import pandas as pd
import numpy as np
import threading
from sklearn.ensemble import RandomForestRegressor


def modelTrain():
    trainData = pd.read_csv("../features/trainFts.csv", sep=",", index_col=None)
    testData = pd.read_csv("../features/testFts.csv", sep=",", index_col=None)
    print(trainData.shape)
    print(testData.shape)
    train = np.array(trainData.iloc[:, 2:])
    test = np.array(testData.iloc[:, 2:])
    trainX = train[:, :-5]
    trainY = [train[:, -5], train[:, -4], train[:, -3], train[:, -2], train[:, -1]]
    result = pd.DataFrame(testData["sku_id"]).reset_index(drop=True)
    for i in range(len(trainY)):
        myThread(trainX, trainY(i), test, "week" + str(i) + ".csv").start()
    #     result = pd.concat([result, pd.Series(clf.predict(test))], axis=1)
    # result.columns = ["sku_id", "week1", "week2", "week3", "week4", "week5"]
    # result.to_csv("../result/submit.csv", sep=",", index=None)


class myThread(threading.Thread):
    def __init__(self, trainX, trainy, test, week):
        threading.Thread.__init__(self)
        self.trainy = trainy
        self.trainX = trainX
        self.test = test
        self.week = week

    def run(self):
        self.getResult()

    def getResult(self):
        clf = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=4)
        print("trainfinish")
        clf.fit(self.trainX, self.trainy)
        pd.DataFrame(clf.predict(self.test)).to_csv("../result/" + self.week, sep=",", index=None)
