from code import ExtractFts
from code import ModelTrain
import pandas as pd
import numpy as np

# 预测日标记
testDateLabels = range(20180501, 20180605, 1)
result = pd.read_csv("../data/submit_example.csv", sep=",", index_col=None)
result = result["sku_id"]
for date in testDateLabels:
    trainData, testData = ExtractFts.extractFts(date, dataSubSkuToFile=False, saveFts=True)
    print(trainData)
    # result=pd.concat([result,ModelTrain.modelTrain(trainData,testData)],axis=1)
