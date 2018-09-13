#!/usr/bin/python

import pandas as pd
import numpy as np
import datetime as dt

goodsPromoteData = pd.read_csv("../data/goods_promote_price.csv", sep=",", index_col=None, low_memory=False)
saleData = pd.read_csv("../features/saledata_sub_presku.csv", sep=",", index_col=None, low_memory=False)
useData = goodsPromoteData[goodsPromoteData["goods_id"].isin(saleData["goods_id"].unique())]
useData.to_csv("../features/goodspromote_sub_presku.csv", sep=",", index=None)
