# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

# In[2]:


goodsInfo = pd.read_csv("D:\competition\BDCI_SCDF\data\goodsinfo.csv", sep=",", index_col=None)

# In[3]:


print("数据量：", goodsInfo.shape)

# In[4]:


print("是否有空值：", goodsInfo.apply(lambda x: True if x.isna().unique().shape[0] == 2 else False, axis=0))
print("goods_id是否唯一：", goodsInfo["goods_id"].is_unique)

# In[5]:


print(goodsInfo.info())

# In[10]:


print("总共七级类目的值：\n", goodsInfo.iloc[:, 1:].apply(lambda x: len(x.unique()), axis=0))

# In[11]:


print("季节属性值\n", goodsInfo["goods_season"].unique())

# In[12]:


goodsInfo = None

# In[4]:


goodsSale = pd.read_csv("D:\competition\BDCI_SCDF\data\goodsale.csv", sep=",", index_col=None, low_memory=False)

# In[5]:


print("数据量：", goodsSale.shape)
print("是否有空值：\n", goodsSale.apply(lambda x: True if x.isna().unique().shape[0] == 2 else False, axis=0))
print("起止日期%d-%d" % (goodsSale["data_date"].max(), goodsSale["data_date"].min()))
print(goodsSale.info())

# In[8]:


print("goods_id是否唯一:", goodsSale["goods_id"].is_unique)
print("sku_id是否唯一:", goodsSale["sku_id"].is_unique)
dstr = goodsSale["data_date"].apply(lambda x: str(x))
print("goods_id与data_date连接是否唯一:", (dstr + goodsSale["goods_id"]).is_unique)
print("sku_id与data_date连接是否唯一:", (dstr + goodsSale["sku_id"]).is_unique)
print("goods_id数量", len(goodsSale["goods_id"].unique()))
print("sku_id数量", len(goodsSale["sku_id"].unique()))
print("数据总量：", len(goodsSale))

# In[9]:


goodsSale = None

# In[11]:


goodsdaily = pd.read_csv("D:\competition\BDCI_SCDF\data\goodsdaily.csv", sep=",", index_col=None)
print("goods_id数量", len(goodsdaily["goods_id"].unique()))
print("数据总量：", len(goodsdaily))
goodsdaily = None

# In[12]:


gs_rela = pd.read_csv("D:\competition\BDCI_SCDF\data\goods_sku_relation.csv", sep=",", index_col=None)

# In[13]:


print("数据量：", gs_rela.shape)

# In[16]:


print("sku_id是否唯一：", gs_rela["sku_id"].is_unique)
print("goods_id是否唯一：", gs_rela["goods_id"].is_unique)
print("goods_id数量：", len(gs_rela["goods_id"].unique()))
print("是否有空值：\n", gs_rela.apply(lambda x: True if x.isna().unique().shape[0] == 2 else False, axis=0))

# In[17]:


gs_rela = None

# In[18]:


goodsPromotePrice = pd.read_csv("D:\competition\BDCI_SCDF\data\goods_promote_price.csv", sep=",", index_col=None)

# In[19]:


print("数据量：", goodsPromotePrice.shape)
print("是否有空值：", goodsPromotePrice.apply(lambda x: True if x.isna().unique().shape[0] == 2 else False, axis=0))
print("goods_id是否唯一：", goodsPromotePrice["goods_id"].is_unique)
print("goods_id数量：", len(goodsPromotePrice["goods_id"].unique()))

# In[20]:


print(goodsPromotePrice.info())

# In[21]:


print("起止日期%d-%d" % (goodsPromotePrice["data_date"].max(), goodsPromotePrice["data_date"].min()))

# In[23]:


dstr = goodsPromotePrice["data_date"].apply(lambda x: str(x))
print("goods_id与data_date连接是否唯一:", (dstr + goodsPromotePrice["goods_id"]).is_unique)

# In[25]:


print(goodsPromotePrice.iloc[:3, :])

# In[26]:


goodsPromotePrice = None

# In[27]:


marketing = pd.read_csv("D:\competition\BDCI_SCDF\data\marketing.csv", sep=",", index_col=None)

# In[29]:


print("数据量：", marketing.shape)
print("是否有空值：", marketing.apply(lambda x: True if x.isna().unique().shape[0] == 2 else False, axis=0))
print("data_date起止时间：%d-%d" % (marketing["data_date"].max(), marketing["data_date"].min()))
print("data_date是否唯一：", marketing["data_date"].is_unique)

# In[30]:


print("活动类型marketing种类", len(marketing["marketing"].unique()))
print("活动类型marketing值", marketing["marketing"].unique())

# In[31]:


print("活动节奏plan种类", len(marketing["plan"].unique()))
print("活动节奏plan值", marketing["plan"].unique())
