#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("C:/Users/USER/Downloads/df.csv")


# # Exploratory data analysis

# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


df.shape


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.shape


# In[10]:


df["Brand"].unique()   ### finding unique value


# In[11]:


df["Brand"].value_counts(normalize=True)*100


# In[12]:


df["Engine Type"].unique()


# In[13]:


df["Registration"].unique()


# In[14]:


df["Engine Type"].value_counts(normalize=True)*100


# In[15]:


df["Registration"].value_counts(normalize=True)*100


# In[16]:


plt.pie(df["Engine Type"].value_counts(normalize=True)*100,labels=['Petrol', 'Diesel', 'Gas', 'Other'])


# In[17]:


plt.pie(df["Registration"].value_counts(normalize=True)*100,labels=['yes','no']) ## normalize is for value to come in percentage


# In[18]:


df["Body"].unique()   ### finding unique value


# In[19]:


plt.pie(df["Body"].value_counts(normalize=True)*100,labels=['sedan', 'van', 'crossover', 'vagon', 'other', 'hatch']) 


# In[20]:


df.info()


# In[21]:


df.describe().T      ## .T only for transpose  ## 5 pont summary


# In[22]:


df.skew()


# In[23]:


sns.pairplot(df)


# In[24]:


df.corr()


# In[25]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)    ### heatmap for co relation


# In[26]:


df.dtypes


# In[27]:


df["Price"]=df["Price"].astype("int")    ### for changing float data type into int .astype
df["EngineV"]=df["EngineV"].astype("int")


# In[28]:


df.dtypes


# In[29]:


### if there is float in dtype change into integer


# In[30]:


## outlier encoding model building


# In[31]:


sns.boxplot(df)


# In[38]:


from sklearn.preprocessing import LabelEncoder


# In[39]:


le=LabelEncoder()  ### for converting data type of object label encoder to int


# In[40]:


df.dtypes


# In[41]:


df["Brand"]=le.fit_transform(df["Brand"])
df["Body"]=le.fit_transform(df["Body"])
df["Engine Type"]=le.fit_transform(df["Engine Type"])
df["Registration"]=le.fit_transform(df["Registration"])
df["Model"]=le.fit_transform(df["Model"])


# In[42]:


df.head()


# In[44]:


df.dtypes


# # MOdel Building

# In[47]:


#1. spliting data into tarin and test   ### without scaling
#2. fitting  model on train data
#3. test it on test data


# In[48]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score    ## r2 for evaluate model


# In[49]:


x=df.drop(["Price"],axis=1)  #### price nko ahe to factor affect karat nhi so drop kel model building sathi


# In[50]:


y=df["Price"]


# In[51]:


y.head()


# In[52]:


x.head()


# In[54]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


# In[56]:


print(x_train.shape)  ## x madhe 8 column ahet
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)  ## y madhe onl orice ahe so ekch n , ahe


# In[57]:


lr=LinearRegression()


# In[58]:


lr.fit(x_train,y_train)  ### lin res sathi train data pathvLA


# In[59]:


y_true,y_pred=y_test,lr.predict(x_test)


# In[60]:


y_true


# In[61]:


y_pred


# 
# # Model Evaluation

# In[62]:


## actual and predicted var evalu aste


# In[69]:


r2_score(y_true,y_pred)   ### jevdh jast % tevdhi good accuracy


# In[64]:


## (100-38=62% accuracy yet ahe)


# In[66]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[67]:


mean_squared_error(y_true,y_pred)


# In[68]:


mean_absolute_error(y_true,y_pred)


# # model building with Scaling

# In[70]:


from sklearn.preprocessing import StandardScaler


# In[76]:


sc=StandardScaler()


# In[77]:


x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.transform(x_test)


# In[78]:


lrs=LinearRegression()


# In[79]:


lrs.fit(x_train_sc,y_train)


# In[80]:


y_true,y_pred=y_test,lrs.predict(x_test_sc)


# In[83]:


r2_score(y_true,y_pred)


# In[82]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[ ]:




