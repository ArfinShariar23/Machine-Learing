#!/usr/bin/env python
# coding: utf-8

# #### Importing Essential Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb


# #### Importing Essential Module here

# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# #### Importing Dataset using Pandas Libriaries

# In[3]:


home = pd.read_csv("HomePrice2.csv")


# In[4]:


home


# #### Checking Our Dataset

# In[5]:


home.head()


# #### Checking Shape

# In[6]:


home.shape


# #### Checking for Null Values

# In[7]:


home.isnull().sum()


# #### Data Visualization

# In[8]:


plt.scatter(home['Area'],home['Price'],marker = '*',color = 'Red')
plt.xlabel("Area in sq feet")
plt.ylabel("Price in bdt tk.")
plt.title("Home Prices in Dhaka")


# #### Spliting Dataset

# In[9]:


x = home.iloc[:,0:-1]
y = home.iloc[:,-1]


# In[10]:


x


# In[11]:


y


# In[12]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)


# In[13]:


x_train


# In[14]:


x_test


# In[15]:


y_train


# In[16]:


y_test


# #### Linear Regression Model Fitting

# In[17]:


LR = LinearRegression()


# In[18]:


LR.fit(x_train, y_train)


# In[19]:


LR.predict(x_test)


# In[20]:


plt.scatter(home['Area'],home['Price'],marker = '*',color = 'Red')
plt.xlabel("Area in sq feet")
plt.ylabel("Price in bdt tk.")
plt.title("Home Prices in Dhaka")
plt.plot(home.Area, LR.predict(home[['Area']]))


# In[21]:


LR.predict([[3600]])


# In[22]:


LR.predict([[4400]])


# ##### Checking Co-efficient

# In[23]:


LR.coef_


# ##### Checking for Intercept

# LR.intercept_

# ##### Calculate Checking

# In[24]:


y = 15.77*4400+2629.84
print(y)

