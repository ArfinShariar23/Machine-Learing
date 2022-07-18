#!/usr/bin/env python
# coding: utf-8

# #### Importing Essential Libraries Here

# In[1]:


import numpy as np
import pandas as pd
import seaborn as snb
import matplotlib.pyplot as plt


# In[78]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# #### Importing CSV file here

# In[2]:


car = pd.read_csv("car driving risk analysis.csv")


# #### Checking Data Set

# In[3]:


car


# In[4]:


car.head()


# In[5]:


car.tail()


# In[7]:


car.shape


# In[8]:


car.describe()


# In[9]:


car.isnull().sum()


# #### Data Visualization

# In[26]:


plt.scatter(car['speed'],car['risk'],marker = '+',color = 'red',alpha = 0.5)
plt.xlabel("Car Speed")
plt.ylabel("Car Risk")
plt.title("Car Driving Risk Analysis")
plt.show()


# #### Spliting Dataset

# In[11]:


x = car.iloc[:,0:-1]
y = car.iloc[:,-1]


# In[14]:


x.head()


# In[15]:


y.head()


# #### Train_Test_Spliting Here

# In[16]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[27]:


x_train


# In[28]:


x_test


# In[29]:


y_train


# In[30]:


y_test


# ##### Fitting Model in Linear Regression

# In[32]:


LR = LinearRegression()


# In[33]:


LR.fit(x_train,y_train)


# In[50]:


predict = LR.predict(x_test)


# In[51]:


predict


# In[41]:


LR.predict([[212]])


# In[48]:


LR.predict([[95]])


# In[70]:


LR.coef_


# In[71]:


LR.intercept_


# In[63]:


plt.scatter(car['speed'],car['risk'],marker = '+',color = 'red')
plt.xlabel("Car Speed")
plt.ylabel("Car Risk")
plt.title("Car Driving Risk Analysis")
plt.plot(car.speed, LR.predict(car[['speed']]))


# In[76]:


print("This is not a Good Example of Dataset")


# In[77]:


print("Thank you")


# In[ ]:




