#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


plt.rcParams['figure.figsize']=(12.0,9.0)


# In[4]:


data = pd.read_csv('dimensions.csv')


# In[5]:


x = data.iloc[:,0]
x


# In[7]:


y = data.iloc[:,1]
y


# In[8]:


plt.scatter(x,y)


# In[10]:


x_mean = np.mean(x)
x_mean


# In[12]:


y_mean = np.mean(y)
y_mean


# 

# In[16]:


num = 0
den = 0
for i in range(len(x)):
    num += (x[i]-x_mean)*(y[i]-y_mean)
    den += (x[i]-x_mean)**2
m = num/den
m


# In[17]:


c = y_mean - m*x_mean
m,c


# In[19]:


y_pred = m*x + c
y_pred


# In[20]:


plt.scatter(x,y)


# In[22]:


plt.plot([min(x),max(x)],[min(y_pred),max(y_pred)],color = 'red')


# In[ ]:




