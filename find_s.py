#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[7]:


data = pd.read_csv('Book1.csv')


# In[8]:


data


# In[9]:


train = np.array(data)[:,:-1]


# In[10]:


train


# In[20]:


label = np.array(data)[:,-1]
label


# In[22]:


h = train[0]
h


# In[25]:


for i,val in enumerate(label):
    if val=="yes":
        temp_h=train[i]
        for j in range(len(h)):
            if(h[j]!=temp_h[j]):
                h[j]="?"
            else:
                pass
print(h)
        


# In[ ]:




