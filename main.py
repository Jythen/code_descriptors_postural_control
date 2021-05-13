#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from stabilogram.stato import Stabilogram
from descriptors import compute_all_features


# In[2]:


forceplate_file_selected = "test.csv"


# In[3]:


data_forceplatform = pd.read_csv(forceplate_file_selected,header=[31],sep=",",index_col=0)
data_forceplatform.head()


# In[4]:


dft = data_forceplatform
X = dft.get(" My")/dft.get(" Fz")
Y = dft.get(' Mx')/ dft.get(' Fz')
X = X - np.mean(X)
Y = Y - np.mean(Y)
X = 100*X
Y = 100*Y

X = X.to_numpy()[4000:7000]
Y= Y.to_numpy()[4000:7000]


# In[5]:


plt.plot(X)
plt.plot(Y)


# In[6]:


data = np.array([X,Y]).T


# In[7]:


valid_index = (np.sum(np.isnan(data),axis=1) == 0)
print(np.sum(valid_index),len(data))


# In[8]:


stato = Stabilogram()
stato.from_array(array=data, original_frequency=100)


# In[9]:


plt.plot(stato.medio_lateral)
plt.plot(stato.antero_posterior)


# In[10]:


features = compute_all_features(stato)


# In[11]:


features

