#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Prepare a classification model using SVM for salary data


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams


# In[3]:


rcParams['figure.figsize'] = 12, 8
import os


# In[4]:


print(os.listdir("D:\\360DigiTMG\\Black Box Technique SVM\\HANDS ON MATERIAL\\"))
train=pd.read_csv("D:\\360DigiTMG\\Black Box Technique SVM\\HANDS ON MATERIAL\\SalaryData_Train.csv")
test=pd.read_csv("D:\\360DigiTMG\\Black Box Technique SVM\\HANDS ON MATERIAL\\SalaryData_Test.csv")


# In[5]:


train.head()


# In[6]:


train.educationno.nunique()


# In[7]:


train.shape


# In[8]:


train.age.value_counts()


# In[9]:


print(1)


# In[10]:


print(2)


# In[11]:


# target=pd.get_dummies(train.salary)
test.head()


# In[12]:


# train.experience.apply(lambda x:x.split("-"))
train.hoursperweek.value_counts()
df=train.copy()


# In[13]:


df.capitalloss.fillna('None',inplace=True)
df.capitalgain.fillna('None',inplace=True)

