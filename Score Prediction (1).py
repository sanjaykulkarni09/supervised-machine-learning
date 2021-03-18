#!/usr/bin/env python
# coding: utf-8

#  # Supervised Learning Using Linear Regression

# ---
# ### What is supervised learning?
# Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a "reasonable" way (see inductive bias). This statistical quality of an algorithm is measured through the so-called generalization error.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## Reading the data

# In[2]:


url = 'http://bit.ly/w-data'

data = pd.read_csv(url)


# In[3]:


data.head()


# ## Visually Analysing the data

# In[4]:


sns.scatterplot(x=data['Hours'], y=data['Scores'], data=data)


# ## What is linear regression?
# 
# Linear regression is perhaps one of the most well known and well understood algorithms in statistics and machine learning.

# In[5]:


sns.regplot(x=data['Hours'], y=data['Scores'], data=data)


# ## Preparing the data for modelling

# In[6]:


X = data.drop('Scores', axis=1)
y = data['Scores']


# In[7]:


# Split the data into training data and validation data
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state = 0)


# ## Building the Model

# In[8]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)


# In[9]:


X_valid


# ## Making predictions for the Validation Data

# In[10]:


y_prediction = model.predict(X_valid)


# In[11]:


df = pd.DataFrame({'Actual': y_valid, 'Predicted': y_prediction})
df


# ##  Evaluating the model's performance

# In[12]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_valid, y_prediction)


# ## Predicting the Score for custom input

# In[13]:


hours = 9
model.predict([[hours]])


# In[ ]:




