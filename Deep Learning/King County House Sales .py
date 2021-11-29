#!/usr/bin/env python
# coding: utf-8

# In[ ]:



### King County House Sales Regression Model


# In[49]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import array

# Loading required libraries


# In[4]:


data_path = 'dataset/kc_house_data.csv'
house_df = pd.read_csv(data_path)
house_df.head()

# Loading the dataset


# In[23]:





# Starting with data analyses & Data preperation



# In[5]:


house_df.shape

# Printing the shape and size of the data frame


# In[7]:


house_df.count()

# Printing the count of all the columns and checking for null


# In[8]:


house_df.describe()

# Getting insights about the dataframe


# In[11]:


house_df.dtypes

# Data types of all colums


# In[12]:


house_df['reg_year'] = house_df['date'].str[:4]
print(house_df['reg_year'])

# Adding a new column called reg_year


# In[14]:


house_df['reg_year'] = house_df['reg_year'].astype('int')

# Convert the data to integer


# In[15]:


house_df.dtyp es


# In[16]:


house_df['house_age'] = np.NAN

# Adding a new column called house_age


# In[17]:


house_df.head()


# In[19]:


for i,j in enumerate (house_df['yr_renovated']):
    if(j==0):
        house_df['house_age'][i] = house_df['reg_year'][i]-house_df['yr_built'][i]
    else:
        house_df['house_age'][i] = house_df['reg_year'][i]-house_df['yr_renovated'][i]

# Calculating house age        


# In[20]:


house_df.head()


# In[22]:


house_df.drop(['date','yr_built','yr_renovated','reg_year'], axis=1, inplace=True)
house_df.head()


# In[24]:


house_df.drop(['id','zipcode','lat','long'], axis=1, inplace=True)
house_df.head()

# Removing irrelevant columns


# In[25]:


house_df = house_df[house_df['house_age']!=-1]
house_df.describe()

# Removing all -1 values from house_age


# In[ ]:



#### Starting with Data Visualisation


# In[29]:


for i in house_df.columns:
    sns.displot(house_df[i])
    plt.show
    
# Plotting and visualising the distribution of variables    


# In[30]:


plt.figure()
sns.pairplot(house_df)
plt.show()

# Plotting a pair plot


# In[32]:


plt.figure(figsize=(20,10))
sns.heatmap(house_df.corr(),annot=True)
plt.show()

# Plotting a heat map
# Multi-Collinearity of House Attributes 


# In[33]:


for i in house_df.columns:
    sns.boxplot(x=house_df[i])
    plt.show()

# Plotting a box plot


# In[34]:



### Creating Regression Model with Keras


# In[38]:


X = house_df.drop('price', axis=1)
y = house_df['price']

# Splitting the dataframe into input X and output Y


# In[41]:


model = keras.Sequential()
model.add(layers.Dense(14, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1))

# Using the default linear activation function
# Defining the model


# In[42]:


# Compiling and fitting the model

model.compile(loss='mse', optimizer='adam')
# Optimizer is Adam , Loss function is Mean Square Error 

history = model.fit(X, y, validation_split=0.33, batch_size=32, epochs=100)
# This builds model for the first time

 


# In[47]:



### Visualising the results


# In[48]:


model.summary()

# Model summary


# In[46]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('no of epochs')
plt.legend(['train','test'],loc='upper left')
plt.show()
 


# In[50]:




### Predicting the price of a house using model


# In[52]:


import numpy
from numpy import array

Xnew = numpy.array([[2,3,1280,5550,1,0,0,4,7,2280,0,1440,5750,60]])
# Giving input random input data for prediction

Xnew = numpy.array(Xnew, dtype=numpy.float64)
# Converting the values into float


Ynew = model.predict(Xnew)
print(Ynew[0])
# Making the prediction


# In[ ]:




