
# coding: utf-8

# In[27]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Dataset

# In[3]:


data=pd.read_csv("Santarosa.csv")


# # Splitting Dataset to Features and Label

# In[23]:


X=data[['TempOut', 'OutHum', 'Bar']]
y=data.Rain


# # Normalizing Label Column

# In[38]:


lab_enc = preprocessing.LabelEncoder()
y = lab_enc.fit_transform(y)


# # Splitting to Train and Test Dataset

# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=324)


# # Selecting Model and Applying Our X_train and y_train Dataset to this Model

# In[40]:


humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train, y_train)


# # Making Prediction

# In[41]:


predictions = humidity_classifier.predict(X_test)


# # Scoring to our Trained Model

# In[46]:


accuracy_score(y_true = y_test, y_pred = predictions)

