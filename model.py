import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import pickle

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('HistoricalData_APPLE.csv',parse_dates = ["Date"], index_col = "Date")
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


sum(df.duplicated())


# In[5]:


len(df)


# In[6]:


df.info()


# In[7]:


df['Open']=df['Open'].str.replace('$','').astype(float)


# In[8]:


df['Close/Last']=df['Close/Last'].str.replace('$','').astype(float)


# In[9]:


df['High']=df['High'].str.replace('$','').astype(float)


# In[10]:


df['Low']=df['Low'].str.replace('$','').astype(float)
df.head()


# In[11]:


df.describe()


# In[12]:


print(len(df))


# In[13]:


df['Close/Last'].plot(figsize=(16,6))


# In[14]:


X  = df[['Open','High','Low','Volume']]
y = df['Close/Last']


# In[15]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X ,y , random_state = 0)


# In[16]:


X_train.shape


# In[17]:


X_test.shape


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import math
regressor = LinearRegression()


# In[19]:


regressor.fit(X_train,y_train)


# In[20]:


print(regressor.coef_)


# In[21]:


print(regressor.intercept_)


# In[22]:


predicted=regressor.predict(X_test)
print(X_test)


# In[23]:


predicted.shape


# In[24]:


dframe=pd.DataFrame(y_test,predicted)


# In[25]:


dfr=pd.DataFrame({'Actual':y_test,'Predicted':predicted})


# In[26]:


print(dfr)


# In[27]:


dfr.head(25)


# In[28]:
    
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


from sklearn import metrics 
print('Results of Linear Regression:\n')
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,predicted))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,predicted))
print('Root Mean Squared Error:',math.sqrt(metrics.mean_squared_error(y_test,predicted)))



# In[29]:


regressor.score(X_test,y_test)*100


# In[30]:


graph=dfr.head(20)


# In[31]:


graph.plot(figsize=(16,6))
graph.plot(kind='bar')

