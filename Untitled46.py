
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle


# In[3]:


data=pd.read_excel("default_of_credit_card_clients.xls");


# In[4]:


data2=data.drop_duplicates(['ID'])


# In[8]:


data=data2


# In[9]:


aa=data['PAY_1']=='Not available'
aa.sum()


# In[10]:


data['PAY_1']=pd.to_numeric(data.PAY_1,errors='coerce')


# In[11]:


data['PAY_1'].isnull().sum()


# In[12]:


data=data.dropna()
data['PAY_1'].isnull().sum()


# In[13]:


data['PAY_1']=data['PAY_1'].astype(int)


# In[14]:


X=data[['SEX','MARRIAGE','AGE','BILL_AMT1','EDUCATION','PAY_1']]
y=data['default payment next month']


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=24)
model=LogisticRegression()


# In[17]:


model.fit(X_train,y_train)


# In[18]:



pre=model.predict(X_test)
print(confusion_matrix(pre,y_test))
print(accuracy_score(y_test,pre))


# In[21]:


filename="finall.pkl"
pickle.dump(model,open(filename,'wb'))
