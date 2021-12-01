#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.tree import export_text


# In[12]:


iris = load_iris()


# In[13]:


clf = DecisionTreeClassifier(random_state=0,max_depth=2)


# In[16]:


cross_val_score(clf,iris.data,iris.target,cv=10)


# In[18]:


X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.5,random_state=0) 


# In[20]:


clf.fit(X_train,y_train)
clf.predict(X_test)


# In[23]:


clf.fit(X_train,y_train)
clf.predict(X_test)


# In[24]:


print("Avg. Accuracy",clf.score(X_train,y_train))  


# In[28]:


clf.predict_log_proba(X_test)


# In[29]:


clf.predict_proba(X_test)
clf.cost_complexity_pruning_path(X_train,y_train)   
clf.fit(X_train,y_train)
clf.get_depth()
clf.get_n_leaves()
plot_tree(clf)  
r = export_text(clf, feature_names=iris['feature_names'])  
r


# In[ ]:




