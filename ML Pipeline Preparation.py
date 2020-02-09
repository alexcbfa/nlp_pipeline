#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[2]:


# import libraries
import pandas as pd
import numpy as np
import re


import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[29]:


# import statements
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy import sparse


# In[4]:


# load data from database
engine = create_engine('sqlite:///Alexandre.db')
df = pd.read_sql_table('Udacity', engine)
X = df.loc[:,'message']
Y = df.iloc[:,4:]

y_columns = Y.columns


# ### 2. Write a tokenization function to process your text data

# In[5]:


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    return words


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[6]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[8]:


def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()
#     classification_report = 
    class_rep_dict = {}
    for label_col in range(len(labels)):
        y_true_label = y_test[:, label_col]
        y_pred_label = y_pred[:, label_col]
        class_rep_dict[labels[label_col]] = classification_report(y_pred=y_pred_label, y_true=y_true_label, labels=labels)

    print("Labels:", labels)
    print("Accuracy:", accuracy)
#     print("Classification report: ", classification_report)
    for label, matrix in class_rep_dict.items():
        print("Classification report for output {}:".format(label))
        print(matrix)
    
    

    


# In[9]:


Y_test =Y_test.values


# In[10]:


pipeline.fit(X_train, Y_train)
Y_pred = pipeline.predict(X_test)
display_results(Y_test, Y_pred)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[40]:


parameters = {"clf__estimator__n_estimators": [10, 100, 250],
     "clf__estimator__max_depth":[8],
     "clf__estimator__random_state":[42]}


rkf = RepeatedKFold(
    n_splits=10,
    n_repeats=2,
    random_state=42
)


cv = GridSearchCV(
    pipeline,
    parameters,
    cv=rkf,
    scoring='accuracy',
    n_jobs=-1)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[41]:


model = cv.fit(X_train,Y_train).best_estimator_ 


# In[39]:



Y_pred = model.predict(X_test)
display_results(Y_test, Y_pred)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




