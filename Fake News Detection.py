#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

import re
import string
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[2]:


# Load Datasets
df=pd.read_csv('datasets/true_fake_data.csv')


# In[3]:


# df = df.drop(["title", "date","label","combined"], axis = 1)


# In[4]:


df = df.sample(frac = 1)


# In[5]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(subset=['text'],inplace=True)


# In[8]:


def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)  
    return text


# In[9]:


df["text"] = df["text"].apply(preprocess)


# In[10]:


df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[11]:


x = df["text"]
y = df["target"]


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# # Logistic Regression

# In[13]:


from sklearn.linear_model import LogisticRegression

pipe = Pipeline([('count_vectorization', CountVectorizer()),
                 ('tfidf_vectorization', TfidfTransformer()),
                 ('LR', LogisticRegression())])

LR = pipe.fit(x_train, y_train)


# In[14]:


pred_dt = LR.predict(x_test)

LR.score(x_test, y_test)


# In[15]:


print(classification_report(y_test, pred_dt))


# # Decision Tree Classification

# In[16]:


from sklearn.tree import DecisionTreeClassifier

pipe = Pipeline([('count_vectorization', CountVectorizer()),
                 ('tfidf_vectorization', TfidfTransformer()),
                 ('DT', DecisionTreeClassifier())])

DT = pipe.fit(x_train, y_train)


# In[17]:


pred_dt = DT.predict(x_test)

DT.score(x_test, y_test)


# In[18]:


print(classification_report(y_test, pred_dt))


# # Random Forest Classifier

# In[19]:


from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('count_vectorization', CountVectorizer()),
                 ('tfidf_vectorization', TfidfTransformer()),
                 ('RFC', RandomForestClassifier())])

RFC = pipe.fit(x_train, y_train)

# RFC = RandomForestClassifier(random_state=0)
# RFC.fit(xv_train, y_train)


# In[20]:


pred_rfc = RFC.predict(x_test)

RFC.score(x_test, y_test)


# In[21]:


print(classification_report(y_test, pred_rfc))


# # Model Testing

# In[23]:


# from sklearn.feature_extraction.text import TfidfVectorizer
#
def prediction_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "True News"
    
def detecting_fake_news(news_text):
    testing_news = {"text":[news_text]}
    new_def_test = pd.DataFrame(testing_news)
    
    new_def_test["text"] = new_def_test["text"].apply(preprocess) 
    new_x_test = new_def_test["text"]
    
    pred_LR = LR.predict(new_x_test)
    pred_DT = DT.predict(new_x_test)
    pred_RFC = RFC.predict(new_x_test)
#     vectorization = TfidfVectorizer()
#     new_xv_test = vectorization.transform(new_x_test)
    
#     pred_LR = LR.predict(new_x_test)
    
#     return print("\n\nLR Prediction: {} ".format(prediction_label(pred_LR[0])))
    

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nRFC Prediction: {} ".format(prediction_label(pred_LR[0]),                                                                                                      
                                                                                                              prediction_label(pred_DT[0]), 
                                                                                                              prediction_label(pred_RFC[0])))


# In[26]:


news_text = str(input())
detecting_fake_news(news_text)

import streamlit as st

st.title('Fake News Classifying App')
st.write('The data for the following example is originally from the National Institute of Diabetes and Digestive and Kidney Diseases and contains information on females at least 21 years old of Pima Indian heritage. This is a sample application and cannot be used as a substitute for real medical advice.')
# image = Image.open('data/diabetes_image.jpg')
# st.image(image, use_column_width=True)
st.write('Please select the model you want to use in the left sidebar then input the news text in the teaxt area below and click on the button below!')
text = st.text_area("Enter Text","Type Here")
st.button('Classify')
 st.success(result)