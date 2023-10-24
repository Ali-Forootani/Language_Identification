#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:28:19 2023

@author: forootani
"""


import pandas as pd
import string
import re
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#######################################

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


df.info()
print(df.labels.value_counts())
df[df.labels == 'nl'].sample(20)
df[df.labels == 'ar'].sample(30)

#######################################
#######################################

def remove_symbols_and_numbers(text):        
        text = re.sub(r'[{}]'.format(string.punctuation), '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[@]', '', text)

        return text.lower()
    
    
def remove_english_letters(text):        
        text = re.sub(r'[a-zA-Z]+', '', text)
        return text.lower()

######################################
######################################

x0 = df.apply(lambda x: remove_english_letters(x.text) if 
              x.labels in ['pt','bg',
                           'vi','fr','nl','el', 'de', 'hi', 'it', 'ar', 'es',
                             'tr', 'sw', 'ur', 'pl', 'ru', 'th', 'zh', 'ja']
              else x.labels, axis = 1)


######################################
######################################

x = x0.apply(remove_symbols_and_numbers)

y = df['labels']

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    test_size=0.1,
                                                    random_state=42)



vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char')


model = pipeline.Pipeline([
    ('vectorizer', vectorizer),
    ('clf', LogisticRegression())])

model.fit(x_train,y_train)


y_pred = model.predict(x_test)


accuracy = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)



plt.figure(figsize=(30,20))
sns.heatmap(cm, annot = True)
plt.show()


def predict(text):
    lang = model.predict([text])
    print('The Language is in',lang[0])


predict("LANGUAGE DETECTION MODEL CHECK")

predict("come stai?")

predict("ich habe hunger!")
predict("haben si einen apfel")

