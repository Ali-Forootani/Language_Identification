#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:16:35 2023

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

# Read a CSV file 'train.csv' into a DataFrame 'extra_df.'
extra_df = pd.read_csv('train.csv')

# Read another CSV file 'Language Detection.csv' into a DataFrame 'df.'
df = pd.read_csv('Language Detection.csv')
df.info()

# Count the occurrences of each unique value in the 'Language' column.
df.Language.value_counts()

# Sample two random rows from the 'df' DataFrame where the 'Language' is 'Russian' and 'Malayalam.'
df[df.Language == 'Russian'].sample(2)
df[df.Language == 'Malayalam'].sample(2)

# Define a function to remove symbols, numbers, and English letters from text.
def remove_symbols_and_numbers(text):
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[@]', '', text)
    return text.lower()

# Create a new Series 'x0' by applying the 'remove_symbols_and_numbers' function based on language.
x0 = df.apply(lambda x: remove_symbols_and_numbers(x.Text) if 
              x.Language in ['Russian','Malayalam','Hindi','Kannada','Tamil','Arabic']
              else x.Text, axis=1)



# Apply the 'remove_symbols_and_numbers' function to 'X0' and store the result in 'X1'.
x = x0.apply(remove_symbols_and_numbers)
print(x)

# Create a Series 'y' from the 'Language' column of the DataFrame.
y = df['Language']
print(y)

# Split the data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Create a TfidfVectorizer with specified parameters.
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char')

# Create a pipeline with the vectorizer and Logistic Regression classifier.
model = pipeline.Pipeline([
    ('vectorizer', vectorizer),
    ('clf', LogisticRegression())
])

# Fit the model on the training data.
model.fit(x_train, y_train)

# Make predictions on the test data.
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print accuracy and classification report.
print("Accuracy is:", accuracy)
print(classification_report(y_test, y_pred))

# Create a heatmap of the confusion matrix.
plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True)
plt.show()

# Define a function to predict the language of a given text.
def predict(text):
    lang = model.predict([text])
    print('The Language is in', lang[0])

############################################
############################################


predict("LANGUAGE DETECTION MODEL CHECK")
# French
predict("VÉRIFICATION DU MODÈLE DE DÉTECTION DE LA LANGUE")
# Arabic
predict("توففحص نموذج الكشف عن اللغة")
# Spanish
predict("VERIFICACIÓN DEL MODELO DE DETECCIÓN DE IDIOMAS")
# Malayalam
predict("ലാംഗ്വേജ് ഡിറ്റക്ഷൻ മോഡൽ ചെക്ക്")

# Russian
predict("ПРОВЕРКА МОДЕЛИ ОПРЕДЕЛЕНИЯ ЯЗЫКА")
# Hindi
predict('भाषा का पता लगाने वाले मॉडल की जांच')
# Hindi
predict(' boyit9h एनालिटिक्स alhgserog 90980879809 bguytfivb ahgseporiga प्रदान करता है')
# Italian
predict("ciao bello, come stai oggi?")
predict("ich habe hunger")
predict("come stai?")

#English
predict("I went to school yesterday and I saw some people sitting on the conrner")










