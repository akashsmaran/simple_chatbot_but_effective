from flask import request, url_for, Flask
from flask_api import FlaskAPI, status, exceptions
#import pyrebase
import numpy as np
import sys
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline



Query = ['hey',
'hi',
'wats up',
'who are you',
'good morning',
'Ssup',
'Whats up',
'Tell me about yourself',
'how are you',
'Where should I eat',
'eating',
'eat',
'I am hungry',
'I want to eat',
'restaurants near me',
'places to eat near me',
'next',
'calories'
]
Intent=[
'Greeting',
'Greeting',
'Greeting',
'Greeting',
'Greeting',
'Greeting',
'Greeting',
'Greeting',
'Greeting',
'Zomato',
'Zomato',
'Zomato',
'Zomato',
'Zomato',
'Zomato',
'Zomato',
'second',
'second']

def text_process(description):
        nopunc = [char for char in description if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return [words for words in nopunc.split() if words.lower() not in stopwords.words('english')]


chatcontext = "hi"
    #data = pd.read_csv('code.csv')
    #main = data['Query']
    #intent = data['Intent']
testing = []    
testing.append(chatcontext)
    
pipeline = Pipeline([        
('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(Query,Intent)    
predictions = pipeline.predict(testing)    
print(predictions)
    
