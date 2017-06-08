#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:40:30 2017

@author: magalidrumare
"""

#import the libraries 
import numpy as np 
import matplotlib as plt 
import pandas as pd 

#import tsv file 
#quoting=3  allow to ignore the ""
dataset=pd.read_csv('Restaurant_reviews.tsv', delimiter='\t', quoting=3)
dataset['Review'][0]
#'Wow...loved this place' 

#Cleaning the texts
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
#result = 'Wow   Loved this place '
    review=review.lower()
#result ='wow    loved  this place ' 
    review=review.split()
#result = ['wow', 'loved', 'this', 'place' ]
    review=[word for word in review if not word in stopwords.words('english')]
#result = ['wow' 'loved' 'place']
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#result =['wow', 'love', 'place']
    review=' '.join(review)
#result = 'wow love place' (back to the sentence'/ inverse de split)
    corpus.append(review)
    
#Create the bag of words (Sparse Matrix)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
#can reduce with PCA 
y=dataset.iloc[:,1].values 
                          

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
              
              
              
              
              
              
              


    
    
    



            
             
              




