#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 08:44:22 2017

@author: magalidrumare
"""

#libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#import the dataset 
dataset=pd.read_csv('Wine.csv')
X=dataset.iloc[:,0:13].values 
y=dataset.iloc[:,13].values 
                           
#Import the dataset 
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values 
y=dataset.iloc[:,4].values 
                                         
#Train and Test set 
from sklearn.model_selection import train_test_split 
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Features Scaling 
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Applying PCA (Linear Separation)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train= pca.fit_transform(X_train) 
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_

from sklearn.decomposition import PCA 
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
variance_expained=pca.explained_variance_ratio_

from sklearn.decomposition import PCA 
pca=PCA(n_component=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
variance_explained=pca.explained_variance_ratio_

#Apply LDA (Linear Separation)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
lda=LDA(n_components=2)
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.transform(X_test)

from sklearn.discrimnant_analysis import LinearDiscriminantAnalysis as LDA 
lda=LDA(n_components=2)
X_train=lda.fit_tranform(X_train,y_train)
X_test=lda.transform(X_test)

#Apply Kernel PCA (Non separable dataset)
from sklearn.decomposition import KernelPCA 
kpca=KernelPCA(n_components=2, kernel='rbf')
X_train=kpca.fit_transform(X_train)
X_test=kpca.transform(X_test)

#Feature Extraction 
#PCA principal componant analysis 
from sklearn.decomposition import PCA 
pca=PCA(n_component=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
variance_explained=pca.explained_variance_ratio_

#LDA linear discriminant analysis 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
lda=LDA(n_components=2)
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.tranform(X_test)

#PCA Kernel ( donnée no séparable linéairement)
from sklearn.decomposition import KernelPCA 
kpca=KernelPCA(n_component=2, kernel='rbf')
X_train=kpca.fit_transform(X_train)
X_test=kpca.transform(X_test)



#Logistic regression classifier 
from sklearn.linear_model import LogisticRegression 
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Prédiction 
y_pred=classifier.predict(X_test)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test,y_pred)

#Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
