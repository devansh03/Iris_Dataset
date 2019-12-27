#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:17:17 2019

@author: devansh
"""

import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
x=iris.data
y=iris.target

import sklearn.model_selection as model_selection
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.2,random_state=4)

krange=range(1,25)
scores={}
s_list=[]
for k in krange:
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(xtrain,ytrain)
  ypred = clf.predict(xtest)
  scores[k] = accuracy_score(ytest,ypred)
  s_list.append(accuracy_score(ytest,ypred))

import matplotlib.pyplot as plt
plt.plot(krange,s_list)
plt.xlabel('value of k for knn')
plt.ylabel('Testing accuracy')

knn=KNeighborsClassifier()
knn.fit(x,y)
classes={0:'setosa',1:'versicolor',2:'virginica'}
x_new=[[3,4,5,2],[5,4,2,2]]
y_predict=knn.predict(x_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])

from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))
