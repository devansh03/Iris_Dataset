#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:19:08 2019

@author: devansh
"""

import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
x=iris.data
y=iris.target
import sklearn.model_selection as model_selection
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.2,random_state=4)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

svclassifier = SVC()
svclassifier.fit(xtrain, ytrain)
ypred = svclassifier.predict(xtest)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))

'''
svclassifier = SVC(kernel='linear')
svclassifier.fit(xtrain, ytrain)
ypred = svclassifier.predict(xtest)
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))'''