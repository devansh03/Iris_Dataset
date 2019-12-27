#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:25:46 2019

@author: devansh
"""

from sklearn import datasets
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
x=iris.data
y=iris.target

import sklearn.model_selection as model_selection
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.2,random_state=4)

'''from sklearn.naive_bayes import MultinomialNB
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
clf=MultinomialNB()
clf.fit(xtrain,ytrain)
ypred=clf.predict(xtest)
scores=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))

from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))'''



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
clf=GaussianNB()
clf.fit(xtrain,ytrain)
ypred=clf.predict(xtest)
scores=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))

from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))

'''
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
clf=BernoulliNB(binarize=0.0)
clf.fit(xtrain,ytrain)
ypred=clf.predict(xtest)
scores=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))

from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))'''