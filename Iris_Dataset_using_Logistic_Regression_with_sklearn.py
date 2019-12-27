#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:17:01 2019

@author: devansh
"""

import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()

x=iris.data
y=iris.target

import numpy as np
x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)
clf=LogisticRegression()
clf.fit(x,y)
ex=clf.predict([[2.6]])
print(ex)
pred=clf.predict(x)
print(len(pred))

x_new=np.linspace(0,3,150).reshape(-1,1)
y_actual=clf.predict(x_new)
print(y_actual)

from sklearn.metrics import classification_report
print(classification_report(y_actual,pred))
