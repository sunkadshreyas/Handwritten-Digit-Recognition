#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:24:23 2020

@author: shreyas
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

dataset = pd.read_csv('mnist_train.csv')
x = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(x,y, test_size=0.25)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Accuracy Score : ",end="")
print(score*100)