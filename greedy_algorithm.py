# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from itertools import compress
from sklearn.metrics import accuracy_score, recall_score
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# target and features
# target feature is diagnosis
df = pd.read_excel("data/data.xlsx")
df.loc[df['diagnosis'] == 'M', 'diagnosis'] = 1
df.loc[df['diagnosis'] == 'B', 'diagnosis'] = 0

Y = df["diagnosis"]
# also removing unnecessary column id
X = df.drop(['diagnosis', 'id'], axis=1)


import sklearn.metrics as mtr
from sklearn import linear_model
model = linear_model.LogisticRegression(max_iter=3000,solver='lbfgs')

#greedy algorithm with LogisticRegression
def best_n_features(X, Y, n):
    list_best_features = []
    for i in range(1,n+1):
        accur_max = -math.inf
        column_min = ''
        for j in X:
            if j in list_best_features:
                continue
            if i == 1:
                k = j
            else:
                k = list_best_features + [j]
            F_j = X[k].to_numpy()
            if i == 1:
                F_j = F_j.reshape(-1,1)
            model.fit(F_j,Y)
            Y_pred = model.predict(F_j)
            accuracy = mtr.accuracy_score(y_pred=Y_pred,y_true=Y)
            if accuracy >= accur_max:
                accur_max = accuracy
                column_min = j
        list_best_features.append(column_min)
        print(list_best_features[-1])

#tests:
#best_n_features(X, Y, 3)
#best_n_features(X, Y, 5)




