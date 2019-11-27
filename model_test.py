# -*- coding: utf-8 -*-
"""
| *@created on:* 26/11/19,
| *@author:* Umesh Kumar,
| *@version:* v0.0.1
|
| *Description:*
|
| *Sphinx Documentation Status:* Complete
|
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# create linear regression object
reg = linear_model.LinearRegression()


def predict(data):
    """

    :param data:
    :return:
    """
    return reg.predict(data)


df = pd.read_csv("data/lucas0_train.csv")
label = df["Lung_cancer"]
input_data = df.drop("Lung_cancer", axis=1)

X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=0.4, random_state=1)

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

predictions = predict(X_test)

# ROC score
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)
print("ROC Score: ", roc_auc)

# GINI
gini = 2 * roc_auc - 1
print("GINI Score: ", gini)

# Predictions for confusion matrix, accuracy
predictions = [1 if output > 0.5 else 0 for output in predictions]

# Accuracy Score
accuracy = accuracy_score(y_test, predictions)
print("accuracy: ", accuracy)

# Confusion Matrix
confusion_matrix_value = confusion_matrix(y_test, predictions, labels=None, sample_weight=None)
print("Confusion Matrix\n", confusion_matrix_value)

print("Trained model")
print("-"*100)

