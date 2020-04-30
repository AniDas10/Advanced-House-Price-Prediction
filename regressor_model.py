import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X_train = pd.read_csv('selected_X_train.csv')
y_train = pd.read_csv('y_train.csv')
y_train.drop([y_train.columns[0]], axis=1, inplace=True)
X_train.drop([X_train.columns[0]], axis=1, inplace=True)

train_x = X_train.iloc[:1001,]
test_x = X_train.iloc[1001:,]
train_y = y_train.iloc[:1001,]
test_y = y_train.iloc[1001:,]
#print(train_y.head(5))
import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(train_x, train_y)

from sklearn.ensemble import RandomForestRegressor
import pickle
filename = 'final_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
y_pred = classifier.predict(test_x)
from sklearn.metrics import r2_score, mean_squared_error

print(r2_score(test_y, y_pred))
