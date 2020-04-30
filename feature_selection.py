import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importing all prediction and selection tools
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


"""
Now that we have engineered our dataset
It's time to use selective features so that 
we can see which features contributes more and can be
selected accordingly.
"""
df = pd.read_csv('engg_X_train.csv')
y_train = np.exp(df[['SalePrice']])
y_train.to_csv('y_train.csv')
X_train = df.drop(['Id','SalePrice'], axis=1)
y_train = df['SalePrice']

"""
For applying feature selection,
first we specifiy Lasso regression model
Select a suitable alpha (equivalent to penalty)
The bigger the alpha, the lesser features will be selected

Followed by selecting features in which coeff are non-zero
"""
feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(X_train,y_train)

selected_features = X_train.columns[(feature_sel_model.get_support())]

print("{0}/{1} features have been selected".format(len(selected_features),X_train.shape[1]))

X_train = X_train[selected_features]
#print(X_train.head(3))
X_train.to_csv('selected_X_train.csv')

"""
Always engineering your dataset before feature selection and 
training the model
"""