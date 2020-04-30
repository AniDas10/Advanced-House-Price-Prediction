import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('train.csv')
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df,df['SalePrice'],test_size=0.1,random_state=0)
"""
#Since we have a separate test data we dont need the above train_test_split()

# Dealing with MISSING VALUES (NaN)
#
# Let's handle categorical variables first
features_nan = [feature for feature in df.columns if df[feature].isnull().sum()>1 and df[feature].dtypes == 'O']

# We replace NaN categories with a common label called 'Missing'
def replace_cat_features(dataset, features_nan):
    data = dataset.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

df = replace_cat_features(df, features_nan)

numerical_with_nan = [feature for feature in df.columns if df[feature].isnull().sum()>1 and df[feature].dtypes != 'O']
for feature in numerical_with_nan:
    median = df[feature].median()

    # Always create a new feature to identify what values were Nan
    # So tmrw if they ask why is there a unknown value,
    # by checking this feature we know it was missing hence replaced for analysis purpose
    df[feature+'nan'] = np.where(df[feature].isnull(),1,0)
    df[feature].fillna(median, inplace=True)

#print(df[numerical_with_nan].isnull().sum())

# We have succesfully engineered categorical and numerical variables now

# Dealing with TEMPORAL VARIABLES
# We analysed that older the house, lower the price
# Thus replace these year features with one feature

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    df[feature]=df['YrSold']-df[feature]

num_features = ['LotArea','LotFrontage','SalePrice','1stFlrSF','GrLivArea']

for feature in num_features:
    df[feature] = np.log(df[feature])

# Handling rare categorical feature
# Replace all categories which occur less than 1% times in the whole dataset
# SO that such categories are grouped as one
categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
for feature in categorical_features:
    temp = df.groupby(feature)['SalePrice'].count()/len(df)
    temp_df = temp[temp>0.01].index
    df[feature] = np.where(df[feature].isin(temp_df), df[feature],'Rare_Var')

"""
Now we aim to map all categorical values in some order
which we select to be the mean of the output and sort it accoridngly
assuming that is hte priority
"""
for feature in categorical_features:
    labels_ordered = df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k:i for i,k in enumerate(labels_ordered,0)}
    df[feature] = df[feature].map(labels_ordered)

## Feature Scaling
feature_scale = [feature for feature in df.columns if feature not in ['Id','SalePrice']]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[feature_scale] = scaler.fit_transform(df[feature_scale])

"""
Now our input data is cooked and can be fed as the training data
We will repeat the same for test_data
And save both as csv files
"""
df.to_csv('engg_X_train.csv', index=False)
