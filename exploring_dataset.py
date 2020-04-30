import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#pd.pandas.set_option('display.max_columns')
df = pd.read_csv('train.csv')
#print(df.head(5))
na_features = [features for features in df.columns if df[features].isnull().sum()>1]

"""
for feature in na_features:
   print(feature, np.round(df[feature].isnull().mean(),4), ' % missing values')
    data = df.copy()
    data[feature] = np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
"""
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
#print(len(numerical_features))
year_features = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

# Analyse dependency of year feature with the price
# We are just exploring the dataset to see trends in the data
"""
df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs Year Sold')
plt.show()
"""
# here we see that as year progresses the house prices are dropping
# again this was for analysis so we take a note of our observationn and comment out
# for detailed analysis we compare and see the difference between year of selling 
# and other year parameters like YearBuilt, yearGarageBlt

"""
for feature in year_features:
    if feature is not 'YrSold':
        data = df.copy()
        data[feature] = data['YrSold']-data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
"""
# In above analysis we note how all year variables behave similarly
# and we observe older the house lower the price, newer the house higher is the price 
# We can use this information to eliminate some year parameters as they behave in a v similar manner

discrete_features = [feature for feature in numerical_features if len(df[feature].unique()) < 25 and feature not in year_features+['Id']]
#print(len(discrete_features))

# It is important to analyse dependency of each variable with the output
# during the analysis stage as we want to see dependency

# so analysing dependency of discrete features
"""
for feature in discrete_features:
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale price')
    plt.show()
"""
# Key observation was that Overall Quality is a discrete variable 
# whose value grows proportionally (exponentially) wrt to Price ('Output')

continuous_features = [feature for feature in numerical_features if feature not in discrete_features+year_features+['Id'] ]
"""
for feature in continuous_features:
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.hist()
    plt.xlabel(feature)
    plt.ylabel('Sale price')
    plt.show()
"""
# We use histograms to study continuous data, we observe that 
# some parameters have a gaussian distribution but most are skewed
# It is very important while solving regression problem to del with gaussian distributions
# So we do log transformations or normalisations to convert skewed data into gaussian distribution


# Let's convert our skewed continuous features to a gaussian spread
# We will be using logarithmic transformation
"""
for feature in continuous_features:
    data = df.copy()
    # log can't be applied if we have a zero in the cell
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('Sale Price')
        plt.show()
"""
# We note that on applying log transformations a monotonic realtion
# Positive correlation is seen that all variables increase with price
# Might be useful later on



# Dealing with Outliers
# Box plot doesnot work with categorical features ***
"""
for feature in continuous_features:
    data = df.copy()
    # log can't be applied if we have a zero in the cell
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        # We use boxplot
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title('Identify the outliers')
        plt.show()
"""
# Note - There are plenty of outliers so we must transform and deal with them ahead

# LETS deal with categorical features now
categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']

# now lets find cardinality, that is
# How many unique categories are there in each feature

"""
for feature in categorical_features:
    print(feature,': ',len(df[feature].unique()))
"""
# Note some feATURES have too many categories
# Probably it is a continuous variable OR
# It is a category and we need to treat it separaetly 
# Other features can be dealt using one hot encoding


# Let's study the dependency of categorical features with SalePrice now
"""
for feature in categorical_features:
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.show()
"""
# We dont see any significant relation with the price 
# So they are probably nominal categories
# Had there been a direct relation it would have been ordinal




"""
Now we have completed our exploratory data analysis,
studied all features extensively and made key notes
as what all changes and features have to be dealt with

Let's proceed to the feature engineering part after this
"""