# Advanced-House-Price-Prediction

## Solving an advanced house price prediction problem using ML pipelining principles. 

## Models used XGBoost, RandomForestRegressor and Lasso Regression (Feature Selection)
## R2_Score :: 86.5%


## Key steps involved in the pipelining:

### 1. Extensive Exploratory data analysis
### 2. Feature Engineering
### 3. Feature Selection
### 4. Training the Model


## Exploratory Data Analysis:

- Key observations and trends in the data were noted down
- All correlations within the variables and the output 'SalePrice' were monitored
- Distribution of missing values were noted.
- Temporal data features were noted.


## Feature Engineering:

- Handling of Missing values
- Substitution of Categorical variables 
- Mapping of certain features into Gaussian distribution
- Handling of temporal features


## Feature Selection:

- Correlation mapping was applied
- Lasso Regression was used for extracting key features
- 21/81 features were extracted


## Model Training:

- Train-test data splits were conducted
- XGBoost along with Random Forest Regressor was used as the classifier
- Model trained : r2_score: 86.5%
- Classifier saved as pickle file for faster execution speed in future.

### Files arrangement-

1. train.csv: original train csv file
2. engg_X_train.csv: New training data after feature engineering was conducted
3. selected_X_train.csv: Updated training data containing selected features only
4. y_train.csv: Contains the output('SalePrice') seperately



*******************************************************************************



