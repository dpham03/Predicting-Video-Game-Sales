import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


train = pd.read_csv("final_train.csv")
test = pd.read_csv("final_test.csv")

# ========================================================================== 
# Exploratory Data Analysis
# ========================================================================== 
# desc_stat = train.describe()
# train.info()
# print(desc_stat)

# # Visualize the distribution of the label "total_sales"
# train["total_sales"].hist(bins=50)
# plt.title("Distribution of Total Sales")
# plt.xlabel("Total Sales (in millions)")
# plt.ylabel("Frequency")
# plt.show()

# # Check for correlation between total sales and other features
# train.plot.scatter(x='critic_score', y='total_sales')
# plt.title("Relationship between Critic Score and Total Sales")
# plt.show()

# # Correlation heatmap for numerical variables
# corr_matrix = train.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title("Correlation Heatmap")
# plt.show()

# # Scatter matrix for key variables
# interest_vars = ['total_sales', 'critic_score', 'na_sales', 'jp_sales', 'pal_sales']
# scatter_matrix(train[interest_vars], figsize=(12, 8))
# plt.show()

# # Visualize distribution and skewness of numerical variables
# numerical_cols = train.select_dtypes(include=[np.number]).columns
# train[numerical_cols].hist(bins=50, figsize=(20, 15))
# plt.show()

# ========================================================================== 
# Data Pre-processing (Train)
# ========================================================================== 
# Check for missing values in the training dataset
count_missing_train = train.isnull().sum().sort_values(ascending=False)
percent_missing_train = (train.isnull().sum() / train.isnull().count() * 100).sort_values(ascending=False)
missing_data_train = pd.concat([count_missing_train, percent_missing_train],axis=1)
print(missing_data_train)

# Drop the 'last_update' column as it is not useful for prediction and has many missing values
train.drop("last_update", axis=1, inplace=True)
test.drop("last_update", axis=1, inplace=True)  # Drop the same column in test set

# Identify categorical and numerical columns
categorical_cols = ['console', 'genre']  # Only 'console' and 'genre' are categorical
numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()

# Remove the target variable 'total_sales' from numerical columns
numerical_cols.remove('total_sales')

# Impute missing numerical values with the median
imputer_median = SimpleImputer(strategy="median")
train[numerical_cols] = imputer_median.fit_transform(train[numerical_cols])
test[numerical_cols] = imputer_median.transform(test[numerical_cols])  # Use the same imputer on the test set

# Impute missing categorical values with the most frequent category
train[categorical_cols] = train[categorical_cols].fillna(train[categorical_cols].mode().iloc[0])
test[categorical_cols] = test[categorical_cols].fillna(test[categorical_cols].mode().iloc[0])

# ========================================================================== 
# FEATURE ENGINEERING (Train and Test)
# ========================================================================== 
# Combine region sales into a single feature for both train and test sets
train['total_region_sales'] = train['na_sales'] + train['jp_sales'] + train['pal_sales'] + train['other_sales']
test['total_region_sales'] = test['na_sales'] + test['jp_sales'] + test['pal_sales'] + test['other_sales']

# Handle Skewed Numerical Features (Apply log transformation)
skewness = train[numerical_cols].skew().sort_values(ascending=False)
skewed_vars = skewness.index[skewness >= 1].to_list()

# Apply log transformation to skewed variables in both train and test sets
train[skewed_vars] = np.log1p(train[skewed_vars])
test[skewed_vars] = np.log1p(test[skewed_vars])

# ========================================================================== 
# OneHot Encoding for Categorical Variables (Train and Test)
# ========================================================================== 
# Use OneHotEncoder for non-ordinal categorical variables
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Fit on the training set and transform both training and test sets
train_encoded = pd.DataFrame(ohe.fit_transform(train[categorical_cols]))
train_encoded.columns = ohe.get_feature_names_out(categorical_cols)

test_encoded = pd.DataFrame(ohe.transform(test[categorical_cols]))
test_encoded.columns = ohe.get_feature_names_out(categorical_cols)

# Concatenate the encoded categorical features with the numerical data (Train and Test)
train_cleaned = pd.concat([train[numerical_cols], train_encoded], axis=1)
test_cleaned = pd.concat([test[numerical_cols], test_encoded], axis=1)

# ========================================================================== 
# Model Training and Hyperparameter Tuning (Grid Search) (run once to find params)
# ========================================================================== 

# Use the already initialized cleaned training dataset
x = train_cleaned  # Features
y = train['total_sales']  # Target
# # Set up the RandomForestRegressor model
# rf_model = RandomForestRegressor(random_state=42)

# # Define the hyperparameters to tune
# param_grid = {
#     'n_estimators': [100, 200, 300],  # Number of trees in the forest
#     'max_depth': [None, 10, 20, 30],  # Depth of each tree
#     'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
#     'min_samples_leaf': [1, 2, 4]     # Minimum samples required to be at a leaf node
# }

# # Set up GridSearchCV to search for the best hyperparameters
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1)

# # Fit the grid search to the training data
# grid_search.fit(x, y)

# # Get the best model from the grid search
# rf_model = grid_search.best_estimator_

# # Print the best hyperparameters found by GridSearchCV
# print("Best parameters:", grid_search.best_params_)

# After running grid search, the below parameters were found to be optimal
# To run the GridSearchCV, uncomment the above code block and uncomment the rf_model line beneath and run it once
rf_model = RandomForestRegressor(
    max_depth=10, 
    min_samples_leaf=4, 
    min_samples_split=10, 
    n_estimators=100,
    random_state=42
)
# ========================================================================== 
# Model Prediction
# ========================================================================== 

rf_model.fit(x, y)  # Train on the entire training dataset

# Make predictions on the final test dataset
# Prepare the test dataset (already cleaned and transformed)
X_final_test = test_cleaned  # Test features (excluding target)

# Predict on the final test dataset
y_final_pred = rf_model.predict(X_final_test)

# Save predictions to a CSV file for submission
final_predictions = pd.DataFrame({'product_id': test['product_id'], 'total_sales': y_final_pred})

final_predictions.to_csv("prediction.csv", index=False)

print("Predictions saved to 'prediction.csv'")