import  pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# 1) Load the data

data = pd.read_csv("housing.csv")

# 2) Stratified shuffling and train_test_split

# Creating Strata...
data["income_cat"] = pd.cut(data["median_income"] , bins=[0 , 1.9 , 3.8, 5.7, 7.6 , np.inf] , labels=[1 , 2 , 3 , 4 , 5])

# An object of stratified class
split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)


for train_index, test_index in split.split(data , data["income_cat"]):
    strat_train_data = data.loc[train_index].drop("income_cat" , axis= 1)
    strat_test_data = data.loc[test_index].drop("income_cat" , axis= 1)

# 3) Now we'll work on training data( but in copy for future risk possibilities)

train_data = strat_train_data.copy()

#4) Separate features and labels from training data

labels_data = train_data["median_house_value"].copy()
features_data = train_data.drop("median_house_value" , axis= 1)

# 5) Now, we'll separate numerical and categorical features for different_typed_pipeline_steps

num_cols = features_data.drop("ocean_proximity", axis= 1).columns.tolist()
cat_cols = ['ocean_proximity']

# 6) Now Construct pipelines

# Pipeline logic(Numerical data preprocessing)
num_pipeline1 = Pipeline([
    ("Imputer" , SimpleImputer(strategy='median')),
    ("Scaling" , StandardScaler())
])  # An object of a pipeline...

# Pipeline logic (Categorical Data preprocessing)
cat_pipeline2 = Pipeline([
    ("Imputer" , SimpleImputer(strategy="most_frequent")),
    ("encoding", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
     ("Scaling" , StandardScaler())
])

# For merging categorical and numeric data
full_pipeline = ColumnTransformer([
    ('numeric', num_pipeline1, num_cols),
    ('category', cat_pipeline2,cat_cols),

])

housing_preprocessed = full_pipeline.fit_transform(strat_train_data)

# 7) Train ML Model (Choose Ml model based on performance)

# a) Linear regressor

lin_reg = LinearRegression()
lin_reg.fit(housing_preprocessed, labels_data)

# b) Decision_tree_regressor
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_preprocessed, labels_data)

# c) Random forest regressor..
random_reg = RandomForestRegressor(random_state=42)
random_reg.fit(housing_preprocessed, labels_data)

# d) Predict (Here, we're training and predicting on same data, which is wrong practice)
# That's why dec-rmse is too low, for eg. 0 (Overfitting)

lin_pred = lin_reg.predict(housing_preprocessed)
lin_rmse = root_mean_squared_error(labels_data, lin_pred)
print(f"The rmse of linear regression is: {lin_rmse}")

dec_pred = dec_reg.predict(housing_preprocessed)
dec_rmse = root_mean_squared_error(labels_data, dec_pred)
print(f"The rmse of Decision tree regression is: {dec_rmse}")
# rmse = 0 (100% accuracy)

random_pred = random_reg.predict(housing_preprocessed)
random_rmse = root_mean_squared_error(labels_data, random_pred)
print(f"The rmse of random forest regression is: {random_rmse}")

# e) Cross Validation for all 3 ml algo

lin_cross = -cross_val_score(lin_reg, housing_preprocessed, labels_data, scoring="neg_root_mean_squared_error" , cv=10)
print(pd.Series(lin_cross).describe())

dec_cross = -cross_val_score(dec_reg, housing_preprocessed, labels_data, scoring="neg_root_mean_squared_error" , cv=10)
print(pd.Series(dec_cross).describe())

random_cross = -cross_val_score(random_reg, housing_preprocessed, labels_data, scoring="neg_root_mean_squared_error" , cv=10)
print(pd.Series(random_cross).describe())