# 1. import all reqiured libaries 

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score

# 2. load the dataset 
housing = pd.read_csv('housing.csv')
# print(housing)

# 3. creat a stratified train and test set 
housing['income_cat'] = pd.cut(housing['median_income'], 
                            bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                            labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2 , random_state=42)

# print(housing)

for train_index , test_index in split.split(housing,housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat' ,axis=1)
    strar_test_set = housing.loc[test_index].drop('income_cat',axis=1)


# 4. we will copy the train dataset 
housing= strat_train_set.copy()

# 5. saparate features and labels 
housing_labels = housing['median_house_value'].copy()
housing = housing.drop(['median_income'],axis=1)
# print(housing_labels)
# print(housing)


# 6. saparate numerical and catagorical columns 
num_cat = housing.drop(['ocean_proximity'] , axis=1).columns.tolist()
cat_attributes = ['ocean_proximity']
# print('cat attr',cat_attributes)

# 7. let's make pipeline 
# for numerical columns 
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ("scale" , StandardScaler())
])

# For catagorical 
cat_pipeline = Pipeline([
    ("onhotencode" ,OneHotEncoder(handle_unknown='ignore'))
])

# full pipeline 
full_pipeline = ColumnTransformer([
    ('num',num_pipeline,num_cat),
    ('cat',cat_pipeline,cat_attributes)
])

# 8. transform the data
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared.shape) 



# 9. Train The modal 
# linear reagression modal 
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds = lin_reg.predict(housing_prepared)   
# lin_rmse = root_mean_squared_error(housing_labels,lin_preds)
# print(f"The root mean squred error for Linear Regression is :{lin_rmse}")
lin_rmses = -cross_val_score(lin_reg,housing_prepared,housing_labels , scoring='neg_root_mean_squared_error' , cv=10)
print(pd.Series(lin_rmses).describe())


# Dicision tree Regression
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels,dec_preds)
# print(f"the root mean squred error for Dicision tree Regression is : {dec_rmse}")
dec_rmses = -cross_val_score(dec_reg,housing_prepared,housing_labels,scoring='neg_root_mean_squared_error',cv=10)
print(pd.Series(dec_rmses).describe())


# Random forest regrressor
random_foresst_reg = RandomForestRegressor()
random_foresst_reg.fit(housing_prepared,housing_labels)
random_foresst_preds = random_foresst_reg.predict(housing_prepared)
# random_foresst_rmse = root_mean_squared_error(housing_labels,random_foresst_preds)
# print(f"the root mean squred error for Random Forest Regressor is : {random_foresst_rmse}")

random_foresst_rmses = -cross_val_score(random_foresst_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(random_foresst_rmses).describe())