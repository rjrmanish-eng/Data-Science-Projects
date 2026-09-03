import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Load The Data
Data = pd.read_csv('Walmart.csv')
print(Data.head())

# Train Test Split
Data['Sales'] = pd.cut(Data['Weekly_Sales'],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])


split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(Data,Data['Sales']):
    strat_train_set = Data.loc[train_index].drop('Sales',axis=1)
    strat_test_set = Data.loc[test_index].drop('Sales',axis=1)

print('Train Data is :' , strat_train_set)    
print('Test Data is :' , strat_test_set)    



