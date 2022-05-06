import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from train_test_split import TrainTestSplitFBMarketData
from clean_tabular import price_pipeline, basic_pipeline

# location for tabular
path_tabular = os.getcwd() + '/data/tabular/' + 'tab_data_raw.pkl'

# load the saved local df
with open(path_tabular, 'rb') as f:
    products_raw_df = pickle.load(f)

# split data into train and test
cat_level = 0
data_splitter = TrainTestSplitFBMarketData(product_cat_level=cat_level)
train_data, test_date = data_splitter.train_test_split(products_raw_df, 0.2)

# # VIEW DATA BEFORE ONEHOT ENCODING
# # print(basic_pipeline.get_params(deep=True))
# # set a category level
# # 0 for broader level; 1 for detailed level (only 2 levels)
# cat_level = 1
# basic_pipeline.set_params(cat_cleaner__cat_selected=cat_level)
# # set a location level
# # 1 for broader level (e.g. London); 0 for detailed level (some region of London)
# loc_level = 0
# basic_pipeline.set_params(loc_cleaner__cat_selected=loc_level)
# # do the transformation
# train_data_basic = basic_pipeline.fit_transform(train_data)
# cats = train_data_basic['category'].unique()
# locs = train_data_basic['location'].unique()
# print('cats', len(cats))
# print('locs', len(locs))
# train_data_basic['category'].value_counts().plot.bar()
# plt.show()
# train_data_basic['location'].value_counts().plot.bar()
# plt.show()
# # check for null values
# if train_data_basic.isna().sum().sum():
#     print(train_data_basic.isna().sum())
#     raise ValueError

# apply a pipleline transform to clean the training data
cat_level = 1   # retain lower level category
loc_level = 0   # retain higher level location
price_pipeline.set_params(common__cat_cleaner__cat_selected=cat_level)
basic_pipeline.set_params(loc_cleaner__cat_selected=loc_level)
train_data_tr = price_pipeline.fit_transform(train_data)
# check for null values
if train_data_tr.isna().sum().sum():
    raise ValueError
# prepare input and target data
price_labels = train_data_tr['price']
price_input_data = train_data_tr.drop(columns=['price'])

# LINEAR REGRESSION
lin_reg = LinearRegression()
lin_reg.fit(price_input_data, price_labels)
price_predictions = lin_reg.predict(price_input_data)

some_data = price_input_data.iloc[:5]
some_labels = price_labels.iloc[:5]
print("Lin predictions:", lin_reg.predict(some_data))
print("Labels:", list(some_labels))

lin_mse = mean_squared_error(price_labels, price_predictions)
lin_rmse = np.sqrt(lin_mse)
print('Lin training loss', lin_rmse)
# high scores indicate underfitting

scores = cross_val_score(lin_reg, price_input_data, price_labels,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)
print('Lin eval loss mean', lin_rmse_scores.mean())
print('Lin eval loss std', lin_rmse_scores.std())
"""
Lin predictions: [ -1045.31617623 -19227.66899571  -7360.00688317  -3822.95342425
   1404.57476709]
Labels: [10.0, 28.0, 5.0, 90.0, 399.0]
Lin training loss 134039.03736863643
Lin eval loss mean 2355243101993860.0
Lin eval loss std 4567985514823170.0
"""

# DECISION TREE REGRESSION
tree_reg = DecisionTreeRegressor()
tree_reg.fit(price_input_data, price_labels)
price_predictions = tree_reg.predict(price_input_data)

some_data = price_input_data.iloc[:5]
some_labels = price_labels.iloc[:5]
print("Tree predictions:", tree_reg.predict(some_data))
print("Labels:", list(some_labels))

tree_mse = mean_squared_error(price_labels, price_predictions)
tree_rmse = np.sqrt(tree_mse)
print('Tree training loss', tree_rmse)

scores = cross_val_score(tree_reg, price_input_data, price_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print('Tree eval loss mean', tree_rmse_scores.mean())
print('Tree eval loss std', tree_rmse_scores.std())
"""
Tree predictions: [ 10.          28.           5.         348.23809524 362.8       ]
Labels: [10.0, 28.0, 5.0, 90.0, 399.0]
Tree training loss 110360.3484883526
Tree eval loss mean 231879.65708459564
Tree eval loss std 133363.45006904562
"""

# RANDOM FOREST REGRESSION
forest_reg = RandomForestRegressor()
forest_reg.fit(price_input_data, price_labels)
price_predictions = forest_reg.predict(price_input_data)

some_data = price_input_data.iloc[:5]
some_labels = price_labels.iloc[:5]
print("forest predictions:", forest_reg.predict(some_data))
print("Labels:", list(some_labels))

forest_mse = mean_squared_error(price_labels, price_predictions)
forest_rmse = np.sqrt(forest_mse)
print('forest training loss', forest_rmse)

scores = cross_val_score(forest_reg, price_input_data, price_labels,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
print('forest eval loss mean', forest_rmse_scores.mean())
print('forest eval loss std', forest_rmse_scores.std())
"""
forest predictions: [ 16.146       39.31129762 284.47333333 339.95594987 374.92168146]
Labels: [10.0, 28.0, 5.0, 90.0, 399.0]
forest training loss 133744.78979108142
forest eval loss mean 244031.33719563982
forest eval loss std 92788.19774250945
"""

# SVR
param_grid = [
    {'kernel': ['linear'], 'C': [10., 30., 100.,
                                 300., 1000., 3000., ]},
    {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
     'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(price_input_data, price_labels)
svr_reg = grid_search.best_estimator_
some_data = price_input_data.iloc[:5]
some_labels = price_labels.iloc[:5]
print("svr predictions:", svr_reg.predict(some_data))
print("Labels:", list(some_labels))
print('best svr params', grid_search.best_params_)
svr_mse = mean_squared_error(price_labels, price_predictions)
svr_rmse = np.sqrt(svr_mse)
print('svr training loss', svr_rmse)

scores = cross_val_score(svr_reg, price_input_data, price_labels,
                         scoring="neg_mean_squared_error", cv=10)
svr_rmse_scores = np.sqrt(-scores)
print('svr eval loss mean', svr_rmse_scores.mean())
print('svr eval loss std', svr_rmse_scores.std())
"""
svr predictions: [10.09963297  28.10020618   5.09983482  59.90007281 398.90021607]
Labels: [10.0, 28.0, 5.0, 90.0, 399.0]
best svr params {'C': 1000.0, 'gamma': 1.0, 'kernel': 'rbf'}
svr training loss 133744.78979108142
svr eval loss mean 58360.41124740732
svr eval loss std 121779.8299752962
"""